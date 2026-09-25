"""
Parses the structured JSON the LLM is now asked to produce (see
configs/config.yaml: llm.prompt_template) and builds `combined_text` - the
text that actually gets embedded downstream.

The prompt is stance-neutral ("what do you think about this claim?") and lets
the model return ONE viewpoint or SEVERAL - one JSON object per distinct stance
it holds - in any of these equivalent forms:
    {"stance": ..., "summary": ..., "reasoning": ...}
    {"stance": ..., ...} {"stance": ..., ...}          (concatenated objects)
    [{"stance": ..., ...}, {"stance": ..., ...}]        (top-level array)
    {"viewpoints": [{"stance": ..., ...}, ...]}         (wrapped array)
Each object gets exploded into its own row here, before embedding, so
downstream code (embeddings, similarity, coverage, diversity) sees one row per
viewpoint and doesn't need to know this happened.

Both key namings (summary/reasoning and perspective_summary/core_reasoning)
are accepted per object. Also recovers bold-markdown
answers (e.g. "**Stance:** Support\\n**Summary:** ...") for models that
occasionally skip JSON entirely and answer in labeled prose instead.

READS from data/llm_outputs/<model>/raw_generations.csv (written by
generate_llama.py - fixed 3-column schema: claim_id, sample_id,
generated_text) and WRITES the full reshaped output to
config["paths"]["llm_outputs_processed"] (data/llm_outputs/<model>/
processed_generations.csv). These are deliberately different files: this
function fully rebuilds the processed CSV from the raw CSV every time it
runs, rather than reading-and-overwriting the same file generate_llama.py
appends to - see generate_llama.py's module docstring for why that
separation matters (it avoids a column-misalignment bug when generation is
resumed after postprocessing has already run once).

Columns in the output CSV:
  viewpoint_id           - 0-indexed position of this viewpoint within its
                            generation call (0 for old single-object schema)
  llm_reported_stance    - self-reported "support"/"undermine"/None if unparsed
  perspective_summary    - the summary field (or full raw text if parsing failed)
  core_reasoning          - the reasoning field (empty string if parsing failed)
  combined_text           - summary + reasoning; THIS is what gets embedded
  parse_ok                - bool, whether some structured answer was recovered
  parse_method            - "json" | "markdown_fallback" | "unparsed" - how it
                            was recovered, for diagnosing model behavior

Also does light cleanup:
  - strips boilerplate prefixes / markdown code fences / <think> blocks
  - drops empty/too-short generations
  - de-duplicates exact repeats of combined_text within the same claim
"""
import json
import os
import re
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, get_model_tag
from src.utils.logging_utils import get_logger

logger = get_logger("postprocess_generations")

BOILERPLATE_PATTERNS = [
    r"^sure[,!]?\s*",
    r"^here('s| is)[^:]*:\s*",
    r"^certainly[,!]?\s*",
    r"^as an ai( language model)?,?\s*",
]

VALID_STANCES = {"support", "undermine"}

# Matches fenced ```json ... ``` or ``` ... ``` blocks, so we can pull the
# JSON out even if the model wraps it in markdown. Captures the whole fence
# body (not just one object) because the model may put several concatenated
# objects or a top-level array inside a single fence.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Some models answer in bold-markdown prose instead of JSON, e.g.:
#   **Stance:** Support
#   **Summary:** ...
#   **Core Reasoning:** ...
# or wrapped in a **Response:**/**Answer:** header with no per-field labels
# at all. These patterns pull out whatever labeled fields are present so
# that answer isn't thrown away just because it isn't valid JSON.
_MD_STANCE_RE = re.compile(r"\*\*\s*stance\s*:?\s*\*\*\s*:?\s*([a-zA-Z]+)", re.IGNORECASE)
_MD_SUMMARY_RE = re.compile(
    r"\*\*\s*summary\s*:?\s*\*\*\s*:?\s*(.*?)(?=\*\*\s*core[ _]?reasoning|\*\*\s*stance|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_MD_REASONING_RE = re.compile(
    r"\*\*\s*core[ _]?reasoning\s*:?\s*\*\*\s*:?\s*(.*?)(?=\*\*\s*stance|\*\*\s*summary|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# A generic wrapper header with no per-field labels underneath, e.g.
# "**Response:**\n<free text>" - not itself a field, just noise to strip
# before falling back to treating the remainder as the summary.
_MD_WRAPPER_HEADER_RE = re.compile(
    r"^\*\*\s*(response|answer)\s*:?\s*\*\*\s*:?\s*", re.IGNORECASE
)

# Some models emit a chain-of-thought block before the actual answer. Some
# chat templates insert the OPENING <think> tag as part of the
# prompt itself via add_generation_prompt=True, so it's already consumed
# before generation starts and never appears in the model's own output -
# only the CLOSING </think> tag shows up (if the model finished thinking
# before hitting max_new_tokens). So we can't rely on matching a
# <think>...</think> pair; we look for a closing tag on its own and strip
# everything up to and including it.
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
# Some setups (or other model families) do emit a full <think>...</think>
# pair in the output - handle that case too.
_THINK_PAIR_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_think_block(text: str) -> str:
    if _THINK_PAIR_RE.search(text):
        return _THINK_PAIR_RE.sub("", text).strip()
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if matches:
        # Strip everything up to and including the LAST closing tag (in case
        # the reasoning trace itself contains stray "</think"-like text).
        return text[matches[-1].end():].strip()
    # No closing tag at all -> if this looks like unterminated reasoning
    # (i.e. no '{' anywhere, so there's nothing to recover), leave it as-is;
    # the caller will fail to find JSON and fall back appropriately, and
    # main()'s diagnostic below flags this pattern specifically.
    return text


def _strip_boilerplate(text: str) -> str:
    cleaned = _strip_think_block(text)
    cleaned = _MD_WRAPPER_HEADER_RE.sub("", cleaned)
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> list:
    """Collect every top-level balanced span in `text` delimited by
    `open_ch`/`close_ch` (used for {...} objects and [...] arrays). Returns
    (start, end, span) triples with character offsets into `text`.

    String-aware: braces/brackets inside double-quoted strings (quoted claims,
    scare quotes in summaries) and backslash escapes don't affect depth, so a
    summary containing e.g. '{note}' can't corrupt the split. Unterminated
    trailing spans (truncated generation) are skipped, not returned."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        start = text.find(open_ch, i)
        if start == -1:
            break
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for j in range(start, n):
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            # No balanced close from here on (truncated tail) - stop scanning
            # rather than mis-splitting the remainder.
            break
        spans.append((start, end, text[start:end + 1]))
        i = end + 1
    return spans


def _extract_json_objects(text: str) -> list:
    """Best-effort extraction of ALL JSON values from a raw LLM completion,
    tolerating markdown fences and leading/trailing chatter. Returns a list of
    parsed Python objects (dicts and/or lists) - one entry per top-level JSON
    value that parsed cleanly. Understands every form the neutral prompt permits:
      - one bare object:              {...}
      - several concatenated objects: {...} {...} ...
      - a top-level array:            [{...}, {...}]
    (plus the legacy {"viewpoints": [...]} wrapper, handled by the caller).
    Fenced code blocks take precedence over surrounding text, matching the old
    single-object behavior; every fence is scanned, not just the first.

    A span strictly contained in another span that parsed successfully is
    dropped (the outer parse already covers it) - e.g. the [{...},{...}]
    array suppresses its inner objects, and a {"viewpoints": [...]} wrapper
    suppresses its inner array. If the outer span fails to parse, the inner
    spans survive and are tried individually."""
    fence_bodies = _JSON_FENCE_RE.findall(text)
    haystacks = fence_bodies if fence_bodies else [text]

    kept = []
    seen_keys = set()
    # NOTE: containment is evaluated separately within each haystack, since
    # offsets from different fence bodies are not comparable.
    for hay in haystacks:
        spans = _scan_balanced(hay, "{", "}") + _scan_balanced(hay, "[", "]")
        found = []  # (start, end, parsed_obj) in this haystack's coordinates
        for start, end, span in spans:
            try:
                obj = json.loads(span)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, (dict, list)):
                found.append((start, end, obj))

        # Suppress inner spans covered by a successfully parsed outer span, and
        # drop exact-duplicate spans (model repeating the same JSON twice).
        for start, end, obj in found:
            covered = any(
                o_start <= start and end <= o_end and (o_start, o_end) != (start, end)
                for o_start, o_end, _ in found
            )
            if covered:
                continue
            key = json.dumps(obj, sort_keys=True)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kept.append(obj)
    return kept


def _normalize_viewpoint(obj: dict) -> dict:
    """Pulls stance/summary/reasoning out of one viewpoint dict, accepting
    both the new key names (summary/reasoning) and the old ones
    (perspective_summary/core_reasoning) for backward compatibility."""
    stance = str(obj.get("stance", "")).strip().lower()
    summary = str(obj.get("summary", obj.get("perspective_summary", ""))).strip()
    reasoning = str(obj.get("reasoning", obj.get("core_reasoning", ""))).strip()
    if stance not in VALID_STANCES:
        stance = None
    return {"llm_reported_stance": stance, "perspective_summary": summary, "core_reasoning": reasoning}


def _parse_markdown_labels(text: str) -> dict:
    """Fallback for models that answer in bold-markdown prose instead of
    JSON (e.g. '**Stance:** Support\\n
    **Summary:** ...\\n**Core Reasoning:** ...'). Returns a normalized
    viewpoint dict if at least a summary or reasoning field could be
    recovered, else None so the caller can fall through to the raw-text
    fallback instead of fabricating an empty answer.
    """
    stance_match = _MD_STANCE_RE.search(text)
    summary_match = _MD_SUMMARY_RE.search(text)
    reasoning_match = _MD_REASONING_RE.search(text)

    if not summary_match and not reasoning_match:
        return None  # no labeled fields at all - nothing to recover here

    stance = stance_match.group(1).strip().lower() if stance_match else None
    if stance not in VALID_STANCES:
        stance = None
    summary = summary_match.group(1).strip() if summary_match else ""
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if not summary and not reasoning:
        return None

    return {"llm_reported_stance": stance, "perspective_summary": summary, "core_reasoning": reasoning}


def parse_structured_output(raw_text: str) -> list:
    """Parse the LLM's JSON output into a list of one-or-more viewpoint dicts:
    [{"llm_reported_stance", "perspective_summary", "core_reasoning", "parse_ok"}, ...]

    Handles every form the neutral prompt permits (see _extract_json_objects):
      - one bare object -> a single-element list
      - several concatenated objects -> one dict per object
      - top-level array [{...}, {...}] -> one dict per element
      - {"viewpoints": [{...}, ...]} wrapper -> one dict per entry
      - anything that fails to parse -> a single-element list with parse_ok=False,
        falling back to the raw text as the summary so no data is silently dropped
    """
    cleaned = _strip_boilerplate(raw_text)

    viewpoints = []
    for obj in _extract_json_objects(cleaned):
        # A top-level array's elements are individual viewpoint objects.
        candidates = obj if isinstance(obj, list) else [obj]
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            # Unwrap the legacy {"viewpoints": [...]} container if present.
            if isinstance(cand.get("viewpoints"), list):
                entries = [vp for vp in cand["viewpoints"] if isinstance(vp, dict)]
            else:
                entries = [cand]
            for vp in entries:
                norm = _normalize_viewpoint(vp)
                if norm["perspective_summary"]:  # require at least a summary
                    norm["parse_ok"] = True
                    norm["parse_method"] = "json"
                    viewpoints.append(norm)
    if viewpoints:
        return viewpoints

    # No valid JSON found (or none of it contained a usable answer) - try
    # recovering a bold-markdown-labeled answer before giving up entirely.
    md_result = _parse_markdown_labels(cleaned)
    if md_result is not None:
        md_result["parse_ok"] = True
        md_result["parse_method"] = "markdown_fallback"
        return [md_result]

    # Final fallback: nothing recoverable. Strip fence debris first, so a
    # model that emitted only empty fences (e.g. ```json [] ```) falls back
    # to a short/empty summary that the downstream length filter drops,
    # instead of surviving as garbage text made of fence markers.
    fallback_text = _JSON_FENCE_RE.sub("", cleaned).strip()
    return [{
        "llm_reported_stance": None,
        "perspective_summary": fallback_text,
        "core_reasoning": "",
        "parse_ok": False,
        "parse_method": "unparsed",
    }]


def build_combined_text(summary: str, reasoning: str) -> str:
    """The text that gets embedded. Concatenating summary + reasoning keeps
    both the stated viewpoint and its justification, rather than the single
    compressed sentence the old pipeline embedded."""
    parts = [p.strip() for p in (summary, reasoning) if p and p.strip()]
    return " ".join(parts)


def build_quality_report(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    """Per-claim generation quality: how many generation calls actually yielded
    usable viewpoints. A call "fails" when none of its parsed viewpoints
    survive cleanup (e.g. degenerate `[]` outputs) - those samples silently
    vanish from embeddings/analysis unless counted here.

    pre_df  - exploded viewpoints BEFORE length/dedup cleanup (needs
              claim_id, sample_id, parse_ok)
    post_df - viewpoint rows AFTER cleanup (needs claim_id, sample_id)
    """
    calls = pre_df.drop_duplicates(["claim_id", "sample_id"])
    n_calls = calls.groupby("claim_id").size().rename("n_calls")
    n_raw = pre_df.groupby("claim_id").size().rename("n_viewpoints_raw")
    parse_ok_rate = pre_df.groupby("claim_id")["parse_ok"].mean().rename("parse_ok_rate")

    usable_keys = set(zip(post_df["claim_id"], post_df["sample_id"]))
    zero_usable = (
        calls[~calls.apply(lambda r: (r["claim_id"], r["sample_id"]) in usable_keys, axis=1)]
        .groupby("claim_id").size().rename("calls_zero_usable")
    )
    n_usable = post_df.groupby("claim_id").size().rename("n_usable")

    report = pd.concat([n_calls, n_raw, parse_ok_rate, n_usable], axis=1)
    report["n_usable"] = report["n_usable"].fillna(0).astype(int)
    report["calls_zero_usable"] = zero_usable.reindex(report.index, fill_value=0).astype(int)
    report["usable_per_call"] = report["n_usable"] / report["n_calls"]
    return report.reset_index()


def main(config: dict = None):
    if config is None:
        config = load_config()

    # Read from the RAW CSV (fixed 3-column schema, append-only, written by
    # generate_llama.py) rather than reading-and-overwriting
    # llm_outputs_processed in place - see module docstring for why.
    raw_dir = resolve(config["paths"]["llm_outputs_raw"])
    model_tag = get_model_tag(config["llm"]["model"])
    raw_path = os.path.join(raw_dir, model_tag, "raw_generations.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run src/generation/generate_llama.py first."
        )
    out_path = resolve(config["paths"]["llm_outputs_processed"])

    df = load_csv(raw_path)

    exploded_rows = []
    for _, row in df.iterrows():
        viewpoints = parse_structured_output(row["generated_text"])
        for vp_id, vp in enumerate(viewpoints):
            exploded_rows.append({
                "claim_id": row["claim_id"],
                "sample_id": row["sample_id"],
                "viewpoint_id": vp_id,
                "generated_text": row["generated_text"],
                **vp,
            })
    df = pd.DataFrame(exploded_rows)
    if df.empty:
        # raw_generations.csv exists but has no rows (e.g. a degraded run).
        # Save an empty processed CSV with the full expected schema and stop
        # here - the stats, quality report, and cleanup below all assume
        # populated columns and would KeyError. Downstream stages will raise
        # their own clear errors on the empty file if run against it.
        logger.warning(
            f"{raw_path} has no rows - nothing to postprocess. Saving an empty "
            f"{out_path} with the expected schema."
        )
        empty = pd.DataFrame(columns=[
            "claim_id", "sample_id", "viewpoint_id", "generated_text",
            "llm_reported_stance", "perspective_summary", "core_reasoning",
            "combined_text", "parse_ok", "parse_method",
        ])
        save_csv(empty, out_path)
        logger.warning(
            "generation_quality.csv was NOT written (no generation calls to score)."
        )
        return empty
    df["combined_text"] = df.apply(
        lambda r: build_combined_text(r["perspective_summary"], r["core_reasoning"]), axis=1
    )

    n_total = len(df)
    n_parse_failed = int((~df["parse_ok"]).sum())
    method_counts = df["parse_method"].value_counts().to_dict()
    logger.info(f"Parse method breakdown: {method_counts}")
    if n_parse_failed:
        # Since the opening <think> tag is typically part of the prompt (not
        # the generated output - see _strip_think_block above), we can't
        # detect truncated reasoning by looking for an unclosed tag. Instead:
        # a failed row with no closing </think> tag AND no '{' anywhere is
        # almost certainly generation cut off mid-thought before it ever
        # reached the JSON answer.
        failed = df.loc[~df["parse_ok"], "generated_text"]
        looks_truncated_reasoning = (
            ~failed.str.contains(r"</think(?:ing)?>", case=False, regex=True, na=False)
            & ~failed.str.contains(r"\{", regex=True, na=False)
        )
        n_truncated = int(looks_truncated_reasoning.sum())
        if n_truncated:
            logger.warning(
                f"{n_truncated}/{n_parse_failed} of the failed-to-parse rows have no closing "
                f"</think> tag and no '{{' at all - almost certainly generation cut off "
                f"mid-reasoning before reaching the JSON answer. If this count is high, increase "
                f"llm.max_new_tokens (or the model's entry under llm.models) further."
            )
        pct = 100 * n_parse_failed / n_total
        logger.warning(
            f"{n_parse_failed}/{n_total} ({pct:.1f}%) viewpoints did not parse as valid "
            f"structured JSON - falling back to raw text for those samples. If this rate "
            f"is high, revisit the prompt template or add few-shot JSON examples."
        )

    n_generations = df.groupby(["claim_id", "sample_id"]).ngroups
    logger.info(
        f"Parsed {n_generations} generation calls into {n_total} viewpoint rows "
        f"({n_total / n_generations:.1f} viewpoints/call on average)"
    )
    pre_df = df  # exploded pre-cleanup snapshot for the quality report below

    before = len(df)
    df = df[df["combined_text"].str.len() > 10]  # drop empty/near-empty
    df = df.drop_duplicates(subset=["claim_id", "combined_text"])
    after = len(df)

    # Quality gate: count, per claim, how many generation calls yielded zero
    # usable viewpoints (degenerate outputs that silently vanish downstream).
    # Saved to the metrics dir so it lands next to the other quality numbers
    # and gets rolled into final_report_summary.json.
    quality_df = build_quality_report(pre_df, df)
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    save_csv(quality_df, os.path.join(metrics_dir, "generation_quality.csv"))
    n_zero_calls = int(quality_df["calls_zero_usable"].sum())
    mean_usable_per_call = float(quality_df["usable_per_call"].mean())
    if n_zero_calls:
        logger.warning(
            f"QUALITY GATE: {n_zero_calls}/{n_generations} generation calls produced "
            f"ZERO usable viewpoints (see generation_quality.csv for the per-claim "
            f"breakdown). Coverage for those claims is computed on fewer samples "
            f"than num_samples_per_claim - consider a rerun; resume alone will "
            f"not regenerate them."
        )
    logger.info(f"Mean usable viewpoints per generation call: {mean_usable_per_call:.2f}")

    logger.info(f"Postprocessing: {before} -> {after} rows after cleanup/dedup")
    logger.info(
        f"Structured-parse success rate: {(n_total - n_parse_failed) / n_total:.1%} "
        f"({n_total - n_parse_failed}/{n_total})"
    )
    save_csv(df, out_path)
    return df


if __name__ == "__main__":
    main()