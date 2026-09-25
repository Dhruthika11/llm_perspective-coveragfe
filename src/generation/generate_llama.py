"""
Generate k sampled opinions per claim from the configured LLM, and save
both per-claim raw JSON files and a raw CSV.

Claims are generated in BATCHES of `generation.batch_size` claims (default
10 - override via config["generation"]["batch_size"]). All claims in a
batch are tokenized together, left-padded to a common length, and decoded
in a single model.generate() call with num_return_sequences=k, so the GPU
processes batch_size * k sequences per step instead of k - roughly a
batch_size-fold speedup over the old one-claim-at-a-time loop, which left
the GPU mostly idle on single-digit-batch decode steps. Memory use scales
with batch_size * k, so if a larger model OOMs, lower batch_size.

Results are checkpointed in batches (every `checkpoint_every_n_claims`
claims, default 10 - override via config["generation"]["checkpoint_every_n_claims"])
rather than only once at the very end. This means a crash/disconnect
partway through a long run doesn't lose everything generated so far - just
re-run and it picks up where it left off, skipping claims already done.
Because rows only reach the CSV at checkpoint flushes, a crash can land
between a per-claim JSON write and the next flush; _reconcile_raw_csv runs
at startup and backfills the CSV from the per-claim JSONs, so no claim that
resume logic considers "done" can be silently missing downstream.

IMPORTANT: this module writes to a RAW CSV
(data/llm_outputs/<model>/raw_generations.csv), NOT the final
processed_generations.csv that postprocess_generations.py produces.
Resume/append logic here only ever touches this raw file, whose schema
(claim_id, sample_id, generated_text) never changes. postprocess_generations.py
reads this raw file and writes the reshaped, multi-column, exploded-viewpoint
output to a *different* path (config["paths"]["llm_outputs_processed"]),
which it's free to fully rebuild every time it runs.

Why this separation matters: an earlier version of this pipeline had
generate_llama.py append raw rows directly into the same file
postprocess_generations.py reshapes in place (adding viewpoint_id,
combined_text, etc. and exploding one row per viewpoint). If generation was
ever resumed *after* postprocessing had already run once, new 3-column raw
rows got appended under what was now a 10-column header, silently
misaligning every column in the file. Keeping raw and processed as
separate files makes that class of bug impossible: raw_generations.csv's
schema is fixed and only ever appended to by this module, and
processed_generations.csv is always a full rebuild, never a partial append.

Usage:
    python -m src.generation.generate_llama
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, get_model_tag
from src.utils.logging_utils import get_logger
from src.generation.llm_config import get_llm_client

logger = get_logger("generate_llama")

RAW_GENERATIONS_FILENAME = "raw_generations.csv"


def _flush_batch(rows: list, out_path: str, write_header: bool) -> None:
    """Append a batch of rows to the raw CSV (creating it with a header on
    the first flush, appending without a header afterward). This file's
    schema is fixed (claim_id, sample_id, generated_text) and this function
    is the ONLY thing that ever writes to it, so appends are always safe."""
    if not rows:
        return
    batch_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    batch_df.to_csv(out_path, mode="a", index=False, header=write_header)


def _rows_from_claim_json(model_dir: str, claim_id: int) -> list:
    """Read one per-claim raw JSON and flatten it into raw CSV rows
    (claim_id, sample_id, generated_text). Returns [] (with a logged
    warning) if the file is unreadable or malformed - a corrupt JSON should
    never crash a resume, it just leaves that claim unrecovered."""
    json_path = os.path.join(model_dir, f"{claim_id}_samples.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return [
            {
                "claim_id": claim_id,
                "sample_id": i,
                "generated_text": text,
            }
            for i, text in enumerate(data.get("samples", []))
        ]
    except Exception as e:
        logger.warning(f"Could not read/parse {json_path}: {e}")
        return []


def _reconcile_raw_csv(model_dir: str, raw_csv_path: str, done_claim_ids: set) -> None:
    """Backfill raw_generations.csv from the per-claim JSONs so every claim
    resume logic considers "done" actually has its rows in the CSV.

    Rows only reach the CSV at checkpoint flushes (every
    checkpoint_every_n_claims claims), so a crash between a per-claim JSON
    write and the next flush would otherwise leave that claim silently
    missing from everything downstream (postprocess/embeddings/analysis all
    read the CSV, not the JSONs). Running this at startup guarantees the
    CSV covers every done claim before generation begins.

    Three cases:
      - CSV missing            -> full rebuild from ALL done-claim JSONs.
      - CSV present, claims missing from it -> append just those claims' rows.
      - CSV unparseable/corrupt -> warn loudly and leave it alone (delete the
        file to trigger a full rebuild on the next run); never rewrite it.
    """
    if not done_claim_ids:
        return

    if not os.path.exists(raw_csv_path):
        rows = []
        for claim_id in sorted(done_claim_ids):
            rows.extend(_rows_from_claim_json(model_dir, claim_id))
        if rows:
            _flush_batch(rows, raw_csv_path, write_header=True)
        logger.info(
            f"raw_generations.csv was missing - reconstructed {len(rows)} rows "
            f"from {len(done_claim_ids)} per-claim JSONs."
        )
        return

    # CSV exists: find claims whose JSON exists but whose rows are absent.
    try:
        existing = pd.read_csv(raw_csv_path, usecols=["claim_id"], dtype={"claim_id": int})
        csv_claim_ids = set(existing["claim_id"].dropna())
    except Exception as e:
        logger.warning(
            f"Could not read {raw_csv_path} ({e}) - it may be corrupt. Delete it "
            f"and re-run to reconstruct the CSV from the per-claim JSONs."
        )
        return

    missing = sorted(done_claim_ids - csv_claim_ids)
    if not missing:
        return

    rows = []
    for claim_id in missing:
        rows.extend(_rows_from_claim_json(model_dir, claim_id))
    if rows:
        _flush_batch(rows, raw_csv_path, write_header=False)
    logger.info(
        f"Backfilled {len(missing)} claim(s) ({len(rows)} rows) into {raw_csv_path} "
        f"that had JSONs but no CSV rows (lost to a crash between JSON write and "
        f"checkpoint flush)."
    )


def main(config: dict = None):
    if config is None:
        config = load_config()

    # Reproducible sampling: seed every RNG that sampling draws from. HF
    # generate() without an explicit generator consumes torch's global RNG in
    # call order, which is fixed (claims generated sequentially in batches),
    # so one seeding here reproduces the whole run for a fixed
    # model/prompt/software stack. Groq-side seeding is handled by the client
    # (per-call derived seeds: base seed + crc32(prompt) + sample_index, so
    # the k samples of a prompt don't collapse to k identical completions).
    seed = config["llm"].get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass  # torch only exists where the HF stack is installed
        logger.info(f"[{get_model_tag(config['llm']['model'])}] Seeded RNGs with {seed}")

    processed_dir = resolve(config["dataset"]["processed_dir"])
    prompts_path = os.path.join(processed_dir, "prompts.csv")

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(
            f"{prompts_path} not found. Run src/data_prep/build_prompts.py first."
        )

    prompts_df = load_csv(prompts_path)
    client = get_llm_client(config)

    k = config["llm"]["num_samples_per_claim"]
    temperature = config["llm"]["temperature"]
    top_p = config["llm"]["top_p"]
    max_new_tokens = config["llm"]["max_new_tokens"]
    generation_cfg = config.get("generation", {})
    checkpoint_every_n_claims = generation_cfg.get("checkpoint_every_n_claims", 10)
    # How many claims to decode together in one model.generate() call.
    # batch_size * k sequences are decoded per step - 10 claims x k=5
    # samples = 50 concurrent decodes vs 5 in the old per-claim loop.
    # Lower this if a bigger model OOMs; raise it if the GPU has headroom.
    batch_size = generation_cfg.get("batch_size", 10)

    raw_dir = resolve(config["paths"]["llm_outputs_raw"])
    model_tag = get_model_tag(config["llm"]["model"])
    model_dir = os.path.join(raw_dir, model_tag)
    os.makedirs(model_dir, exist_ok=True)

    # NOTE: this is the RAW output path, not config["paths"]["llm_outputs_processed"].
    # See module docstring for why these must be different files.
    raw_csv_path = os.path.join(model_dir, RAW_GENERATIONS_FILENAME)

    # Resume support: a claim counts as "already done" if its per-claim raw
    # JSON file exists. That file is written exactly once per claim by this
    # module and never touched again by anything else, so it's a safe,
    # unambiguous source of truth for what's been generated - independent of
    # whatever state raw_generations.csv or processed_generations.csv are in.
    already_done_claim_ids = {
        int(row["claim_id"]) for _, row in prompts_df.iterrows()
        if os.path.exists(os.path.join(model_dir, f"{row['claim_id']}_samples.json"))
    }

    # Reconcile BEFORE anything else: guarantee every "done" claim (JSON on
    # disk) has its rows in the raw CSV, regardless of where the last
    # checkpoint flush left it. Handles the missing-CSV rebuild, the
    # partial-CSV backfill, and warns on a corrupt CSV without touching it.
    _reconcile_raw_csv(model_dir, raw_csv_path, already_done_claim_ids)
    raw_csv_exists = os.path.exists(raw_csv_path)
    if already_done_claim_ids:
        logger.info(
            f"[{model_tag}] Found {len(already_done_claim_ids)} claims with existing raw "
            f"JSON output - resuming, skipping those."
        )

    remaining_df = prompts_df[~prompts_df["claim_id"].isin(already_done_claim_ids)]
    if len(remaining_df) == 0:
        logger.info(f"[{model_tag}] All {len(prompts_df)} claims already generated - nothing to do.")
        return load_csv(raw_csv_path) if raw_csv_exists else pd.DataFrame(
            columns=["claim_id", "sample_id", "generated_text"]
        )

    pending_rows = []
    claims_since_flush = 0
    header_needed = not raw_csv_exists  # only write header if the file doesn't already exist
    n_generated_total = len(already_done_claim_ids)

    remaining_records = remaining_df.to_dict("records")
    n_batches = (len(remaining_records) + batch_size - 1) // batch_size
    logger.info(
        f"[{model_tag}] Generating for {len(remaining_records)} claims in batches of "
        f"{min(batch_size, len(remaining_records))} ({k} samples each -> "
        f"{min(batch_size, len(remaining_records)) * k} sequences per generate call)"
    )

    for batch_start in tqdm(
        range(0, len(remaining_records), batch_size),
        total=n_batches,
        desc=f"Generating ({model_tag}, batch_size={batch_size} claims)",
    ):
        batch = remaining_records[batch_start:batch_start + batch_size]

        # THE core speedup: all claims in `batch` are tokenized, padded,
        # and decoded together in one model.generate() call, producing
        # batch_size * k sequences in a single GPU-busy pass instead of
        # batch_size sequential single-claim calls.
        batch_samples = client.generate_k_samples_batch(
            [r["prompt"] for r in batch], k=k, temperature=temperature,
            top_p=top_p, max_new_tokens=max_new_tokens,
        )

        for row, samples in zip(batch, batch_samples):
            claim_id = row["claim_id"]

            # Save raw per-claim JSON. This is what resume checks against, and
            # is also handy for debugging / re-running analysis without
            # re-generating.
            raw_path = os.path.join(model_dir, f"{claim_id}_samples.json")
            with open(raw_path, "w") as f:
                json.dump({"claim_id": int(claim_id), "claim_text": row["claim_text"],
                           "samples": samples}, f, indent=2)

            for i, text in enumerate(samples):
                pending_rows.append({
                    "claim_id": claim_id,
                    "sample_id": i,
                    "generated_text": text,
                })

            claims_since_flush += 1
            n_generated_total += 1

        if claims_since_flush >= checkpoint_every_n_claims:
            _flush_batch(pending_rows, raw_csv_path, write_header=header_needed)
            logger.info(
                f"[{model_tag}] Checkpointed {len(pending_rows)} rows "
                f"({n_generated_total}/{len(prompts_df)} claims done) -> {raw_csv_path}"
            )
            pending_rows = []
            claims_since_flush = 0
            header_needed = False

    # Flush whatever's left over that didn't fill a full checkpoint batch.
    if pending_rows:
        _flush_batch(pending_rows, raw_csv_path, write_header=header_needed)
        logger.info(
            f"[{model_tag}] Final checkpoint: {len(pending_rows)} rows "
            f"({n_generated_total}/{len(prompts_df)} claims done) -> {raw_csv_path}"
        )

    merged_df = load_csv(raw_csv_path)
    logger.info(f"[{model_tag}] Done - {len(merged_df)} total generated samples (raw) -> {raw_csv_path}")
    logger.info(
        f"[{model_tag}] Run postprocess_generations next to build the processed/exploded "
        f"CSV that embeddings/analysis actually read from."
    )
    return merged_df


if __name__ == "__main__":
    main()