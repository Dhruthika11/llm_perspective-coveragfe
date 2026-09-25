"""Unit tests for the structured-output parser.

Run: python -m pytest tests/ -q   (from the project root)

These lock in every output shape the neutral prompt permits (single object,
concatenated objects, top-level array, viewpoints wrapper, fenced variants)
plus the degenerate shapes weak models emit (empty arrays, bare values), so
future prompt/parser edits are verified in seconds instead of by GPU runs.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.postprocess_generations import (
    parse_structured_output,
    build_combined_text,
    build_quality_report,
    main as postprocess_main,
)


def _obj(stance, summary="a summary sentence here", reasoning="a reason here"):
    return {"stance": stance, "summary": summary, "core_reasoning": reasoning}


SUPPORT = json.dumps(_obj("support"))
UNDERMINE = json.dumps(_obj("undermine"))


def _methods(vps):
    return [v["parse_method"] for v in vps]


def _stances(vps):
    return [v["llm_reported_stance"] for v in vps]


def test_single_bare_object():
    vps = parse_structured_output(SUPPORT)
    assert len(vps) == 1
    assert _methods(vps) == ["json"] and _stances(vps) == ["support"]


def test_concatenated_objects_with_chatter():
    vps = parse_structured_output("Here you go: " + SUPPORT + " and also " + UNDERMINE + " done.")
    assert len(vps) == 2
    assert _methods(vps) == ["json", "json"]
    assert _stances(vps) == ["support", "undermine"]


def test_top_level_array_no_double_count():
    # Inner objects must not be counted twice (once standalone, once via array).
    vps = parse_structured_output("[" + SUPPORT + ", " + UNDERMINE + "]")
    assert len(vps) == 2
    assert _stances(vps) == ["support", "undermine"]


def test_viewpoints_wrapper_no_double_count():
    text = json.dumps({"viewpoints": [_obj("support"), _obj("undermine")]})
    vps = parse_structured_output(text)
    assert len(vps) == 2
    assert _stances(vps) == ["support", "undermine"]


def test_fenced_multiples_and_fenced_array():
    vps = parse_structured_output("```json\n" + SUPPORT + "\n" + UNDERMINE + "\n```")
    assert len(vps) == 2
    vps = parse_structured_output("```json\n[" + SUPPORT + "," + UNDERMINE + "]\n```")
    assert len(vps) == 2


def test_braces_and_brackets_inside_strings():
    text = json.dumps({"stance": "support", "summary": "a {note} and [bracket] here",
                       "core_reasoning": "r"})
    vps = parse_structured_output(text)
    assert len(vps) == 1 and _stances(vps) == ["support"]


def test_old_key_names():
    text = json.dumps({"stance": "undermine", "perspective_summary": "old keys",
                       "core_reasoning": "r"})
    vps = parse_structured_output(text)
    assert len(vps) == 1 and _stances(vps) == ["undermine"]


def test_think_prefix_stripped():
    vps = parse_structured_output("rambling thought</think>\n" + SUPPORT)
    assert len(vps) == 1 and _methods(vps) == ["json"]


def test_truncated_tail_ignored():
    vps = parse_structured_output(SUPPORT + ' {"stance": "supp')
    assert len(vps) == 1 and _methods(vps) == ["json"]


def test_exact_duplicate_objects_collapse():
    vps = parse_structured_output(SUPPORT + " " + SUPPORT)
    assert len(vps) == 1


def test_markdown_fallback():
    vps = parse_structured_output(
        "**Stance:** Support\n**Summary:** md sum\n**Core Reasoning:** md reason")
    assert len(vps) == 1 and _methods(vps) == ["markdown_fallback"]


def test_unparsed_fallback():
    vps = parse_structured_output("just some rambling with no structure at all")
    assert len(vps) == 1 and _methods(vps) == ["unparsed"]
    assert vps[0]["llm_reported_stance"] is None


def test_degenerate_empty_array_yields_no_viewpoints():
    # Weak models emit bare []: zero JSON viewpoints -> caller falls back to
    # the unparsed path (short summary, dropped downstream by length filter).
    for text in ("[]", "[]\n[]", "```json\n[]\n```"):
        vps = parse_structured_output(text)
        assert _methods(vps) == ["unparsed"], text
        assert len(build_combined_text(vps[0]["perspective_summary"],
                                       vps[0]["core_reasoning"])) <= 10, text


def test_degenerate_bare_values_yield_no_viewpoints():
    for text in ('["null"]', '["Undermine"]'):
        vps = parse_structured_output(text)
        assert _methods(vps) == ["unparsed"], text


def test_quality_report_counts():
    import pandas as pd
    pre = pd.DataFrame([
        {"claim_id": 1, "sample_id": 0, "parse_ok": True, "parse_method": "json"},
        {"claim_id": 1, "sample_id": 0, "parse_ok": True, "parse_method": "json"},
        {"claim_id": 1, "sample_id": 1, "parse_ok": False, "parse_method": "unparsed"},
        {"claim_id": 2, "sample_id": 0, "parse_ok": True, "parse_method": "json"},
    ])
    post = pre.iloc[[0, 1, 3]].reset_index(drop=True)  # sample (1,1) dropped
    rep = build_quality_report(pre, post)
    r1 = rep.loc[rep["claim_id"] == 1].iloc[0]
    r2 = rep.loc[rep["claim_id"] == 2].iloc[0]
    assert (r1["n_calls"], r1["n_usable"], r1["calls_zero_usable"]) == (2, 2, 1)
    assert r2["calls_zero_usable"] == 0
    assert r2["usable_per_call"] == pytest.approx(1.0)


def test_empty_raw_csv_saves_schema_and_does_not_crash(tmp_path):
    import pandas as pd
    # Header-only raw CSV (degraded run): previously KeyError'd on the missing
    # parsed columns; now saves an empty processed CSV with the full schema.
    raw_dir = tmp_path / "raw" / "y"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(columns=["claim_id", "sample_id", "generated_text"]).to_csv(
        raw_dir / "raw_generations.csv", index=False
    )
    config = {
        "paths": {
            "llm_outputs_raw": str(tmp_path / "raw"),
            "llm_outputs_processed": str(tmp_path / "processed.csv"),
            "metrics_dir": str(tmp_path / "metrics"),
        },
        "llm": {"model": "x/y"},
    }
    df = postprocess_main(config)
    assert df.empty
    out = pd.read_csv(tmp_path / "processed.csv")
    assert list(out.columns) == [
        "claim_id", "sample_id", "viewpoint_id", "generated_text",
        "llm_reported_stance", "perspective_summary", "core_reasoning",
        "combined_text", "parse_ok", "parse_method",
    ]
