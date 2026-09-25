"""Unit tests for the reported-vs-predicted stance cross-check (no GPU needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in the agreement contract: only rows with a clean
support/undermine on BOTH sides count toward the rate; missing/unparseable
self-reports are counted separately, never folded into a bucket.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.stance_crosscheck import compute_crosscheck


def _df(rows):
    return pd.DataFrame(rows, columns=["llm_reported_stance", "predicted_stance"])


def test_full_agreement():
    res = compute_crosscheck(_df([
        ("support", "support"),
        ("undermine", "undermine"),
        ("support", "support"),
    ]))
    assert res["agreement_rate"] == 1.0
    assert res["n_valid"] == 3
    assert res["n_reported_missing"] == 0


def test_mixed_agreement_and_confusion():
    res = compute_crosscheck(_df([
        ("support", "support"),
        ("support", "undermine"),
        ("undermine", "undermine"),
        ("undermine", "support"),
    ]))
    assert res["agreement_rate"] == 0.5
    assert res["n_valid"] == 4
    conf = {(r["llm_reported_stance"], r["predicted_stance"]): r["n"]
            for _, r in res["confusion"].iterrows()}
    assert conf == {("support", "support"): 1, ("support", "undermine"): 1,
                    ("undermine", "undermine"): 1, ("undermine", "support"): 1}


def test_missing_self_reports_excluded_but_counted():
    res = compute_crosscheck(_df([
        ("support", "support"),
        (None, "support"),        # parse_ok=False -> NaN on CSV roundtrip
        ("unclear", "undermine"),  # unparseable stance label
        (None, "gibberish"),      # both sides bad
    ]))
    assert res["n_valid"] == 1
    assert res["agreement_rate"] == 1.0
    assert res["n_reported_missing"] == 3


def test_empty_dataframe_null_rate():
    res = compute_crosscheck(_df([]))
    assert res["n_valid"] == 0
    assert res["agreement_rate"] is None
