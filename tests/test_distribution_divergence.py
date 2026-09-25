"""Unit tests for distribution divergence (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in the zero-sample contract: a claim with ZERO LLM samples must
get js_divergence=None + NaN LLM ratios - never a fabricated neutral
distribution that would distort the dataset mean.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.distribution_divergence import compute_divergence_row


def test_empty_llm_side_yields_none_not_fabricated():
    row = compute_divergence_row(
        pd.Series(["support", "undermine"]),
        pd.Series([], dtype=str),
    )
    assert row["js_divergence"] is None
    assert all(np.isnan(row[k]) for k in
               ("llm_support_ratio", "llm_undermine_ratio", "llm_unclear_ratio"))
    # human side still computed
    assert row["human_support_ratio"] == 0.5
    assert row["human_undermine_ratio"] == 0.5


def test_normal_case_matches_js_formula():
    human = pd.Series(["support", "support", "undermine"])
    llm = pd.Series(["support", "undermine", "undermine", "undermine"])
    row = compute_divergence_row(human, llm)
    h = np.array([2 / 3, 1 / 3, 0.0])
    l = np.array([0.25, 0.75, 0.0])
    expected = float(jensenshannon(h, l, base=2) ** 2)
    assert row["js_divergence"] == expected
    assert (row["llm_support_ratio"], row["llm_undermine_ratio"], row["llm_unclear_ratio"]) == (0.25, 0.75, 0.0)


def test_identical_distributions_zero_divergence():
    stances = pd.Series(["support", "undermine", "unclear"])
    row = compute_divergence_row(stances, stances)
    assert row["js_divergence"] == 0.0
