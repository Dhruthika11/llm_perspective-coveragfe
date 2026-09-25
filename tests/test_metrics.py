"""Unit tests for the pure metric functions (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.coverage_score import compute_coverage_for_claim
from src.analysis.diversity_score import mean_pairwise_similarity
from src.analysis.distribution_divergence import stance_distribution


def test_coverage_tiers():
    # 2 LLM samples x 3 perspectives; tau=0.75
    sim = np.array([[0.9, 0.1, 0.2],
                    [0.1, 0.8, 0.1]])
    cov, per_sample, stance_cov, rows = compute_coverage_for_claim(
        sim, [101, 102, 103], ["support", "undermine", "support"], 0.75)
    assert cov == pytest.approx(2 / 3)          # persp 101, 102 covered
    assert per_sample == pytest.approx(2 / 6)   # each sample covers 1 of 3
    assert stance_cov["support"] == pytest.approx(1 / 2)
    assert stance_cov["undermine"] == pytest.approx(1.0)
    assert [r["covered"] for r in rows] == [True, True, False]


def test_coverage_empty_perspectives_is_nan():
    sim = np.zeros((2, 0))
    cov, per_sample, stance_cov, rows = compute_coverage_for_claim(sim, [], [], 0.75)
    assert np.isnan(cov) and rows == []


def test_diversity_identical_samples_zero():
    emb = np.array([[1.0, 0.0]] * 4)
    assert mean_pairwise_similarity(emb) == pytest.approx(1.0)
    assert 1 - mean_pairwise_similarity(emb) == pytest.approx(0.0)


def test_diversity_single_sample_is_nan():
    assert np.isnan(mean_pairwise_similarity(np.array([[1.0, 0.0]])))


def test_stance_distribution_includes_unclear():
    d = stance_distribution(pd.Series(["support", "support", "undermine", "unclear"]))
    assert d.tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_stance_distribution_empty_is_neutral():
    d = stance_distribution(pd.Series([], dtype=str))
    assert d.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
