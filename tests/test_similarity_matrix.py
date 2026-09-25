"""Unit tests for the cosine-similarity plumbing (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in that every cosine-via-dot-product call site stays a TRUE cosine
even if embeddings were saved unnormalized (embedding.normalize=false):
ensure_normalized renormalizes instead of silently producing wrong numbers.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.similarity_matrix import cosine_sim_matrix, ensure_normalized
from src.analysis.diversity_score import mean_pairwise_similarity


def test_ensure_normalized_returns_unit_rows():
    emb = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]])
    out = ensure_normalized(emb)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms[:2], 1.0)
    assert out[0].tolist() == [0.6, 0.8]
    # zero-norm row doesn't produce NaN/inf
    assert np.all(np.isfinite(out[2]))


def test_ensure_normalized_noop_on_normalized():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = ensure_normalized(emb)
    assert out is emb or np.array_equal(out, emb)  # numerically identical


def test_cosine_sim_matrix_correct_when_unnormalized():
    # [[3,4]] and [[6,8]] point the same way -> cosine 1.0, NOT the raw dot
    # product 50 that the old code would have returned.
    a = np.array([[3.0, 4.0]])
    b = np.array([[6.0, 8.0], [4.0, -3.0]])
    sim = cosine_sim_matrix(a, b)
    assert np.allclose(sim, [[1.0, 0.0]], atol=1e-7)


def test_diversity_still_zero_for_identical_unnormalized():
    emb = np.array([[2.0, 0.0]] * 3)
    assert mean_pairwise_similarity(emb) == 1.0
