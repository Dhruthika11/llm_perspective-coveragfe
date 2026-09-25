"""
For each claim, build the cosine similarity matrix:

    rows    = LLM samples for that claim
    columns = human perspectives for that claim

Since BGE embeddings are saved normalized, cosine similarity reduces to a
plain dot product, which is what we use here for speed.

Exposes build_all_similarity_matrices() which other analysis scripts import
directly, so the (sometimes large) matrices don't need to be written to disk.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv
from src.utils.logging_utils import get_logger

logger = get_logger("similarity_matrix")

_norm_warned = False  # module-level: warn only once per process, not per claim


def ensure_normalized(emb: np.ndarray, name: str = "embeddings") -> np.ndarray:
    """Safety net for every cosine-via-dot-product call site. Coverage,
    stance, diversity, and calibration all assume L2-normalized rows (which
    config embedding.normalize=true guarantees at write time) - but if that
    setting is ever flipped, all downstream cosine math silently breaks.
    This checks row norms and renormalizes (guarding zero-norm rows against
    div-by-zero) with a one-time warning, instead of trusting the config.
    """
    global _norm_warned
    if emb.size == 0:
        return emb
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.allclose(norms, 1.0, atol=1e-3):
        return emb
    if not _norm_warned:
        logger.warning(
            f"{name} are not L2-normalized (config embedding.normalize is probably "
            f"false) - renormalizing in memory so cosine similarities stay correct. "
            f"Consider setting embedding.normalize: true instead."
        )
        _norm_warned = True
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) -> (n, m) similarity matrix. Inputs are
    renormalized defensively (see ensure_normalized) so this stays a true
    cosine even if the saved embeddings were written unnormalized."""
    a = ensure_normalized(a, "similarity matrix rows (LLM)")
    b = ensure_normalized(b, "similarity matrix rows (human perspectives)")
    return a @ b.T


def load_embeddings(config: dict):
    embeddings_dir = resolve(config["paths"]["embeddings_dir"])

    persp_emb = np.load(os.path.join(embeddings_dir, "perspective_embeddings.npy"))
    persp_idx = load_csv(os.path.join(embeddings_dir, "perspective_embeddings_index.csv"))

    llm_emb = np.load(os.path.join(embeddings_dir, "llm_output_embeddings.npy"))
    llm_idx = load_csv(os.path.join(embeddings_dir, "llm_output_embeddings_index.csv"))

    return persp_emb, persp_idx, llm_emb, llm_idx


def build_all_similarity_matrices(config: dict) -> dict:
    """
    Returns:
        { claim_id: {
              "sim_matrix": np.ndarray (n_llm_samples x n_perspectives),
              "perspective_ids": list,
              "perspective_stances": list,
              "sample_ids": list,
          }, ... }
    """
    persp_emb, persp_idx, llm_emb, llm_idx = load_embeddings(config)

    claim_ids = sorted(set(persp_idx["claim_id"]).intersection(set(llm_idx["claim_id"])))
    logger.info(f"Building similarity matrices for {len(claim_ids)} claims")

    results = {}
    for claim_id in claim_ids:
        p_mask = (persp_idx["claim_id"] == claim_id).values
        l_mask = (llm_idx["claim_id"] == claim_id).values

        p_emb_claim = persp_emb[p_mask]
        l_emb_claim = llm_emb[l_mask]

        sim = cosine_sim_matrix(l_emb_claim, p_emb_claim)  # (n_llm, n_persp)

        # Carry the self-reported stance + viewpoint id through alongside the
        # matrix rows (both optional: legacy index CSVs may lack the columns).
        # llm_reported_stances aligns 1:1 with sim rows and powers the
        # reported-vs-predicted stance cross-check (stance_crosscheck.py).
        if "llm_reported_stance" in llm_idx.columns:
            reported = [None if pd.isna(v) else v
                        for v in llm_idx.loc[l_mask, "llm_reported_stance"]]
        else:
            reported = [None] * int(l_mask.sum())

        results[claim_id] = {
            "sim_matrix": sim,
            "perspective_ids": persp_idx.loc[p_mask, "perspective_id"].tolist(),
            "perspective_stances": persp_idx.loc[p_mask, "stance"].tolist(),
            "sample_ids": llm_idx.loc[l_mask, "sample_id"].tolist(),
            "viewpoint_ids": (llm_idx.loc[l_mask, "viewpoint_id"].tolist()
                              if "viewpoint_id" in llm_idx.columns
                              else [None] * int(l_mask.sum())),
            "llm_reported_stances": reported,
        }
    return results


if __name__ == "__main__":
    config = load_config()
    matrices = build_all_similarity_matrices(config)
    for cid, data in matrices.items():
        logger.info(f"Claim {cid}: sim_matrix shape = {data['sim_matrix'].shape}")
