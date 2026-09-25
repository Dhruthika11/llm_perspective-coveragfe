"""
Core metric: does the set of LLM-sampled opinions, collectively, "cover"
every distinct human perspective on a claim?

coverage(claim) = |{ human perspectives p : max_i sim(LLM_sample_i, p) >= tau }|
                   ------------------------------------------------------------
                               total human perspectives for claim

Also reports per-perspective best-match similarity (useful for debugging
which specific perspectives the LLM is missing).
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, save_csv
from src.utils.logging_utils import get_logger
from src.analysis.similarity_matrix import build_all_similarity_matrices
from src.analysis.threshold_calibration import get_similarity_threshold

logger = get_logger("coverage_score")


def compute_coverage_for_claim(sim_matrix, perspective_ids, perspective_stances, threshold: float):
    """sim_matrix: (n_llm_samples, n_perspectives)"""
    n_persp = sim_matrix.shape[1]
    n_samples = sim_matrix.shape[0]

    # --- Tier 1: All-samples combined coverage ---
    best_match_per_persp = sim_matrix.max(axis=0)  # (n_perspectives,)
    covered = best_match_per_persp >= threshold
    coverage_score = float(covered.sum()) / n_persp if n_persp > 0 else float("nan")

    # --- Tier 2: Per-sample coverage ---
    # For each sample, count how many perspectives are above threshold
    covered_per_sample = (sim_matrix >= threshold).sum(axis=1)  # (n_samples,)
    per_sample_coverage = float(covered_per_sample.mean() / n_persp) if n_samples > 0 and n_persp > 0 else float("nan")

    # --- Tier 3: Per-stance coverage ---
    unique_stances = set(perspective_stances)
    stance_coverage = {}
    for s in unique_stances:
        # p_mask is never empty here: s comes from the observed stances, so at
        # least the perspective it was observed on matches.
        p_mask = [i for i, stance in enumerate(perspective_stances) if stance == s]
        # Best match for perspectives of this stance (across all samples)
        best_matches = sim_matrix.max(axis=0)[p_mask]  # (n_perspectives_of_this_stance,)
        n_in_stance = len(p_mask)
        n_covered_s = int((best_matches >= threshold).sum())
        stance_coverage[s] = n_covered_s / n_in_stance

    # Build per-perspective rows with best_similarity and covered flag
    rows = []
    for j in range(n_persp):
        rows.append({
            "perspective_id": perspective_ids[j],
            "stance": perspective_stances[j],
            "best_similarity": float(best_match_per_persp[j]),
            "covered": bool(covered[j]),
        })

    return coverage_score, per_sample_coverage, stance_coverage, rows


def main(config: dict = None):
    if config is None:
        config = load_config()
    threshold, threshold_meta = get_similarity_threshold(config)
    matrices = build_all_similarity_matrices(config)

    per_claim_rows = []
    per_perspective_rows = []
    # Collect any stance labels beyond support/undermine (e.g. "unclear")
    # seen across claims, so their coverage isn't silently discarded.
    other_stance_keys = set()

    for claim_id, data in matrices.items():
        coverage_score, per_sample_cov, stance_cov, persp_rows = compute_coverage_for_claim(
            data["sim_matrix"], data["perspective_ids"],
            data["perspective_stances"], threshold,
        )
        n_persp = len(data["perspective_ids"])
        n_covered = sum(r["covered"] for r in persp_rows)

        row = {
            "claim_id": claim_id,
            "coverage_score": coverage_score,
            "n_perspectives": n_persp,
            "n_covered": n_covered,
            "n_llm_samples": data["sim_matrix"].shape[0],
            "per_sample_coverage": per_sample_cov,
            "support_coverage": stance_cov.get("support", float("nan")),
            "undermine_coverage": stance_cov.get("undermine", float("nan")),
        }
        # Surface every other stance bucket (typically "unclear") instead of
        # dropping it - these perspectives are still counted in Tier 1's
        # coverage_score/n_perspectives, so their own coverage number should
        # be visible too, not silently discarded.
        for stance_key, cov_val in stance_cov.items():
            if stance_key not in ("support", "undermine"):
                col = f"{stance_key}_coverage"
                row[col] = cov_val
                other_stance_keys.add(col)
        per_claim_rows.append(row)

        for r in persp_rows:
            r["claim_id"] = claim_id
            per_perspective_rows.append(r)

    per_claim_df = pd.DataFrame(per_claim_rows).sort_values("claim_id")
    if other_stance_keys:
        logger.info(f"Non-support/undermine stance coverage columns present: {sorted(other_stance_keys)}")
    per_perspective_df = pd.DataFrame(per_perspective_rows).sort_values(["claim_id", "perspective_id"])

    metrics_dir = resolve(config["paths"]["metrics_dir"])
    save_csv(per_claim_df, os.path.join(metrics_dir, "per_claim_coverage.csv"))
    save_csv(per_perspective_df, os.path.join(metrics_dir, "per_perspective_coverage.csv"))

    overall_mean = per_claim_df["coverage_score"].mean()
    logger.info(f"Mean coverage score across all claims: {overall_mean:.3f} (threshold tau={threshold:.3f}, source={threshold_meta['source']})")
    logger.info(f"Saved per-claim and per-perspective coverage CSVs to {metrics_dir}")

    return per_claim_df, per_perspective_df


if __name__ == "__main__":
    main()
