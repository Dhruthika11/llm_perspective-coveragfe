"""
Rolls up every per-claim metric CSV into one final_report_summary.json with
dataset-wide aggregate numbers - the headline results for a paper/report.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_json, load_json
from src.utils.logging_utils import get_logger
from src.analysis.threshold_calibration import get_similarity_threshold

logger = get_logger("summarize_results")

# Descriptive stance-mix buckets below are heuristic cut points (not
# calibrated the way the coverage similarity threshold is - see
# threshold_calibration.py) used purely to summarize the shape of a claim's
# stance coverage for reporting. Treat these as illustrative, not validated.
STANCE_DOMINANT_THRESHOLD = 0.5   # "clearly covers this stance"
STANCE_ABSENT_THRESHOLD = 0.1     # "barely/doesn't cover this stance"
STANCE_PRESENT_THRESHOLD = 0.3    # "covers this stance at least somewhat"


def main(config: dict = None):
    if config is None:
        config = load_config()
    metrics_dir = resolve(config["paths"]["metrics_dir"])

    coverage_df = load_csv(os.path.join(metrics_dir, "per_claim_coverage.csv"))
    divergence_df = load_csv(os.path.join(metrics_dir, "divergence_scores.csv"))
    diversity_df = load_csv(os.path.join(metrics_dir, "diversity_scores.csv"))

    # Generation quality gate numbers (written by postprocess_generations).
    # Guarded: old runs predating generation_quality.csv still summarize.
    quality_path = os.path.join(metrics_dir, "generation_quality.csv")
    quality_df = load_csv(quality_path) if os.path.exists(quality_path) else None

    threshold, threshold_meta = get_similarity_threshold(config)

    supp = coverage_df["support_coverage"]
    und = coverage_df["undermine_coverage"]
    # A claim can have zero human perspectives of a given stance, in which
    # case coverage_score.py records that stance's coverage as NaN (there's
    # nothing to cover). Both `NaN > x` and `NaN <= x` are False in pandas,
    # so naively bucketing on those comparisons makes such claims silently
    # fail every bucket and vanish from all four counts. Split them out into
    # their own explicit bucket instead of letting them disappear.
    has_both_stance_data = supp.notna() & und.notna()
    n_missing_stance_data = int((~has_both_stance_data).sum())

    supp_ok, und_ok = supp[has_both_stance_data], und[has_both_stance_data]

    summary = {
        "num_claims": int(len(coverage_df)),
        "similarity_threshold": threshold,
        "similarity_threshold_source": threshold_meta["source"],
        "similarity_threshold_validated": threshold_meta["source"] == "calibrated",
        "llm_model": config["llm"]["model"],
        "embedding_model": config["embedding"]["model"],
        "num_samples_per_claim": config["llm"]["num_samples_per_claim"],
        "mean_coverage_score": float(coverage_df["coverage_score"].mean()),
        "min_coverage_score": float(coverage_df["coverage_score"].min()),
        "max_coverage_score": float(coverage_df["coverage_score"].max()),
        "mean_per_sample_coverage": float(coverage_df["per_sample_coverage"].mean()),
        "mean_support_coverage": float(supp.mean()),
        "mean_undermine_coverage": float(und.mean()),
        "mean_js_divergence": float(divergence_df["js_divergence"].mean()),
        "mean_diversity_score": float(diversity_df["diversity_score"].mean()),
        "claims_with_full_coverage": int((coverage_df["coverage_score"] >= 0.999).sum()),
        "claims_with_zero_coverage": int((coverage_df["coverage_score"] <= 0.001).sum()),
        # Stance-mix buckets: computed ONLY over claims that have at least one
        # human perspective of both stances (see has_both_stance_data above),
        # so they're mutually exclusive and exhaustive over that subset -
        # claims missing stance data are counted separately, not dropped.
        "claims_support_only": int(
            ((supp_ok > STANCE_DOMINANT_THRESHOLD) & (und_ok <= STANCE_ABSENT_THRESHOLD)).sum()
        ),
        "claims_undermine_only": int(
            ((und_ok > STANCE_DOMINANT_THRESHOLD) & (supp_ok <= STANCE_ABSENT_THRESHOLD)).sum()
        ),
        "claims_both_stances": int(
            ((supp_ok > STANCE_PRESENT_THRESHOLD) & (und_ok > STANCE_PRESENT_THRESHOLD)).sum()
        ),
        "claims_neither_stance": int(
            ((supp_ok <= STANCE_ABSENT_THRESHOLD) & (und_ok <= STANCE_ABSENT_THRESHOLD)).sum()
        ),
        "claims_missing_stance_data": n_missing_stance_data,
        # Generation quality: usable samples actually embedded per claim
        # (n_llm_samples in per_claim_coverage counts embedded rows, so a
        # shortfall vs num_samples_per_claim means calls were lost to
        # degenerate outputs - see generation_quality.csv for the breakdown).
        "mean_embedded_samples_per_claim": float(coverage_df["n_llm_samples"].mean()),
        "min_embedded_samples_per_claim": int(coverage_df["n_llm_samples"].min()),
    }
    if quality_df is not None:
        summary["mean_usable_viewpoints_per_call"] = float(quality_df["usable_per_call"].mean())
        summary["generation_calls_zero_usable"] = int(quality_df["calls_zero_usable"].sum())
        summary["mean_parse_ok_rate"] = float(quality_df["parse_ok_rate"].mean())
        # Claims that lost ALL their samples to degenerate outputs: they have
        # no rows in the embedding index, so they never reach coverage at all.
        # (The sibling JS-divergence NaN handling lives in
        # distribution_divergence.compute_divergence_row.)
        summary["claims_with_no_llm_samples"] = int((quality_df["n_usable"] == 0).sum())

    # Stance cross-check (written by stance_crosscheck.py). Guarded: old runs
    # predating it still summarize.
    agreement_path = os.path.join(metrics_dir, "stance_agreement.json")
    if os.path.exists(agreement_path):
        agreement = load_json(agreement_path)
        summary["stance_agreement_rate"] = agreement.get("agreement_rate")
        summary["stance_agreement_n_valid"] = agreement.get("n_valid")
    if threshold_meta["source"] == "calibrated":
        summary["threshold_calibration_roc_auc"] = threshold_meta["roc_auc"]
        summary["threshold_calibration_pr_auc"] = threshold_meta["pr_auc"]
        summary["threshold_calibration_n_pairs"] = threshold_meta["n_pairs_labeled"]

    out_path = resolve(config["paths"]["summary_file"])
    save_json(summary, out_path)
    logger.info(f"Saved final summary -> {out_path}")
    logger.info(summary)
    return summary


if __name__ == "__main__":
    main()
