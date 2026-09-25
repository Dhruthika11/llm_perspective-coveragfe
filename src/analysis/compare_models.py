"""
Aggregates final_report_summary.json across all model result directories under results/
and produces a comparative summary table and JSON report.

Usage:
    python -m src.analysis.compare_models
"""
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("compare_models")


def main():
    config = load_config()
    results_root = resolve("results")
    
    if not os.path.exists(results_root):
        logger.warning(f"Results root directory {results_root} does not exist.")
        return

    # Find all model subdirectories with a final_report_summary.json
    model_summaries = []
    for entry in os.listdir(results_root):
        sub_dir = os.path.join(results_root, entry)
        if os.path.isdir(sub_dir):
            summary_path = os.path.join(sub_dir, "final_report_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r") as f:
                        data = json.load(f)
                        data["model_dir"] = entry
                        model_summaries.append(data)
                except Exception as e:
                    logger.warning(f"Could not read {summary_path}: {e}")

    if not model_summaries:
        logger.info("No model summary reports found yet.")
        return

    df = pd.DataFrame(model_summaries)
    
    # Reorder columns for readability if present
    preferred_cols = [
        "model_dir", "llm_model", "num_claims", "similarity_threshold",
        "mean_coverage_score", "mean_per_sample_coverage",
        "mean_support_coverage", "mean_undermine_coverage",
        "mean_js_divergence", "mean_diversity_score",
        "claims_with_zero_coverage", "claims_with_full_coverage"
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    out_csv = os.path.join(results_root, "cross_model_comparison.csv")
    out_json = os.path.join(results_root, "cross_model_comparison.json")
    
    save_csv(df, out_csv)
    save_json(model_summaries, out_json)

    logger.info(f"Cross-model comparison saved -> {out_csv}")
    print("\n=== CROSS-MODEL COMPARISON ===")
    print(df.to_string(index=False))
    print(f"\nSaved comparison summary to {out_csv}")


if __name__ == "__main__":
    main()
