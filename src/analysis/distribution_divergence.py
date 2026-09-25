"""
Compares the *distribution* of stances in the human perspectives for a claim
against the distribution of (classified) stances in the LLM's samples, using
Jensen-Shannon divergence. Low divergence = the LLM's opinion spread matches
the real spread; high divergence = the LLM is skewed toward one side.

Requires stance_classifier.py to have been run first (llm_sample_stances.csv).
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv
from src.utils.logging_utils import get_logger

logger = get_logger("distribution_divergence")

STANCE_LABELS = ["support", "undermine", "unclear"]


def stance_distribution(stances: pd.Series) -> np.ndarray:
    # All three buckets (Perspectrum perspectives - and the NN stance
    # classifier - can label "unclear"). Reindexing drops anything outside
    # these labels from the total, consistent with coverage_score.py, which
    # counts unclear perspectives in its denominator rather than discarding
    # them.
    counts = stances.value_counts().reindex(STANCE_LABELS, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])  # neutral fallback, avoids div-by-zero
    return (counts / total).values


def compute_divergence_row(human_stances: pd.Series, llm_stances: pd.Series) -> dict:
    """Per-claim stance-mix comparison: human ratios, LLM ratios, JS divergence.

    If the LLM produced ZERO usable samples for this claim, its ratios are
    NaN and js_divergence is None - deliberately NOT the [1/3,1/3,1/3]
    neutral fallback, which would fabricate a distribution and distort the
    dataset mean with a number computed against made-up data. main() counts
    these claims and warns; pandas .mean() skips the NaNs automatically.
    """
    human_dist = stance_distribution(human_stances)
    if len(llm_stances) == 0:
        llm_dist = np.array([float("nan")] * 3)
        js_divergence = None
    else:
        llm_dist = stance_distribution(llm_stances)
        # jensenshannon returns distance (sqrt of divergence); square it for divergence.
        js_distance = jensenshannon(human_dist, llm_dist, base=2)
        js_divergence = float(js_distance ** 2) if not np.isnan(js_distance) else None

    return {
        "human_support_ratio": human_dist[0],
        "human_undermine_ratio": human_dist[1],
        "human_unclear_ratio": human_dist[2],
        "llm_support_ratio": llm_dist[0],
        "llm_undermine_ratio": llm_dist[1],
        "llm_unclear_ratio": llm_dist[2],
        "js_divergence": js_divergence,
    }


def main(config: dict = None):
    if config is None:
        config = load_config()
    processed_dir = resolve(config["dataset"]["processed_dir"])
    metrics_dir = resolve(config["paths"]["metrics_dir"])

    persp_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))
    llm_stance_df = load_csv(os.path.join(metrics_dir, "llm_sample_stances.csv"))

    rows = []
    n_excluded = 0
    for claim_id in sorted(persp_df["claim_id"].unique()):
        human_stances = persp_df.loc[persp_df["claim_id"] == claim_id, "stance"]
        llm_stances = llm_stance_df.loc[llm_stance_df["claim_id"] == claim_id, "predicted_stance"]

        row = {"claim_id": claim_id, **compute_divergence_row(human_stances, llm_stances)}
        if row["js_divergence"] is None:
            n_excluded += 1
        rows.append(row)

    if n_excluded:
        logger.warning(
            f"{n_excluded} claim(s) had zero LLM samples - js_divergence set to null "
            f"(excluded from the mean), not a fabricated neutral distribution. "
            f"See generation_quality.csv for which claims lost all their samples."
        )

    df = pd.DataFrame(rows)
    save_csv(df, os.path.join(metrics_dir, "divergence_scores.csv"))
    # Also save under the friendlier name used in the README/stance table.
    save_csv(df, os.path.join(metrics_dir, "stance_distribution.csv"))

    logger.info(f"Mean JS divergence across claims: {df['js_divergence'].mean():.3f}")
    return df


if __name__ == "__main__":
    main()
