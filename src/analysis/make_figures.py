"""
Generates the three summary figures referenced in the README:
  - coverage_histogram.png
  - stance_balance_comparison.png
  - similarity_heatmap_examples.png (first claim as an example)

Run after coverage_score.py, distribution_divergence.py, and
similarity_matrix.py have produced their outputs.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv
from src.utils.logging_utils import get_logger
from src.analysis.similarity_matrix import build_all_similarity_matrices

logger = get_logger("make_figures")


def plot_coverage_histogram(metrics_dir, figures_dir):
    df = load_csv(os.path.join(metrics_dir, "per_claim_coverage.csv"))
    plt.figure(figsize=(6, 4))
    sns.histplot(df["coverage_score"], bins=10, kde=False)
    plt.xlabel("Coverage score")
    plt.ylabel("Number of claims")
    plt.title("Distribution of per-claim perspective coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "coverage_histogram.png"), dpi=150)
    plt.close()


def plot_stance_balance(metrics_dir, figures_dir):
    # Per-claim scatter: human support ratio (x) vs LLM support ratio (y).
    # A per-claim grouped bar chart was used previously, but it degrades into
    # an unreadable wall of bars at full-dataset scale (907 claims) - a
    # scatter with a diagonal reference shows the same (mis)alignment at any
    # scale. Points above the diagonal = LLM more supportive than humans.
    df = load_csv(os.path.join(metrics_dir, "stance_distribution.csv"))

    plt.figure(figsize=(6, 5))
    plt.scatter(df["human_support_ratio"], df["llm_support_ratio"],
                alpha=0.5, s=18)
    lims = [-0.02, 1.02]
    plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1,
             label="perfect alignment")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Human support-stance ratio")
    plt.ylabel("LLM support-stance ratio")
    plt.title(f"Human vs LLM stance balance per claim (n={len(df)})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "stance_balance_comparison.png"), dpi=150)
    plt.close()


def plot_similarity_heatmap_example(config, figures_dir):
    matrices = build_all_similarity_matrices(config)
    if not matrices:
        # Degraded run (e.g. every sample failed parsing): no matrix to show.
        logger.warning("No similarity matrices built - skipping example heatmap.")
        return
    first_claim_id = sorted(matrices.keys())[0]
    data = matrices[first_claim_id]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        data["sim_matrix"], annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[f"P{pid}" for pid in data["perspective_ids"]],
        yticklabels=[f"S{sid}" for sid in data["sample_ids"]],
    )
    plt.xlabel("Human perspectives")
    plt.ylabel("LLM samples")
    plt.title(f"Similarity heatmap - claim {first_claim_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "similarity_heatmap_examples.png"), dpi=150)
    plt.close()


def main(config: dict = None):
    if config is None:
        config = load_config()
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    figures_dir = resolve(config["paths"]["figures_dir"])
    os.makedirs(figures_dir, exist_ok=True)

    plot_coverage_histogram(metrics_dir, figures_dir)
    plot_stance_balance(metrics_dir, figures_dir)
    plot_similarity_heatmap_example(config, figures_dir)
    logger.info(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
