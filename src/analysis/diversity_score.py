"""
Measures how diverse the LLM's own samples are with each other, independent
of the human data. If an LLM's 15 samples for a claim are all near-identical
(high average pairwise similarity), that alone is evidence it isn't
representing multiple perspectives - regardless of which single perspective
it happens to match.

diversity(claim) = 1 - mean(pairwise cosine similarity among LLM samples)

Higher = more internally diverse samples.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv
from src.utils.logging_utils import get_logger
from src.analysis.similarity_matrix import ensure_normalized

logger = get_logger("diversity_score")


def mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    """embeddings: (n, d). Excludes self-similarity (diagonal). Renormalized
    defensively (see ensure_normalized) so the mean is a true cosine even if
    the saved embeddings were written unnormalized."""
    embeddings = ensure_normalized(embeddings, "diversity embeddings")
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    sim = embeddings @ embeddings.T
    off_diagonal_sum = sim.sum() - np.trace(sim)
    off_diagonal_count = n * n - n
    return float(off_diagonal_sum / off_diagonal_count)


def main(config: dict = None):
    if config is None:
        config = load_config()
    embeddings_dir = resolve(config["paths"]["embeddings_dir"])

    llm_emb = np.load(os.path.join(embeddings_dir, "llm_output_embeddings.npy"))
    llm_idx = load_csv(os.path.join(embeddings_dir, "llm_output_embeddings_index.csv"))

    rows = []
    for claim_id in sorted(llm_idx["claim_id"].unique()):
        mask = (llm_idx["claim_id"] == claim_id).values
        claim_embeddings = llm_emb[mask]
        avg_sim = mean_pairwise_similarity(claim_embeddings)
        diversity = 1 - avg_sim if not np.isnan(avg_sim) else None

        rows.append({
            "claim_id": claim_id,
            "n_samples": int(mask.sum()),
            "mean_pairwise_similarity": avg_sim,
            "diversity_score": diversity,
        })

    df = pd.DataFrame(rows)
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    save_csv(df, os.path.join(metrics_dir, "diversity_scores.csv"))
    logger.info(f"Mean diversity score across claims: {df['diversity_score'].mean():.3f}")
    return df


if __name__ == "__main__":
    main()
