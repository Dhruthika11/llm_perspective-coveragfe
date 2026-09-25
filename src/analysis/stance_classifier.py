"""
Tags each LLM-generated sample with a stance (support/undermine) using
embedding nearest-neighbor against the labeled human perspectives for the
same claim - no separate trained classifier needed, since Perspectrum
perspectives already carry ground-truth stance labels.

stance(LLM_sample) = stance of the human perspective it is most similar to
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, save_csv
from src.utils.logging_utils import get_logger
from src.analysis.similarity_matrix import build_all_similarity_matrices

logger = get_logger("stance_classifier")


def classify_claim(sim_matrix: np.ndarray, perspective_stances: list, sample_ids: list):
    """sim_matrix: (n_llm_samples, n_perspectives) -> list of predicted stances, one per sample."""
    nearest_idx = sim_matrix.argmax(axis=1)  # for each LLM sample, index of most similar perspective
    predicted_stances = [perspective_stances[i] for i in nearest_idx]
    confidences = sim_matrix.max(axis=1)
    return predicted_stances, confidences


def main(config: dict = None):
    if config is None:
        config = load_config()
    matrices = build_all_similarity_matrices(config)

    rows = []
    for claim_id, data in matrices.items():
        predicted_stances, confidences = classify_claim(
            data["sim_matrix"], data["perspective_stances"], data["sample_ids"]
        )
        # viewpoint_id/llm_reported_stance align 1:1 with sim rows (one output
        # row per embedded viewpoint). .get() defaults keep this working with
        # any matrix dicts built without the optional keys.
        viewpoint_ids = data.get("viewpoint_ids", [None] * len(predicted_stances))
        reported_stances = data.get("llm_reported_stances", [None] * len(predicted_stances))
        for sample_id, vp_id, reported, stance, conf in zip(
                data["sample_ids"], viewpoint_ids, reported_stances,
                predicted_stances, confidences):
            rows.append({
                "claim_id": claim_id,
                "sample_id": sample_id,
                "viewpoint_id": vp_id,
                "llm_reported_stance": reported,
                "predicted_stance": stance,
                "confidence": float(conf),
            })

    df = pd.DataFrame(rows)
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    save_csv(df, os.path.join(metrics_dir, "llm_sample_stances.csv"))
    logger.info(f"Classified stance for {len(df)} LLM samples")
    return df


if __name__ == "__main__":
    main()
