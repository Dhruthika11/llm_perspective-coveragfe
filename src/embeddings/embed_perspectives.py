"""
Embed every human perspective in perspectives.csv with BGE, and save the
embeddings alongside an index CSV so rows stay aligned with the original data.

Usage:
    python -m src.embeddings.embed_perspectives
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv
from src.utils.logging_utils import get_logger
from src.embeddings.embed_model import BGEEmbedder

logger = get_logger("embed_perspectives")


def main(config: dict = None):
    if config is None:
        config = load_config()
    processed_dir = resolve(config["dataset"]["processed_dir"])
    persp_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))

    embedder = BGEEmbedder(
        model_name=config["embedding"]["model"],
        normalize=config["embedding"]["normalize"],
    )
    logger.info(f"Embedding {len(persp_df)} human perspectives...")
    embeddings = embedder.embed(
        persp_df["text"].tolist(),
        batch_size=config["embedding"]["batch_size"],
    )

    embeddings_dir = resolve(config["paths"]["embeddings_dir"])
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "perspective_embeddings.npy"), embeddings)

    # Save index so row i of the .npy always maps to row i here.
    index_df = persp_df[["claim_id", "perspective_id", "stance"]].copy()
    save_csv(index_df, os.path.join(embeddings_dir, "perspective_embeddings_index.csv"))

    logger.info(f"Saved perspective embeddings: shape={embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    main()
