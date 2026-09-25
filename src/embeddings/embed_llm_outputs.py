"""
Embed every LLM-generated sample with BGE (same model/settings as human
perspectives, so the two embedding spaces are directly comparable).

Embeds `combined_text` (perspective_summary + core_reasoning produced by
src/generation/postprocess_generations.py), NOT the raw single-sentence
`generated_text`. Embedding the fuller, structured text is the fix for the
information-loss problem: a one-sentence compression prompt guaranteed
semantic loss before the text ever reached the embedder, so coverage
computed on it partly measured prompt-compression failure rather than the
LLM's actual perspective-space coverage.

Usage:
    python -m src.embeddings.embed_llm_outputs
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv
from src.utils.logging_utils import get_logger
from src.embeddings.embed_model import BGEEmbedder

logger = get_logger("embed_llm_outputs")


def main(config: dict = None):
    if config is None:
        config = load_config()
    gen_path = resolve(config["paths"]["llm_outputs_processed"])
    gen_df = load_csv(gen_path)

    if "combined_text" not in gen_df.columns:
        raise ValueError(
            "processed_generations.csv has no 'combined_text' column. Run "
            "src/generation/postprocess_generations.py (it parses the structured "
            "LLM JSON output and builds combined_text) before embedding."
        )

    embedder = BGEEmbedder(
        model_name=config["embedding"]["model"],
        normalize=config["embedding"]["normalize"],
    )
    logger.info(f"Embedding {len(gen_df)} LLM-generated samples (combined_text)...")
    embeddings = embedder.embed(
        gen_df["combined_text"].tolist(),
        batch_size=config["embedding"]["batch_size"],
    )

    embeddings_dir = resolve(config["paths"]["embeddings_dir"])
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "llm_output_embeddings.npy"), embeddings)

    # Carry the self-reported stance and viewpoint id through so downstream
    # steps (e.g. cross-checking the embedding-nearest-neighbor stance
    # classifier against what the LLM itself claimed, or tracing a row back
    # to which viewpoint within a multi-viewpoint generation it came from)
    # can use it without re-reading the raw CSV.
    index_cols = ["claim_id", "sample_id"]
    for optional_col in ("viewpoint_id", "llm_reported_stance"):
        if optional_col in gen_df.columns:
            index_cols.append(optional_col)
    index_df = gen_df[index_cols].copy()
    save_csv(index_df, os.path.join(embeddings_dir, "llm_output_embeddings_index.csv"))

    logger.info(f"Saved LLM output embeddings: shape={embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    main()
