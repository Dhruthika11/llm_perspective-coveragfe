"""
Top-level orchestrator for the whole pipeline.

Usage:
    python run_pipeline.py                    # run everything, in order
    python run_pipeline.py --stage data_prep   # run just one stage
    python run_pipeline.py --stage generation
    python run_pipeline.py --stage embeddings
    python run_pipeline.py --stage calibration
    python run_pipeline.py --stage analysis
    python run_pipeline.py --model 7b          # run with 7B model
    python run_pipeline.py --model 3b --provider groq  # run with 3B Groq
"""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logging_utils import get_logger
from src.utils.io_utils import load_config as _load_config

logger = get_logger("run_pipeline")

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config.yaml")


def load_model_config(model_key: str) -> dict:
    """Load model-specific configuration from the config.yaml models section."""
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    if model_key in config["llm"]["models"]:
        return config["llm"]["models"][model_key]
    return {}


def build_config(model: str = None, provider: str = None) -> dict:
    """Build the final, fully-resolved config for a run: base config.yaml,
    with --model/--provider overrides applied, routed through
    io_utils.load_config() so the model-namespaced output paths (results/
    <model_tag>/..., data/llm_outputs/<model_tag>/...) actually get derived.

    This is the single place model overrides get applied - every stage
    below receives the result of this function, so paths and llm settings
    stay consistent across the whole run instead of each stage re-loading
    the un-namespaced config.yaml straight off disk.
    """
    with open(_CONFIG_PATH, "r") as f:
        raw_config = yaml.safe_load(f)

    if model:
        model_cfg = load_model_config(model)
        if model_cfg:
            # name/provider are required - without them the model entry is
            # meaningless, so skip the override loudly rather than KeyError.
            if "name" not in model_cfg or "provider" not in model_cfg:
                logger.warning(
                    f"--model {model} entry is missing 'name' and/or 'provider' - "
                    f"ignoring the model override and using the global llm settings."
                )
            else:
                raw_config["llm"]["model"] = model_cfg["name"]
                raw_config["llm"]["provider"] = model_cfg["provider"]
                # Sampling params are optional per entry: a partial entry
                # inherits the global llm.* defaults instead of KeyError-ing.
                raw_config["llm"]["temperature"] = model_cfg.get(
                    "temperature", raw_config["llm"].get("temperature", 0.8))
                raw_config["llm"]["top_p"] = model_cfg.get(
                    "top_p", raw_config["llm"].get("top_p", 0.95))
                raw_config["llm"]["max_new_tokens"] = model_cfg.get(
                    "max_new_tokens", raw_config["llm"].get("max_new_tokens", 512))
            # Larger models (e.g. 12b/32b) carry their own smaller batch_size
            # in the config entry (global batch_size=10 would OOM them) -
            # apply it here so each --model run self-configures.
            if "batch_size" in model_cfg:
                raw_config["generation"]["batch_size"] = model_cfg["batch_size"]
                logger.info(f"Overriding generation.batch_size with --model {model}: {model_cfg['batch_size']}")
            logger.info(f"Effective model after --model {model}: {raw_config['llm']['model']}")
        else:
            logger.warning(f"--model {model} not found in configs/config.yaml llm.models - ignoring.")

    if provider:
        raw_config["llm"]["provider"] = provider
        logger.info(f"Overriding provider with --provider {provider}")

    # Routes raw_config through load_config(overrides=...), which derives
    # the model-namespaced paths (see src/utils/io_utils.py) from
    # raw_config["llm"]["model"] - this is what makes different --model runs
    # write to separate results/<model_tag>/ and data/llm_outputs/<model_tag>/
    # folders instead of overwriting each other.
    config = _load_config(overrides=raw_config)
    logger.info(f"Using model: {config['llm']['model']} via {config['llm']['provider']}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Run the LLM perspective coverage pipeline.")
    parser.add_argument(
        "--stage", choices=["data_prep", "generation", "embeddings", "calibration", "analysis"], default=None,
        help="Run only this stage. Omit to run the full pipeline in order.",
    )
    parser.add_argument(
        "--model", choices=["1.5b", "3b", "7b", "12b", "32b"],
        help="LLM model size to use (overrides config.yaml default).",
    )
    parser.add_argument(
        "--provider", choices=["huggingface", "groq"],
        help="LLM provider to use (overrides config.yaml default).",
    )
    args = parser.parse_args()

    config = build_config(model=args.model, provider=args.provider)

    STAGES = {
        "data_prep": run_data_prep,
        "generation": run_generation,
        "embeddings": run_embeddings,
        "calibration": run_calibration,
        "analysis": run_analysis,
    }

    if args.stage:
        logger.info(f"Running stage: {args.stage}")
        STAGES[args.stage](config)
    else:
        logger.info("Running full pipeline in order")
        for stage_name in ["data_prep", "generation", "embeddings", "calibration", "analysis"]:
            logger.info(f"Running stage: {stage_name}")
            STAGES[stage_name](config)

    logger.info("Pipeline complete.")


def run_data_prep(config: dict = None):
    """Run data preparation stage."""
    if config is None:
        config = build_config()

    logger.info("=== STAGE: data_prep ===")

    from src.data_prep.parse_perspectrum import main as parse_main
    from src.data_prep.build_prompts import main as build_prompts_main

    parse_main(config)
    build_prompts_main(config)


def run_generation(config: dict = None):
    """Run generation stage."""
    if config is None:
        config = build_config()

    logger.info("=== STAGE: generation ===")

    from src.generation import generate_llama, postprocess_generations

    generate_llama.main(config)
    postprocess_generations.main(config)


def run_embeddings(config: dict = None):
    """Run embeddings stage."""
    if config is None:
        config = build_config()

    logger.info("=== STAGE: embeddings ===")

    from src.embeddings import embed_perspectives, embed_llm_outputs
    embed_perspectives.main(config)
    embed_llm_outputs.main(config)


def run_calibration(config: dict = None):
    """Run calibration stage."""
    if config is None:
        config = build_config()

    logger.info("=== STAGE: calibration ===")

    from src.analysis import threshold_calibration
    threshold_calibration.main(config)


def run_analysis(config: dict = None):
    """Run analysis stage."""
    if config is None:
        config = build_config()

    logger.info("=== STAGE: analysis ===")

    from src.analysis import (
        coverage_score, stance_classifier, stance_crosscheck, distribution_divergence,
        diversity_score, make_figures, summarize_results,
    )
    coverage_score.main(config)
    stance_classifier.main(config)
    stance_crosscheck.main(config)
    distribution_divergence.main(config)
    diversity_score.main(config)
    make_figures.main(config)
    summarize_results.main(config)


if __name__ == "__main__":
    main()
