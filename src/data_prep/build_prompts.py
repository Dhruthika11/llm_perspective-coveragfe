"""
Build the actual text prompts sent to the LLM for each claim, using the
template defined in configs/config.yaml. Kept separate from generation so
you can inspect/edit prompts without touching model-calling code.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.logging_utils import get_logger

logger = get_logger("build_prompts")


def build_prompts(claims_df: pd.DataFrame, template: str) -> pd.DataFrame:
    rows = []
    for _, row in claims_df.iterrows():
        prompt = template.format(claim=row["claim_text"])
        rows.append({
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "prompt": prompt,
        })
    return pd.DataFrame(rows)


def main(config: dict = None):
    # Use provided config or load default
    if config is None:
        from src.utils.io_utils import load_config
        config = load_config()

    processed_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), config["dataset"]["processed_dir"])
    claims_df = pd.read_csv(os.path.join(processed_dir, "claims.csv"))

    prompts_df = build_prompts(claims_df, config["llm"]["prompt_template"])
    out_path = os.path.join(processed_dir, "prompts.csv")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(prompts_df).to_csv(out_path, index=False)
    logger.info(f"Built {len(prompts_df)} prompts -> {out_path}")


if __name__ == "__main__":
    main()