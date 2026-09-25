"""
Cross-checks what the LLM *said* its stance was (llm_reported_stance, from the
structured generation output) against what the embedding nearest-neighbor
stance classifier *thinks* it was (predicted_stance).

Agreement near 1.0 means the classifier's stance assignments - which drive
coverage's per-stance breakdown and the JS divergence - are consistent with
the model's own self-reports. Low agreement means the two disagree and at
least one of them is unreliable for that model.

Only rows where BOTH sides are a clean support/undermine count toward the
agreement rate; rows with a missing/unparseable self-report
(parse_ok=False) are counted separately as n_reported_missing, not silently
folded into either bucket.

Reads metrics/llm_sample_stances.csv (written by stance_classifier.py) and
writes metrics/stance_crosscheck.csv + metrics/stance_agreement.json.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("stance_crosscheck")

VALID_STANCES = ("support", "undermine")
AGREEMENT_JSON = "stance_agreement.json"
CROSSCHECK_CSV = "stance_crosscheck.csv"


def compute_crosscheck(df: pd.DataFrame) -> dict:
    """Pure confusion-matrix computation over a stances DataFrame with
    llm_reported_stance and predicted_stance columns. Returns the agreement
    summary dict (also what gets saved to stance_agreement.json)."""
    reported_col = df["llm_reported_stance"] if "llm_reported_stance" in df.columns else pd.Series(dtype=str)
    predicted_col = df["predicted_stance"]

    valid_mask = reported_col.isin(VALID_STANCES) & predicted_col.isin(VALID_STANCES)
    valid = df.loc[valid_mask]
    n_valid = int(len(valid))
    n_reported_missing = int((~reported_col.isin(VALID_STANCES)).sum())

    agreement_rate = (
        float((valid["llm_reported_stance"] == valid["predicted_stance"]).mean())
        if n_valid else None
    )
    confusion = (
        valid.groupby(["llm_reported_stance", "predicted_stance"]).size()
        .reset_index(name="n")
        .sort_values(["llm_reported_stance", "predicted_stance"])
    )
    return {
        "n_valid": n_valid,
        "n_reported_missing": n_reported_missing,
        "agreement_rate": agreement_rate,
        "confusion": confusion,  # DataFrame, converted to records for JSON below
    }


def main(config: dict = None) -> dict:
    if config is None:
        config = load_config()
    metrics_dir = resolve(config["paths"]["metrics_dir"])

    df = load_csv(os.path.join(metrics_dir, "llm_sample_stances.csv"))
    result = compute_crosscheck(df)

    confusion_df = result.pop("confusion")
    save_csv(confusion_df, os.path.join(metrics_dir, CROSSCHECK_CSV))
    result["confusion"] = confusion_df.to_dict("records")
    save_json(result, os.path.join(metrics_dir, AGREEMENT_JSON))

    if result["agreement_rate"] is None:
        logger.warning(
            "No rows with both a clean self-reported and predicted stance - "
            "agreement rate is null. Check llm_sample_stances.csv."
        )
    else:
        logger.info(
            f"Stance cross-check: agreement={result['agreement_rate']:.3f} "
            f"over {result['n_valid']} rows ({result['n_reported_missing']} rows "
            f"with missing/unparseable self-report excluded)."
        )
    return result


if __name__ == "__main__":
    main()
