"""
Validates the cosine-similarity threshold (analysis.similarity_threshold)
empirically instead of treating tau=0.75 as a given.

Why this exists (fix for "arbitrary threshold"):
  coverage_score.py's "covered" decision is entirely a step function of
  tau. Without evidence that tau=0.75 is where BGE cosine similarity
  actually separates "same perspective" pairs from "different perspective"
  pairs, the coverage number is a heuristic, not a validated measurement.

What this script does:
  1. Samples n_llm_samples LLM-generated outputs (config: calibration.seed
     for reproducibility), and for each, pairs it with the human
     perspectives on the SAME claim (capped at max_perspectives_per_sample).
     Same-claim pairing matters: it reproduces the actual comparison the
     coverage metric makes (LLM sample vs. human perspective on the same
     claim), including both matching- and opposing-stance pairs, rather than
     random cross-claim pairs that would trivially separate at near-zero
     similarity and make the threshold look better calibrated than it is.
  2. Labels each pair as "same perspective" (1) or not (0), using either:
       - judge_mode: "llm"    -> a strict-criteria LLM-as-judge call
       - judge_mode: "manual" -> exports pairs with a blank human_label
         column for a person to fill in, then re-run with --score-only
  3. Computes the BGE cosine similarity for every pair (same embedding
     model/settings as the rest of the pipeline).
  4. Plots the ROC and Precision-Recall curves of similarity vs. label, and
     reports the threshold that maximizes Youden's J (tpr - fpr) and the
     threshold that maximizes F1, saving both plus ROC-AUC/PR-AUC to
     results/metrics/threshold_calibration.json.

coverage_score.py reads that JSON (via get_similarity_threshold below) and
uses the empirically recommended tau automatically, falling back to the
config default - loudly logged as unvalidated - only if calibration hasn't
been run yet.

Usage:
    python -m src.analysis.threshold_calibration
    python -m src.analysis.threshold_calibration --score-only   # after manually filling in human_label
"""
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import load_config, resolve, load_csv, save_csv, save_json, load_json
from src.utils.logging_utils import get_logger
from src.embeddings.embed_model import BGEEmbedder

logger = get_logger("threshold_calibration")

CALIBRATION_JSON = "threshold_calibration.json"
CALIBRATION_PAIRS_CSV = "threshold_calibration_pairs.csv"
CALIBRATION_FIGURE = "threshold_roc_pr.png"

JUDGE_PROMPT_TEMPLATE = """You are a strict, careful annotator comparing two statements of opinion \
on the same claim.

Claim: "{claim}"

Statement A: "{text_a}"
Statement B: "{text_b}"

Do Statement A and Statement B express the SAME perspective? They count as \
the SAME perspective only if they share BOTH:
  (1) the same stance (support vs. oppose) on the claim, AND
  (2) substantially the same underlying reason/justification.
Two statements with the same stance but different reasoning are NOT the \
same perspective. Two statements with different stances are NOT the same \
perspective.

Answer with exactly one word: YES or NO."""


def build_candidate_pairs(claims_df, perspectives_df, generations_df, n_llm_samples: int,
                           max_perspectives_per_sample: int, seed: int) -> pd.DataFrame:
    claim_text_by_id = dict(zip(claims_df["claim_id"], claims_df["claim_text"]))

    # Sameness of the pair sample is driven by pandas random_state=seed below,
    # so the candidate set is reproducible without a separate random.Random.

    eligible_claim_ids = set(perspectives_df["claim_id"]).intersection(set(generations_df["claim_id"]))
    gen_pool = generations_df[generations_df["claim_id"].isin(eligible_claim_ids)]

    n_sample = min(n_llm_samples, len(gen_pool))
    sampled_gen = gen_pool.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    rows = []
    for _, gen_row in sampled_gen.iterrows():
        claim_id = gen_row["claim_id"]
        claim_persps = perspectives_df[perspectives_df["claim_id"] == claim_id]
        if len(claim_persps) > max_perspectives_per_sample:
            claim_persps = claim_persps.sample(n=max_perspectives_per_sample, random_state=seed)

        llm_text = gen_row.get("combined_text", gen_row.get("generated_text"))
        for _, p_row in claim_persps.iterrows():
            rows.append({
                "claim_id": claim_id,
                "claim_text": claim_text_by_id.get(claim_id, ""),
                "llm_sample_id": gen_row["sample_id"],
                "llm_text": llm_text,
                "perspective_id": p_row["perspective_id"],
                "human_text": p_row["text"],
                "human_stance": p_row["stance"],
            })

    pairs_df = pd.DataFrame(rows)
    logger.info(
        f"Built {len(pairs_df)} candidate pairs from {n_sample} LLM samples "
        f"across {pairs_df['claim_id'].nunique() if len(pairs_df) else 0} claims"
    )
    return pairs_df


def label_pairs_with_llm_judge(pairs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    from src.generation.llm_config import get_llm_client

    judge_config = dict(config)
    # Judge calls should be byte-identical per pair: drop the generation
    # seed so no per-sample derived seed is forwarded (at temperature=0 the
    # judge is deterministic anyway - this just keeps the request payload
    # explicit and independent of generation-side seeding).
    judge_config["llm"] = dict(config["llm"])
    judge_config["llm"].pop("seed", None)
    judge_model_key = config.get("calibration", {}).get("judge_model")
    if judge_model_key and judge_model_key in config.get("llm", {}).get("models", {}):
        model_cfg = config["llm"]["models"][judge_model_key]
        judge_config["llm"]["model"] = model_cfg["name"]
        judge_config["llm"]["provider"] = model_cfg["provider"]
        # Groq-side reasoning control (e.g. "none" to disable thinking on the
        # qwen judge). get_llm_client forwards this to GroqLlamaClient; absent
        # means default API behavior. Judge temperature/token cap still come
        # from calibration.judge_temperature/judge_max_new_tokens below.
        if model_cfg.get("reasoning_effort") is not None:
            judge_config["llm"]["reasoning_effort"] = model_cfg["reasoning_effort"]
        logger.info(f"Using designated judge model for calibration: {model_cfg['name']}")

    client = get_llm_client(judge_config)
    temperature = config["calibration"].get("judge_temperature", 0.0)
    max_new_tokens = config["calibration"].get("judge_max_new_tokens", 10)

    labels = []
    raw_responses = []
    for _, row in pairs_df.iterrows():
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            claim=row["claim_text"], text_a=row["llm_text"], text_b=row["human_text"],
        )
        try:
            response = client.generate_k_samples(
                prompt, k=1, temperature=temperature, top_p=1.0, max_new_tokens=max_new_tokens,
            )[0]
        except Exception as e:
            logger.warning(f"LLM-judge call failed for pair (claim {row['claim_id']}): {e}")
            response = ""

        raw_responses.append(response)
        answer = response.strip().upper()
        labels.append(1 if answer.startswith("YES") else (0 if answer.startswith("NO") else None))

    pairs_df = pairs_df.copy()
    pairs_df["llm_judge_raw"] = raw_responses
    pairs_df["llm_judge_label"] = labels

    n_unparsed = pairs_df["llm_judge_label"].isna().sum()
    if n_unparsed:
        logger.warning(
            f"{n_unparsed}/{len(pairs_df)} LLM-judge responses were not a clean YES/NO and "
            f"were dropped from calibration - inspect llm_judge_raw in the pairs CSV."
        )
    return pairs_df


def compute_similarities(pairs_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    embedder = BGEEmbedder(
        model_name=config["embedding"]["model"],
        normalize=config["embedding"]["normalize"],
    )
    llm_emb = embedder.embed(pairs_df["llm_text"].tolist(), batch_size=config["embedding"]["batch_size"])
    human_emb = embedder.embed(pairs_df["human_text"].tolist(), batch_size=config["embedding"]["batch_size"])

    # Rows are expected L2-normalized (cosine = row-wise dot product), but
    # ensure_normalized() keeps this true even if embedding.normalize is off.
    from src.analysis.similarity_matrix import ensure_normalized
    llm_emb = ensure_normalized(llm_emb, "calibration LLM embeddings")
    human_emb = ensure_normalized(human_emb, "calibration human embeddings")
    sims = np.sum(llm_emb * human_emb, axis=1)
    pairs_df = pairs_df.copy()
    pairs_df["cosine_similarity"] = sims
    return pairs_df


def analyze_roc_pr(pairs_df: pd.DataFrame, label_col: str, default_tau: float, figures_dir: str) -> dict:
    labeled = pairs_df.dropna(subset=[label_col, "cosine_similarity"])
    y_true = labeled[label_col].astype(int).values
    y_score = labeled["cosine_similarity"].values

    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both positive and negative labeled pairs to calibrate a threshold "
            f"(got {n_pos} positive, {n_neg} negative). Sample more pairs or check labeling."
        )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    tau_youden = float(roc_thresholds[int(np.argmax(youden_j))])

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    # precision/recall have one more element than pr_thresholds; align by dropping the last
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
        0.0,
    )
    tau_f1max = float(pr_thresholds[int(np.argmax(f1))]) if len(pr_thresholds) else default_tau

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    best_idx = int(np.argmax(youden_j))
    axes[0].scatter([fpr[best_idx]], [tpr[best_idx]], color="red", zorder=5,
                     label=f"Youden-optimal tau={tau_youden:.2f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve: BGE cosine similarity vs. judged label")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
    best_f1_idx = int(np.argmax(f1)) if len(f1) else 0
    if len(pr_thresholds):
        axes[1].scatter([recall[best_f1_idx]], [precision[best_f1_idx]], color="red", zorder=5,
                         label=f"F1-optimal tau={tau_f1max:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left", fontsize=8)

    fig.suptitle(f"Threshold calibration (n={len(labeled)} pairs, "
                 f"{n_pos} positive / {n_neg} negative) - config default tau={default_tau}")
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, CALIBRATION_FIGURE)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    recommended_tau = tau_f1max  # F1-optimal is the standard choice for an imbalanced retrieval-style decision
    result = {
        "n_pairs_labeled": int(len(labeled)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tau_youden_j": tau_youden,
        "tau_f1_max": tau_f1max,
        "recommended_tau": float(recommended_tau),
        "config_default_tau": float(default_tau),
        "figure_path": os.path.relpath(fig_path, resolve("")),
    }
    return result


def get_similarity_threshold(config: dict) -> tuple[float, dict]:
    """Single source of truth used by coverage_score.py: prefer the
    empirically calibrated tau if it exists and analysis.use_calibrated_threshold
    is true, else fall back to the config heuristic (loudly logged as such).

    Returns (threshold, meta) where meta describes provenance.
    """
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    calibration_path = os.path.join(metrics_dir, CALIBRATION_JSON)
    use_calibrated = config.get("analysis", {}).get("use_calibrated_threshold", True)
    default_tau = config["analysis"]["similarity_threshold"]

    if use_calibrated and os.path.exists(calibration_path):
        calib = load_json(calibration_path)
        tau = calib["recommended_tau"]
        logger.info(
            f"Using empirically calibrated threshold tau={tau:.3f} "
            f"(F1-optimal on {calib['n_pairs_labeled']} labeled pairs, "
            f"ROC-AUC={calib['roc_auc']:.3f}, PR-AUC={calib['pr_auc']:.3f}). "
            f"See results/metrics/{CALIBRATION_JSON} and "
            f"results/figures/{CALIBRATION_FIGURE}."
        )
        return tau, {"source": "calibrated", **calib}

    logger.warning(
        f"No threshold calibration found (or use_calibrated_threshold=false) - "
        f"falling back to UNVALIDATED heuristic tau={default_tau} from config.yaml. "
        f"Run `python -m src.analysis.threshold_calibration` to validate this threshold "
        f"against labeled pairs before trusting the coverage numbers."
    )
    return default_tau, {"source": "config_default_unvalidated", "config_default_tau": default_tau}


def main(config: dict = None, score_only: bool = False):
    if config is None:
        config = load_config()
    calib_cfg = config.get("calibration", {})
    metrics_dir = resolve(config["paths"]["metrics_dir"])
    figures_dir = resolve(config["paths"]["figures_dir"])
    default_tau = config["analysis"]["similarity_threshold"]
    pairs_path = os.path.join(metrics_dir, CALIBRATION_PAIRS_CSV)

    if score_only:
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"{pairs_path} not found - run without score_only first.")
        pairs_df = load_csv(pairs_path)
        label_col = "human_label" if "human_label" in pairs_df.columns and pairs_df["human_label"].notna().any() \
            else "llm_judge_label"
    else:
        processed_dir = resolve(config["dataset"]["processed_dir"])
        claims_df = load_csv(os.path.join(processed_dir, "claims.csv"))
        perspectives_df = load_csv(os.path.join(processed_dir, "perspectives.csv"))
        generations_df = load_csv(resolve(config["paths"]["llm_outputs_processed"]))

        pairs_df = build_candidate_pairs(
            claims_df, perspectives_df, generations_df,
            n_llm_samples=calib_cfg.get("n_llm_samples", 50),
            max_perspectives_per_sample=calib_cfg.get("max_perspectives_per_sample", 6),
            seed=calib_cfg.get("seed", 42),
        )
        pairs_df = compute_similarities(pairs_df, config)

        judge_mode = calib_cfg.get("judge_mode", "llm")
        if judge_mode == "llm":
            pairs_df = label_pairs_with_llm_judge(pairs_df, config)
            label_col = "llm_judge_label"
        elif judge_mode == "manual":
            pairs_df["human_label"] = np.nan  # blank column for a person to fill in (0/1)
            save_csv(pairs_df, pairs_path)
            logger.info(
                f"judge_mode=manual: wrote {len(pairs_df)} unlabeled pairs to {pairs_path}. "
                f"Fill in the human_label column (1=same perspective, 0=different) and re-run "
                f"with score_only=True (or `--score-only` from the CLI)."
            )
            return None
        else:
            raise ValueError(f"Unknown calibration.judge_mode: {judge_mode!r} (use 'llm' or 'manual')")

        save_csv(pairs_df, pairs_path)

    result = analyze_roc_pr(pairs_df, label_col, default_tau, figures_dir)
    save_json(result, os.path.join(metrics_dir, CALIBRATION_JSON))

    logger.info(
        f"Calibration complete: ROC-AUC={result['roc_auc']:.3f}, PR-AUC={result['pr_auc']:.3f}, "
        f"tau_youden={result['tau_youden_j']:.3f}, tau_f1max={result['tau_f1_max']:.3f}, "
        f"recommended_tau={result['recommended_tau']:.3f} (config default was {default_tau})."
    )
    logger.info(f"Saved pairs -> {pairs_path}")
    logger.info(f"Saved figure -> {os.path.join(figures_dir, CALIBRATION_FIGURE)}")
    logger.info(f"Saved summary -> {os.path.join(metrics_dir, CALIBRATION_JSON)}")
    return result


def _cli():
    """Thin CLI entry point - only parses sys.argv when this module is run
    standalone (`python -m src.analysis.threshold_calibration`). main() itself
    takes plain arguments so it's safe to call in-process (e.g. from
    run_pipeline.py) without main() trying to re-parse the caller's own argv."""
    parser = argparse.ArgumentParser(description="Calibrate the coverage similarity threshold.")
    parser.add_argument(
        "--score-only", action="store_true",
        help="Skip pair generation/judging; read an existing pairs CSV (with a filled-in "
             "human_label column, e.g. after manual labeling) and just compute ROC/PR.",
    )
    parser.add_argument(
        "--model", choices=["1.5b", "3b", "7b", "12b", "32b"], default=None,
        help="LLM model size to use (same ladder choices as run_pipeline.py). "
             "Selects the model-namespaced metrics/figures dirs so calibration "
             "lands next to that model's other results. Omit for the config default.",
    )
    args = parser.parse_args()
    config = None
    if args.model:
        # Lazy import: run_pipeline lazily imports this module inside its
        # stage functions, so importing it here at module level would risk a
        # cycle - resolve only when the flag is actually used.
        from run_pipeline import build_config
        config = build_config(model=args.model)
    main(config=config, score_only=args.score_only)


if __name__ == "__main__":
    _cli()
