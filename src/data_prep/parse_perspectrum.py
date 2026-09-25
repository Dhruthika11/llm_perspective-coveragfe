"""
Parse the raw Perspectrum JSON files into two flat, easy-to-use tables:

  data/processed/claims.csv        -> claim_id, claim_text, split
  data/processed/perspectives.csv  -> claim_id, perspective_id, text, stance

IMPORTANT — which files you actually need:
  - claims_file (perspectrum_with_answers_v1.0.json): the REAL source of
    claim text + nested perspective clusters. Required.
  - perspective_pool_file (perspective_pool_v1.0.json): maps perspective_id
    -> perspective text. Required.
  - dataset_split_file (dataset_split_v1.0.json): ONLY maps claim_id ->
    "train"/"test"/"dev". It has no claim text at all. Optional - used only
    to fill in the `split` column if present.

Download all three from:
  https://github.com/CogComp/perspectrum/tree/master/data/dataset
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.io_utils import resolve, load_json, save_csv, load_config
from src.utils.logging_utils import get_logger

logger = get_logger("parse_perspectrum")


def main(config: dict = None):
    # Use provided config or load default
    if config is not None:
        pass  # config already loaded with overrides
    else:
        config = load_config()

    raw_dir = resolve(config["dataset"]["raw_dir"])

    def normalize_stance(raw_label: str) -> str:
        """Normalize a raw label to a clean stance."""
        label = str(raw_label).upper()
        if "SUPPORT" in label:
            return "support"
        if "UNDERMINE" in label:
            return "undermine"
        return "unclear"


    def _iter_entries(raw):
        """Normalize a dict-or-list top-level JSON structure into (key, entry) pairs."""
        if isinstance(raw, dict):
            return list(raw.items())
        if isinstance(raw, list):
            return list(enumerate(raw))
        raise TypeError(f"Unexpected top-level JSON type: {type(raw)}")


    def _build_perspective_lookup(persp_raw) -> dict:
        """perspective_pool_v1.0.json entries -> {perspective_id (int): text (str)}.
        Handles both a flat {pid: "text"} mapping and {pid: {"text": "..."}} /
        a list of {"pId": ..., "text": ...} objects.
        """
        lookup = {}
        skipped = 0
        for key, entry in _iter_entries(persp_raw):
            if isinstance(entry, str):
                try:
                    lookup[int(key)] = entry
                except (TypeError, ValueError):
                    skipped += 1
                continue
            if isinstance(entry, dict):
                pid = entry.get("pId", key)
                text = entry.get("text")
                if text is not None:
                    try:
                        lookup[int(pid)] = text
                        continue
                    except (TypeError, ValueError):
                        pass
            skipped += 1
        if skipped:
            logger.warning(f"Skipped {skipped} unrecognized entries in perspective pool file")
        return lookup


    def _load_split_labels(config) -> dict:
        """Optional: dataset_split_v1.0.json maps claim_id -> 'train'/'test'/'dev'.
        Returns {} if the file isn't configured or can't be found/parsed.
        """
        split_file = config["dataset"].get("dataset_split_file")
        if not split_file:
            return {}
        raw_dir = resolve(config["dataset"]["raw_dir"])
        split_path = os.path.join(raw_dir, split_file)
        if not os.path.exists(split_path):
            return {}
        try:
            split_raw = load_json(split_path)
        except Exception as e:
            logger.warning(f"Could not read {split_path}: {e}")
            return {}

        labels = {}
        for key, value in _iter_entries(split_raw):
            if isinstance(value, str):
                try:
                    labels[int(key)] = value
                except (TypeError, ValueError):
                    continue
        logger.info(f"Loaded {len(labels)} split labels from {split_path}")
        return labels


    def parse(config: dict) -> tuple:
        raw_dir = resolve(config["dataset"]["raw_dir"])
        claims_path = os.path.join(raw_dir, config["dataset"]["claims_file"])
        persp_path = os.path.join(raw_dir, config["dataset"]["perspective_pool_file"])

        logger.info(f"Loading claims from {claims_path}")
        claims_raw = load_json(claims_path)

        logger.info(f"Loading perspective pool from {persp_path}")
        persp_raw = load_json(persp_path)

        persp_text_lookup = _build_perspective_lookup(persp_raw)
        split_labels = _load_split_labels(config)

        claims_rows = []
        persp_rows = []
        skipped_claims = 0

        for key, entry in _iter_entries(claims_raw):
            if not isinstance(entry, dict):
                skipped_claims += 1
                continue

            claim_id = entry.get("cId", key)
            try:
                claim_id = int(claim_id)
            except (TypeError, ValueError):
                skipped_claims += 1
                continue

            claim_text = entry.get("text") or entry.get("claim_text")
            if claim_text is None:
                skipped_claims += 1
                continue

            claims_rows.append({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "split": split_labels.get(claim_id, entry.get("split", "unknown")),
            })

            for cluster in entry.get("perspectives", []):
                stance = normalize_stance(
                    cluster.get("stance_label_3") or cluster.get("stance_label_5") or "unclear"
                )
                for pid in cluster.get("pids", []):
                    try:
                        pid = int(pid)
                    except (TypeError, ValueError):
                        continue
                    text = persp_text_lookup.get(pid)
                    if text is None:
                        continue
                    persp_rows.append({
                        "claim_id": claim_id,
                        "perspective_id": pid,
                        "text": text,
                        "stance": stance,
                    })

        if skipped_claims:
            logger.warning(f"Skipped {skipped_claims} entries in claims file that had no usable claim text")

        claims_df = pd.DataFrame(claims_rows).drop_duplicates("claim_id")
        persp_df = pd.DataFrame(persp_rows).drop_duplicates(["claim_id", "perspective_id"])

        logger.info(f"Parsed {len(claims_df)} claims and {len(persp_df)} perspectives")

        if len(claims_df) == 0:
            raise ValueError(
                f"Parsed 0 claims from {claims_path}. Make sure this file is "
                "perspectrum_with_answers_v1.0.json (claim text + nested "
                "perspectives), NOT dataset_split_v1.0.json (which only has "
                "train/test/dev labels, no claim text). Download it from:\n"
                "  https://github.com/CogComp/perspectrum/tree/master/data/dataset\n\n"
                "To inspect what you currently have, run:\n"
                f"  python3 -c \"import json; d=json.load(open('{claims_path}')); "
                "print(type(d)); print(list(d.items())[0] if isinstance(d, dict) else d[0])\""
            )
        if len(persp_df) == 0:
            logger.warning(
                "Parsed 0 perspectives. Claims were found, but perspective ids "
                "referenced in the claims file didn't match any entry in the "
                "perspective pool file - double check both files came from the same dataset version/release."
            )

        return claims_df, persp_df


    processed_dir = resolve(config["dataset"]["processed_dir"])
    claims_df, persp_df = parse(config)

    save_csv(claims_df, os.path.join(processed_dir, "claims.csv"))
    save_csv(persp_df, os.path.join(processed_dir, "perspectives.csv"))
    logger.info(f"Saved processed CSVs to {processed_dir}")


if __name__ == "__main__":
    main()