"""Unit tests for generation resume/reconciliation (filesystem-only, no GPU).

Run: python -m pytest tests/ -q   (from the project root)

These lock in _reconcile_raw_csv's contract: after it runs, every claim whose
per-claim JSON exists ("done") has its rows in raw_generations.csv - fixing
the silent data-loss when a crash lands between a JSON write and the next
checkpoint flush (rows were only flushed every N claims).
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.generate_llama import (
    _flush_batch,
    _reconcile_raw_csv,
    _rows_from_claim_json,
)


def _write_claim_json(model_dir, claim_id, samples=("alpha", "beta")):
    path = os.path.join(model_dir, f"{claim_id}_samples.json")
    with open(path, "w") as f:
        json.dump({"claim_id": claim_id, "claim_text": f"claim {claim_id}",
                   "samples": list(samples)}, f)
    return path


def _csv_claim_ids(path):
    return set(pd.read_csv(path)["claim_id"].astype(int))


def test_csv_missing_full_rebuild(tmp_path):
    model_dir = str(tmp_path)
    csv_path = os.path.join(model_dir, "raw_generations.csv")
    _write_claim_json(model_dir, 1, ("a", "b"))
    _write_claim_json(model_dir, 2, ("c",))

    _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={1, 2})

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["claim_id", "sample_id", "generated_text"]
    assert _csv_claim_ids(csv_path) == {1, 2}
    # row counts per claim match the JSON sample counts
    assert len(df[df.claim_id == 1]) == 2
    assert len(df[df.claim_id == 2]) == 1


def test_partial_backfill_appends_only_missing_claims(tmp_path):
    model_dir = str(tmp_path)
    csv_path = os.path.join(model_dir, "raw_generations.csv")
    # Claim 1 crashed AFTER its JSON write but BEFORE the flush -> JSON only.
    _write_claim_json(model_dir, 1, ("recovered",))
    # Claim 2's JSON + rows both made it to disk before the crash.
    _write_claim_json(model_dir, 2, ("safe", "rows"))
    _flush_batch(
        [{"claim_id": 2, "sample_id": 0, "generated_text": "safe"},
         {"claim_id": 2, "sample_id": 1, "generated_text": "rows"}],
        csv_path, write_header=True,
    )

    _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={1, 2})

    df = pd.read_csv(csv_path)
    # Claim 1 recovered, claim 2 NOT duplicated
    assert _csv_claim_ids(csv_path) == {1, 2}
    assert len(df[df.claim_id == 1]) == 1
    assert df[df.claim_id == 1]["generated_text"].iloc[0] == "recovered"
    assert len(df[df.claim_id == 2]) == 2
    # Appended without a duplicate header row
    assert df["generated_text"].notna().all()


def test_complete_csv_is_noop(tmp_path):
    model_dir = str(tmp_path)
    csv_path = os.path.join(model_dir, "raw_generations.csv")
    _write_claim_json(model_dir, 1, ("x",))
    _flush_batch([{"claim_id": 1, "sample_id": 0, "generated_text": "x"}],
                 csv_path, write_header=True)
    before = open(csv_path).read()

    _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={1})

    assert open(csv_path).read() == before  # byte-identical, nothing appended


def test_header_only_csv_recovers_all_done_claims(tmp_path):
    model_dir = str(tmp_path)
    csv_path = os.path.join(model_dir, "raw_generations.csv")
    _write_claim_json(model_dir, 1, ("a",))
    _write_claim_json(model_dir, 2, ("b",))
    # Crash flushed just the header before any rows.
    _flush_batch([], csv_path, write_header=True) if False else pd.DataFrame(
        columns=["claim_id", "sample_id", "generated_text"]
    ).to_csv(csv_path, index=False)

    _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={1, 2})

    assert _csv_claim_ids(csv_path) == {1, 2}


def test_corrupt_json_warns_but_doesnt_crash(tmp_path, caplog):
    # NOTE: generate_llama's logger sets propagate=False (by design, to avoid
    # double-logging), so caplog's root handler never sees its records - attach
    # caplog's handler to the module logger directly for this test.
    import logging
    from src.generation.generate_llama import logger as gen_logger
    gen_logger.addHandler(caplog.handler)
    try:
        model_dir = str(tmp_path)
        csv_path = os.path.join(model_dir, "raw_generations.csv")
        with open(os.path.join(model_dir, "3_samples.json"), "w") as f:
            f.write("{not valid json")
        _write_claim_json(model_dir, 4, ("fine",))

        # Missing-CSV path: one corrupt JSON, one good one - no crash, good one recovered.
        _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={3, 4})
        assert _csv_claim_ids(csv_path) == {4}
        assert any("Could not read/parse" in r.message for r in caplog.records)

        # Backfill path with a corrupt JSON also doesn't crash.
        caplog.clear()
        _reconcile_raw_csv(model_dir, csv_path, done_claim_ids={3, 4})
        assert _csv_claim_ids(csv_path) == {4}
        assert any("Could not read/parse" in r.message for r in caplog.records)
    finally:
        gen_logger.removeHandler(caplog.handler)


def test_rows_from_claim_json_roundtrip(tmp_path):
    model_dir = str(tmp_path)
    _write_claim_json(model_dir, 7, ("s0", "s1", "s2"))
    rows = _rows_from_claim_json(model_dir, 7)
    assert rows == [
        {"claim_id": 7, "sample_id": 0, "generated_text": "s0"},
        {"claim_id": 7, "sample_id": 1, "generated_text": "s1"},
        {"claim_id": 7, "sample_id": 2, "generated_text": "s2"},
    ]
    assert _rows_from_claim_json(model_dir, 999) == []  # nonexistent -> []


def test_empty_done_set_is_noop(tmp_path):
    model_dir = str(tmp_path)
    csv_path = os.path.join(model_dir, "raw_generations.csv")
    # No JSONs, no CSV: nothing happens, no file created.
    _reconcile_raw_csv(model_dir, csv_path, done_claim_ids=set())
    assert not os.path.exists(csv_path)
