"""Unit tests for threshold_calibration CLI plumbing (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)
"""
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis import threshold_calibration


def test_cli_model_flag_routes_through_build_config(monkeypatch):
    sentinel = {"llm": {"model": "m"}, "sentinel": True}
    recorded = {}

    fake_run_pipeline = types.ModuleType("run_pipeline")
    fake_run_pipeline.build_config = MagicMock(return_value=sentinel)
    monkeypatch.setitem(sys.modules, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        threshold_calibration, "main",
        lambda config=None, score_only=False: recorded.update(
            config=config, score_only=score_only),
    )
    monkeypatch.setattr(
        sys, "argv", ["threshold_calibration", "--model", "3b", "--score-only"])

    threshold_calibration._cli()

    fake_run_pipeline.build_config.assert_called_once_with(model="3b")
    assert recorded == {"config": sentinel, "score_only": True}


def test_cli_without_model_loads_default(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        threshold_calibration, "main",
        lambda config=None, score_only=False: recorded.update(
            config=config, score_only=score_only),
    )
    monkeypatch.setattr(sys, "argv", ["threshold_calibration"])

    threshold_calibration._cli()

    assert recorded == {"config": None, "score_only": False}
