"""Unit tests for io_utils.load_config (no GPU/model needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in two load_config contracts: callers' dicts are never mutated
in place, and a config without llm.model fails loudly instead of silently
deriving paths under a "default_model" tag.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.io_utils import load_config


def _base_cfg():
    return {
        "llm": {"model": "org/some-model", "provider": "huggingface"},
        "paths": {"metrics_dir": "results/metrics/"},
    }


def test_load_config_does_not_mutate_overrides():
    cfg = _base_cfg()
    snapshot = {"llm": dict(cfg["llm"]), "paths": dict(cfg["paths"])}
    loaded = load_config(overrides=cfg)
    # caller's nested dicts untouched...
    assert cfg == snapshot
    # ...while the returned config carries the derived paths.
    assert loaded["paths"]["metrics_dir"] != snapshot["paths"]["metrics_dir"]
    assert "some_model" in loaded["paths"]["metrics_dir"]


def test_load_config_missing_model_raises():
    with pytest.raises(ValueError, match="llm.model"):
        load_config(overrides={"llm": {"provider": "groq"}, "paths": {}})
    with pytest.raises(ValueError, match="llm.model"):
        load_config(overrides={"paths": {}})
