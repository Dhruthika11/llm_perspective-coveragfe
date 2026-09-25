"""Unit tests for run_pipeline's model-override resolution (no GPU needed).

Run: python -m pytest tests/ -q   (from the project root)

These lock in that model entries with missing sampling keys inherit the
global llm.* defaults instead of KeyError-ing, and that entries missing
name/provider are skipped loudly rather than crashing.
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import run_pipeline


def _write_cfg(tmp_path, models):
    cfg = {
        "llm": {
            "model": "global/model",
            "provider": "huggingface",
            "temperature": 0.9,
            "top_p": 0.8,
            "max_new_tokens": 256,
            "models": models,
        },
        "generation": {},
        "paths": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


def test_partial_model_entry_inherits_globals(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, {"mini": {"name": "org/mini", "provider": "groq"}})
    monkeypatch.setattr(run_pipeline, "_CONFIG_PATH", cfg_path)
    config = run_pipeline.build_config(model="mini")
    assert config["llm"]["model"] == "org/mini"
    assert config["llm"]["provider"] == "groq"
    assert config["llm"]["temperature"] == 0.9   # inherited, not KeyError
    assert config["llm"]["top_p"] == 0.8
    assert config["llm"]["max_new_tokens"] == 256


def test_model_entry_missing_name_or_provider_is_skipped(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, {"broken": {"temperature": 0.1}})
    monkeypatch.setattr(run_pipeline, "_CONFIG_PATH", cfg_path)
    config = run_pipeline.build_config(model="broken")
    # override ignored, global model stands
    assert config["llm"]["model"] == "global/model"
    assert config["llm"]["provider"] == "huggingface"


def test_full_model_entry_overrides_everything(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path, {"full": {
        "name": "org/full", "provider": "groq", "temperature": 0.1,
        "top_p": 0.5, "max_new_tokens": 64, "batch_size": 2,
    }})
    monkeypatch.setattr(run_pipeline, "_CONFIG_PATH", cfg_path)
    config = run_pipeline.build_config(model="full")
    llm = config["llm"]
    assert (llm["model"], llm["provider"], llm["temperature"],
            llm["top_p"], llm["max_new_tokens"]) == ("org/full", "groq", 0.1, 0.5, 64)
    assert config["generation"]["batch_size"] == 2
    # model-namespaced paths still derived (tag for "org/full" is "full")
    assert "/full/" in config["paths"]["metrics_dir"]
