"""Small shared helpers: config loading, JSON/CSV I/O, path resolution."""
from copy import deepcopy
import json
import os
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(".", "_").replace("-", "_")


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    if overrides is not None:
        # Deep-copy: the caller's dict must not gain the derived
        # model-namespaced path keys as a side effect of loading.
        config = deepcopy(overrides)
    else:
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    model_name = config.get("llm", {}).get("model")
    if not model_name:
        raise ValueError(
            "Config has no llm.model set - cannot derive the model-namespaced "
            "output paths (results/<model_tag>/..., data/llm_outputs/<model_tag>/...)."
        )
    model_tag = get_model_tag(model_name)
    
    paths = config.get("paths", {})
    paths["llm_outputs_processed"] = f"data/llm_outputs/{model_tag}/processed_generations.csv"
    paths["embeddings_dir"] = f"results/{model_tag}/embeddings/"
    paths["metrics_dir"] = f"results/{model_tag}/metrics/"
    paths["figures_dir"] = f"results/{model_tag}/figures/"
    paths["summary_file"] = f"results/{model_tag}/final_report_summary.json"
    
    return config


def resolve(path: str) -> str:
    """Turn a repo-relative path from config.yaml into an absolute path."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
