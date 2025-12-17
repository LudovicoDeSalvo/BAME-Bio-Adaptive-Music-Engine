import json
import os
import random
from pathlib import Path


def project_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root(), path)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = resolve_path(path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
