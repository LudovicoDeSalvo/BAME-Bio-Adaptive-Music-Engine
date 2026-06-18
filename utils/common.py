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


def get_device():
    """Single source of truth for the compute device."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize(x, axis: int = -1, eps: float = 1e-8):
    """Project vectors onto the unit sphere (numpy).

    Used to put every MERT/action vector that enters FAISS, the world model,
    or the replay buffer onto one shared manifold. Physio (z-scored) vectors
    are NOT passed through this.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on for reproducible runs.

    Seeds ``random``, ``numpy``, and (if available) torch CPU+CUDA, and sets
    cuDNN to deterministic mode. Call this at the top of every train_* entry
    point and pass the same seed into ``env.reset(seed=...)``.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_npz_checked(path: str, required_keys):
    """Load an .npz and validate required keys are present (boundary check)."""
    import numpy as np

    data = np.load(path, allow_pickle=True)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"{path} missing keys {missing}; has {list(data.files)}"
        )
    return data


def require_columns(df, columns, source: str = "dataframe") -> None:
    """Validate that a DataFrame contains the expected columns."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{source} missing columns {missing}; has {list(df.columns)}")


def parse_clip_id(clip_id):
    """Parse a clip id ``"{sid}_{pid}_{sno}_{chunk}"`` into its 4 fields.

    Single source of truth for the clip-id contract (previously re-implemented
    with positional ``str.split('_')`` in faiss/simulator/context/inference).
    Uses ``rsplit('_', 3)`` so a ``song_id`` containing underscores is preserved
    intact, and validates that the trailing ``chunk`` field is an integer.

    Returns ``(sid, pid, sno, chunk: int)``. Raises ``ValueError`` if the id does
    not have the 4 trailing fields or the chunk is non-numeric — callers that
    accept already-unique / non-conforming ids should catch that and treat the
    raw id as its own key.
    """
    parts = str(clip_id).rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"clip_id {clip_id!r} does not have 4 '_'-separated fields")
    sid, pid, sno, chunk = parts
    return sid, pid, sno, int(chunk)  # int() raises ValueError on a bad chunk
