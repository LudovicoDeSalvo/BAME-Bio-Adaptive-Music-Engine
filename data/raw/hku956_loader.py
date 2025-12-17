import os
from typing import Dict, List, Tuple

import numpy as np

from utils.common import ensure_dir, load_config, resolve_path
from data.raw.windows import summarize_windows, window_signal


SIGNAL_TYPES = ["EDA", "BVP", "TEMP", "HR", "IBI"]


def _read_csv_fallback(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            values = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(float(line.split(",")[0]))
                except Exception:
                    continue
            return np.asarray(values, dtype=float)


def _load_ratings(path: str):
    try:
        import pandas as pd  # type: ignore

        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError("pandas is required to read ratings CSV") from exc


def _load_personality(path: str):
    try:
        import pandas as pd  # type: ignore

        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError("pandas is required to read personality CSV") from exc


def _signal_path(data_root: str, participant_id: str, signal: str, song_no: int, song_id: int) -> str:
    filename = f"{song_no}_{song_id}.csv"
    return os.path.join(data_root, "1. physiological_signals", participant_id, signal, filename)


def _extract_record_features(
    data_root: str,
    participant_id: str,
    song_no: int,
    song_id: int,
    window_seconds: int,
    window_stride_seconds: int,
    sample_rates: Dict[str, int],
    max_windows: int | None,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for signal in SIGNAL_TYPES:
        rate_key = signal.lower()
        rate = int(sample_rates.get(rate_key, 1))
        window_size = max(1, int(window_seconds * rate))
        stride = max(1, int(window_stride_seconds * rate))
        path = _signal_path(data_root, participant_id, signal, song_no, song_id)
        if not os.path.exists(path):
            features.append(np.zeros(5, dtype=float))
            continue
        raw = _read_csv_fallback(path)
        windows = window_signal(raw, window_size, stride)
        if max_windows is not None:
            windows = windows[: max_windows]
        features.append(summarize_windows(windows))
    return np.concatenate(features, axis=0)


def process_and_cache_data(config_path: str = "configs/config.yaml") -> Tuple[str, str]:
    config = load_config(config_path)
    paths = config.get("paths", {})
    training = config.get("training", {})
    limits = training.get("limits", {})

    data_root = resolve_path(paths.get("data_root", "data/raw/HKU956"))
    processed_dir = resolve_path(paths.get("processed_dir", "data/processed"))
    ratings_csv = resolve_path(paths.get("ratings_csv", ""))
    personality_csv = resolve_path(paths.get("personality_csv", ""))
    physio_cache = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    profile_cache = resolve_path(paths.get("profile_cache", "data/processed/profile_cache.csv"))

    ensure_dir(processed_dir)

    ratings = _load_ratings(ratings_csv)
    personality = _load_personality(personality_csv)

    max_records = limits.get("max_records")
    if max_records:
        ratings = ratings.head(int(max_records))

    window_seconds = int(training.get("window_seconds", 10))
    window_stride_seconds = int(training.get("window_stride_seconds", 5))
    sample_rates = training.get("sample_rate_hz", {})
    max_windows = limits.get("max_windows_per_record")
    if max_windows is not None:
        max_windows = int(max_windows)

    features: List[np.ndarray] = []
    valence: List[float] = []
    arousal: List[float] = []
    song_ids: List[int] = []
    participant_ids: List[str] = []

    for _, row in ratings.iterrows():
        participant_id = row["participant_id"]
        song_no = int(row["song_no"])
        song_id = int(row["song_id"])
        feat = _extract_record_features(
            data_root,
            participant_id,
            song_no,
            song_id,
            window_seconds,
            window_stride_seconds,
            sample_rates,
            max_windows,
        )
        features.append(feat)
        valence.append(float(row.get("valence_rating", 0.0)))
        arousal.append(float(row.get("arousal_rating", 0.0)))
        song_ids.append(song_id)
        participant_ids.append(participant_id)

    feature_names = []
    for signal in SIGNAL_TYPES:
        for name in ["mean", "std", "min", "max", "delta"]:
            feature_names.append(f"{signal.lower()}_{name}")

    np.savez(
        physio_cache,
        features=np.asarray(features, dtype=float),
        valence=np.asarray(valence, dtype=float),
        arousal=np.asarray(arousal, dtype=float),
        song_ids=np.asarray(song_ids, dtype=int),
        participant_ids=np.asarray(participant_ids, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
    )

    personality.to_csv(profile_cache, index=False)
    return physio_cache, profile_cache
