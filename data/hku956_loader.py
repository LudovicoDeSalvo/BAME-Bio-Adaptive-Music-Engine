import os
from typing import Dict, List, Tuple

import numpy as np

from data.windows import (
    N_FEATURES,
    compute_window_features,
    make_multichannel_tensor,
    normalize_windows,
    summarize_windows,
)
from utils.common import ensure_dir, load_config, resolve_path


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


def _get_rate(sample_rates: Dict[str, int], key: str, default: int = 1) -> int:
    return int(sample_rates.get(key.lower(), sample_rates.get(key.upper(), default)))


def _resample_signal(signal: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if orig_rate <= 0 or target_rate <= 0 or signal.size <= 1 or orig_rate == target_rate:
        return signal
    duration = (signal.size - 1) / orig_rate
    t_orig = np.linspace(0.0, duration, num=signal.size)
    t_new = np.linspace(0.0, duration, num=int(duration * target_rate) + 1)
    return np.interp(t_new, t_orig, signal)


def _extract_record_tensor(
    data_root: str,
    participant_id: str,
    song_no: int,
    song_id: int,
    window_seconds: int,
    window_stride_seconds: int,
    sample_rates: Dict[str, int],
    max_windows: int | None,
    target_rate: int,
    skip_missing: bool,
) -> np.ndarray | None:
    raw_signals: Dict[str, np.ndarray] = {}
    missing_signals = []
    for signal in SIGNAL_TYPES:
        path = _signal_path(data_root, participant_id, signal, song_no, song_id)
        if not os.path.exists(path):
            missing_signals.append(signal)
            continue
        raw_signals[signal] = _read_csv_fallback(path)

    if missing_signals:
        if skip_missing:
            return None
        raise FileNotFoundError(f"Missing signals {missing_signals} for {participant_id} song_id={song_id}")

    resampled = {}
    for sig, data in raw_signals.items():
        rate = _get_rate(sample_rates, sig, default=target_rate)
        resampled[sig] = _resample_signal(data, rate, target_rate)

    window_size = max(1, int(window_seconds * target_rate))
    stride = max(1, int(window_stride_seconds * target_rate))

    tensor = make_multichannel_tensor(
        resampled,
        SIGNAL_TYPES,
        window_size,
        stride,
        drop_last=False,
        pad_mode="edge",
        nan_policy="mean",
    )
    if max_windows is not None:
        tensor = tensor[: max_windows]
    return tensor


def process_and_cache_data(config_path: str = "configs/config.yaml", skip_missing: bool = False) -> Tuple[str, str]:
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
    ensure_dir(os.path.dirname(physio_cache))
    ensure_dir(os.path.dirname(profile_cache))

    ratings = _load_ratings(ratings_csv)
    personality = _load_personality(personality_csv)

    max_records = limits.get("max_records")
    if max_records:
        ratings = ratings.head(int(max_records))

    window_seconds = int(training.get("window_seconds", 10))
    window_stride_seconds = int(training.get("window_stride_seconds", 5))
    sample_rates = training.get("sample_rate_hz", {})
    normalize_mode = training.get("normalize_mode", "per_record")
    target_rate = _get_rate(sample_rates, "target", default=0)
    if target_rate <= 0:
        vals = [v for k, v in sample_rates.items() if str(k).lower() != "target"]
        target_rate = int(max(vals)) if vals else 1
    max_windows = limits.get("max_windows_per_record")
    if max_windows is not None:
        max_windows = int(max_windows)

    raw_tensors: List[np.ndarray] = []
    features: List[np.ndarray] = []
    per_window_features: List[np.ndarray] = []
    window_counts: List[int] = []
    valence: List[float] = []
    arousal: List[float] = []
    song_ids: List[int] = []
    song_nos: List[int] = []
    participant_ids: List[str] = []

    for _, row in ratings.iterrows():
        participant_id = str(row["participant_id"])
        song_no = int(row["song_no"])
        song_id = int(row["song_id"])
        tensor = _extract_record_tensor(
            data_root,
            participant_id,
            song_no,
            song_id,
            window_seconds,
            window_stride_seconds,
            sample_rates,
            max_windows,
            target_rate,
            skip_missing,
        )
        if tensor is None or tensor.shape[0] == 0:
            continue
        raw_tensors.append(tensor)
        window_counts.append(tensor.shape[0])
        valence.append(float(row.get("valence_rating", 0.0)))
        arousal.append(float(row.get("arousal_rating", 0.0)))
        song_ids.append(song_id)
        song_nos.append(song_no)
        participant_ids.append(participant_id)

    if not raw_tensors:
        raise RuntimeError("No valid records found for physio cache.")

    norm_tensors: List[np.ndarray] = []
    stats = {}
    if normalize_mode == "global":
        stacked = np.concatenate(raw_tensors, axis=0)
        _, stats = normalize_windows(stacked, mode="zscore")
        for t in raw_tensors:
            norm, _ = normalize_windows(t, mode="zscore", stats=stats)
            norm_tensors.append(norm)
    else:
        for t in raw_tensors:
            norm, _ = normalize_windows(t, mode="zscore")
            norm_tensors.append(norm)

    for norm in norm_tensors:
        per_win = compute_window_features(norm)
        per_window_features.append(per_win)
        feat = []
        for idx, _sig in enumerate(SIGNAL_TYPES):
            feat.append(summarize_windows(norm[:, idx, :]))
        features.append(np.concatenate(feat, axis=0))

    max_windows_final = max(window_counts) if window_counts else 0
    padded_windows = np.zeros(
        (len(per_window_features), max_windows_final, len(SIGNAL_TYPES), N_FEATURES), dtype=float
    )
    for idx, wf in enumerate(per_window_features):
        if wf.shape[0] == 0:
            continue
        take = min(wf.shape[0], max_windows_final)
        padded_windows[idx, :take, :, :] = wf[:take]

    feature_names = []
    for signal in SIGNAL_TYPES:
        for name in ["mean", "std", "min", "max", "delta", "slope"]:
            feature_names.append(f"{signal.lower()}_{name}")

    norm_mean = stats.get("mean")
    norm_std = stats.get("std")
    if norm_mean is None:
        norm_mean = np.array([])
    if norm_std is None:
        norm_std = np.array([])

    np.savez(
        physio_cache,
        features=np.asarray(features, dtype=float),
        window_features=padded_windows,
        window_counts=np.asarray(window_counts, dtype=int),
        valence=np.asarray(valence, dtype=float),
        arousal=np.asarray(arousal, dtype=float),
        song_ids=np.asarray(song_ids, dtype=int),
        song_no=np.asarray(song_nos, dtype=int),
        participant_ids=np.asarray(participant_ids, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    personality.to_csv(profile_cache, index=False)
    return physio_cache, profile_cache


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Process HKU956 and build physio cache.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip records with missing signals.")
    args = parser.parse_args()

    physio_cache, profile_cache = process_and_cache_data(config_path=args.config, skip_missing=args.skip_missing)
    print(f">> Saved physio cache to: {physio_cache}")
    print(f">> Saved profile cache to: {profile_cache}")


if __name__ == "__main__":
    _main()
