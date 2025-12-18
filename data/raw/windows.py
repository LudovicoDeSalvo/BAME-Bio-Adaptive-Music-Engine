import numpy as np

N_FEATURES = 6  # mean, std, min, max, delta, slope


def _sanitize_array(arr: np.ndarray, nan_policy: str) -> np.ndarray:
    """
    Replace NaN/Inf according to policy. Supports any array shape (1D to 3D+).
    For 3D with shape (windows, channels, time), nan_policy='mean' is per-channel.
    """
    arr = np.asarray(arr, dtype=float)
    if nan_policy == "propagate":
        return arr
    if nan_policy == "mean":
        if arr.ndim == 3:
            with np.errstate(all="ignore"):
                channel_mean = np.nanmean(arr, axis=(0, 2))
            channel_mean = np.nan_to_num(channel_mean, nan=0.0, posinf=0.0, neginf=0.0)
            return np.where(np.isfinite(arr), arr, channel_mean[None, :, None])
        if arr.ndim == 2:
            with np.errstate(all="ignore"):
                col_mean = np.nanmean(arr, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
            return np.where(np.isfinite(arr), arr, col_mean[None, :])
        finite = arr[np.isfinite(arr)]
        fill_value = float(finite.mean()) if finite.size else 0.0
        return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    # default: zero
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _pad_window(window: np.ndarray, target: int, pad_mode: str, pad_value: float) -> np.ndarray:
    if window.shape[0] >= target:
        return window[:target]
    pad_width = target - window.shape[0]
    if pad_mode == "edge":
        if window.shape[0] == 0:
            return np.zeros(target, dtype=float)
        return np.pad(window, (0, pad_width), mode="edge")
    if pad_mode in ("zero", "constant"):
        return np.pad(window, (0, pad_width), mode="constant", constant_values=pad_value)
    raise ValueError(f"Unsupported pad_mode: {pad_mode}")


def window_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    *,
    drop_last: bool = False,
    pad_mode: str = "edge",
    pad_value: float = 0.0,
    nan_policy: str = "zero",
) -> np.ndarray:
    """
    Segment a 1D signal into fixed windows.

    Returns shape: (n_windows, window_size).
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if signal is None:
        return np.empty((0, window_size), dtype=float)
    signal = _sanitize_array(signal, nan_policy).reshape(-1)
    if stride <= 0:
        stride = window_size

    n = signal.shape[0]
    if n == 0:
        return np.empty((0, window_size), dtype=float)

    if n < window_size:
        if drop_last:
            return np.empty((0, window_size), dtype=float)
        return _pad_window(signal, window_size, pad_mode, pad_value)[None, :]

    windows = []
    last_start = None
    for start in range(0, n - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
        last_start = start

    last_covered_end = 0 if last_start is None else last_start + window_size
    if not drop_last and last_covered_end < n:
        tail_start = max(last_covered_end, n - window_size)
        tail = signal[tail_start:]
        windows.append(_pad_window(tail, window_size, pad_mode, pad_value))

    return np.asarray(windows, dtype=float)


def window_multimodal(
    signals: dict[str, np.ndarray],
    window_size: int,
    stride: int,
    *,
    drop_last: bool = False,
    pad_mode: str = "edge",
    pad_value: float = 0.0,
    nan_policy: str = "zero",
) -> dict[str, np.ndarray]:
    """
    Window multiple signals ensuring aligned window counts.

    Trims all signals to the minimum length before windowing.
    Assumes signals are already aligned in time (HKU956 is pre-aligned).
    """
    if not signals:
        return {}
    lengths = []
    for name, sig in signals.items():
        if sig is None:
            raise ValueError(f"Signal '{name}' is missing.")
        lengths.append(np.asarray(sig).size)
    if not lengths:
        return {k: np.empty((0, window_size), dtype=float) for k in signals}
    min_len = min(lengths)

    output: dict[str, np.ndarray] = {}
    for name, sig in signals.items():
        arr = _sanitize_array(sig, nan_policy).reshape(-1)[:min_len]
        output[name] = window_signal(
            arr,
            window_size,
            stride,
            drop_last=drop_last,
            pad_mode=pad_mode,
            pad_value=pad_value,
            nan_policy=nan_policy,
        )
    # Ensure same number of windows by trimming to the smallest count.
    min_windows = min(win.shape[0] for win in output.values())
    if min_windows == 0:
        return {k: np.empty((0, window_size), dtype=float) for k in output}
    for key in output:
        output[key] = output[key][:min_windows]
    return output


def make_multichannel_tensor(
    signals: dict[str, np.ndarray],
    channel_order: list[str],
    window_size: int,
    stride: int,
    *,
    drop_last: bool = False,
    pad_mode: str = "edge",
    pad_value: float = 0.0,
    nan_policy: str = "zero",
) -> np.ndarray:
    """
    Build a (n_windows, n_channels, window_size) tensor in a fixed channel order.
    """
    aligned = window_multimodal(
        signals,
        window_size,
        stride,
        drop_last=drop_last,
        pad_mode=pad_mode,
        pad_value=pad_value,
        nan_policy=nan_policy,
    )
    missing = [c for c in channel_order if c not in aligned]
    if missing:
        raise ValueError(f"Missing channels in aligned signals: {missing}")
    # If no windows survived trimming, return an empty tensor with the expected shape.
    if not aligned or next(iter(aligned.values())).shape[0] == 0:
        return np.empty((0, len(channel_order), window_size), dtype=float)
    return np.stack([aligned[name] for name in channel_order], axis=1)


def compute_window_features(windows: np.ndarray, nan_policy: str = "zero") -> np.ndarray:
    """
    Compute per-window statistics.

    Supports shapes (n_windows, window_size) or (n_windows, n_channels, window_size).
    Returns shape (n_windows, n_channels, n_features) where n_channels=1 for 2D input.
    """
    arr = np.asarray(windows, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError("windows must have shape (n_windows, window_size) or (n_windows, n_channels, window_size)")

    arr = _sanitize_array(arr, nan_policy)
    last = arr[..., -1]
    first = arr[..., 0]
    length = arr.shape[-1]
    denom = max(length - 1, 1)

    feats = np.stack(
        [
            np.mean(arr, axis=-1),
            np.std(arr, axis=-1),
            np.min(arr, axis=-1),
            np.max(arr, axis=-1),
            last - first,
            (last - first) / denom,
        ],
        axis=-1,
    )
    return feats  # (n_windows, n_channels, N_FEATURES)


def normalize_windows(
    windows: np.ndarray,
    *,
    mode: str = "zscore",
    stats: dict | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    Normalize windows per-channel across all windows/time.
    Returns normalized windows and the statistics used.
    """
    arr = np.asarray(windows, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError("windows must have shape (n_windows, window_size) or (n_windows, n_channels, window_size)")

    if stats is None:
        stats = {}
    axes = (0, 2)  # average over windows and time, per channel

    if mode == "zscore":
        mean = np.asarray(stats.get("mean", arr.mean(axis=axes)), dtype=float)
        std = np.asarray(stats.get("std", arr.std(axis=axes)), dtype=float)
        norm = (arr - mean[None, :, None]) / (std[None, :, None] + eps)
        stats_out = {"mean": mean, "std": std}
    elif mode == "minmax":
        min_v = np.asarray(stats.get("min", arr.min(axis=axes)), dtype=float)
        max_v = np.asarray(stats.get("max", arr.max(axis=axes)), dtype=float)
        norm = (arr - min_v[None, :, None]) / (max_v[None, :, None] - min_v[None, :, None] + eps)
        stats_out = {"min": min_v, "max": max_v}
    elif mode == "none":
        norm = arr
        stats_out = stats
    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")

    return norm, stats_out


def summarize_windows(windows: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Backwards-compatible summary: average per-window features into a single vector.
    Pads variable-length windows if a list is provided.
    """
    if isinstance(windows, list):
        if not windows:
            return np.zeros(N_FEATURES, dtype=float)
        max_len = max(np.asarray(w).shape[0] for w in windows if w is not None)
        padded = []
        for w in windows:
            if w is None:
                continue
            arr = np.asarray(w, dtype=float).reshape(-1)
            padded.append(_pad_window(arr, max_len, "edge", 0.0))
        if not padded:
            return np.zeros(N_FEATURES, dtype=float)
        arr = np.stack(padded, axis=0)
    else:
        arr = np.asarray(windows, dtype=float)
    feats = compute_window_features(arr)  # (n_windows, n_channels, n_features)
    return feats.mean(axis=(0, 1))
