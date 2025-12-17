import numpy as np


def window_signal(signal: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    if signal is None:
        return []
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if window_size <= 0:
        return [signal]
    if len(signal) <= window_size:
        return [signal]
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
    if not windows:
        windows.append(signal)
    return windows


def summarize_windows(windows: list[np.ndarray]) -> np.ndarray:
    if not windows:
        return np.zeros(5, dtype=float)
    feats = []
    for w in windows:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.size == 0:
            continue
        feats.append(
            [
                float(np.mean(w)),
                float(np.std(w)),
                float(np.min(w)),
                float(np.max(w)),
                float(w[-1] - w[0]),
            ]
        )
    if not feats:
        return np.zeros(5, dtype=float)
    return np.mean(np.asarray(feats, dtype=float), axis=0)
