import numpy as np

from data.windows import (
    extract_stats, compute_window_features, summarize_windows,
    N_STATS, N_FEATURES, WINDOW_SIZE, SAMPLING_RATES,
)


def test_extract_stats_empty_returns_zeros():
    out = extract_stats(np.array([]))
    assert out.shape == (N_STATS,)
    assert np.all(out == 0)


def test_extract_stats_values():
    out = extract_stats(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (N_STATS,)
    assert np.isclose(out[0], 2.0)          # mean
    assert np.isclose(out[2], 1.0)          # min
    assert np.isclose(out[3], 3.0)          # max
    assert np.isclose(out[4], 2.0)          # dynamic range


def test_full_window_features_shape():
    # one full 10s window of each signal at its sampling rate
    signals = {name: np.ones(WINDOW_SIZE * sr) for name, sr in SAMPLING_RATES.items()}
    wins = compute_window_features(signals)
    assert len(wins) == 1
    assert wins[0].shape == (N_FEATURES,)


def test_partial_window_is_zeroed():
    # Signals of different durations: EDA drives max_duration (25s -> 2 windows),
    # but HR only has 12s of data. The second window [10s,20s] for HR is a
    # truncated partial window and must be zeroed, not given skewed stats.
    eda_sr = SAMPLING_RATES["EDA"]
    hr_sr = SAMPLING_RATES["HR"]
    signals = {
        "EDA": np.ones(int(25 * eda_sr)),
        "HR": np.ones(int(12 * hr_sr)),
    }
    wins = compute_window_features(signals)
    assert len(wins) == 2  # 25 // STEP_SIZE

    order = ["EDA", "BVP", "TEMP", "HR", "IBI"]
    hr_off = order.index("HR") * N_STATS
    # window 0 [0,10]: HR full -> non-zero mean
    assert wins[0][hr_off] != 0
    # window 1 [10,20]: HR only has 2s of data -> truncated -> zeroed slice
    assert np.all(wins[1][hr_off:hr_off + N_STATS] == 0)


def test_summarize_empty():
    assert summarize_windows([]).shape == (N_FEATURES,)
