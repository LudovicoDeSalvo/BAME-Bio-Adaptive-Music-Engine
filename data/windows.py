import numpy as np
from scipy import stats

# --- Configuration ---
WINDOW_SIZE = 10  # Seconds per window 
STEP_SIZE = 10    
SAMPLING_RATES = {
    "EDA": 4,
    "BVP": 64,
    "TEMP": 4,
    "HR": 1,
    "IBI": 1 
}

N_STATS = 6 # mean, std, min, max, dynamic range, slope
N_SIGNALS = 5 # EDA, BVP, TEMP, HR, IBI
N_FEATURES = N_SIGNALS * N_STATS

def get_slope(y):

    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))

    try:
        slope = np.polyfit(x, y, 1)[0]
        return slope
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return 0.0

def extract_stats(signal):
    """calculates statistical features for 1D array"""
    if len(signal) == 0:
        return np.zeros(N_STATS)
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    dynamic_range = max_val - min_val
    slope = get_slope(signal)
    
    return np.array([mean_val, std_val, min_val, max_val, dynamic_range, slope])

def compute_window_features(signals_map: dict) -> list:
    """  
    args:
        signals_map: dict {'EDA': np.array, 'BVP': np.array, ...}
    
    returns:
        list of np.array for each window
    """
    # determine duration in seconds based on EDA or whatever is longest
    max_duration = 0
    for name, data in signals_map.items():
        if name in SAMPLING_RATES and len(data) > 0:
            duration = len(data) / SAMPLING_RATES[name]
            max_duration = max(max_duration, duration)
    
    if max_duration == 0:
        return []

    # iterate windows
    windows = []
    n_windows = int(max_duration // STEP_SIZE)
    
    for i in range(n_windows):
        start_sec = i * STEP_SIZE
        end_sec = start_sec + WINDOW_SIZE
        
        window_feats = []
        
        target_order = ["EDA", "BVP", "TEMP", "HR", "IBI"]
        
        for sig_name in target_order:
            if sig_name in signals_map and len(signals_map[sig_name]) > 0:
                sr = SAMPLING_RATES.get(sig_name, 1)
                start_idx = int(start_sec * sr)
                end_idx = int(end_sec * sr)
                
                sig_data = signals_map[sig_name]
                expected_len = int(WINDOW_SIZE * sr)
                if start_idx < len(sig_data):
                    valid_end = min(end_idx, len(sig_data))
                    chunk = sig_data[start_idx:valid_end]
                    # Drop truncated tail windows: stats on a partial window
                    # (< WINDOW_SIZE*sr samples) skew the feature vector.
                    if len(chunk) < expected_len:
                        chunk = np.array([])
                else:
                    chunk = np.array([])

                stats_vec = extract_stats(chunk)
            else:
                # missing signal -> zeros
                stats_vec = np.zeros(N_STATS)
            
            window_feats.append(stats_vec)
            
        # concatenation
        windows.append(np.concatenate(window_feats))
        
    return windows

def summarize_windows(windows: list) -> np.ndarray:
    """averages all window feature vectors into a single global vector"""
    
    if not windows:
        return np.zeros(N_FEATURES)

    arr = np.stack(windows)

    return np.mean(arr, axis=0)

def apply_window_norm(windows: np.ndarray, mean, std) -> np.ndarray:
    """Per-channel z-score a [N, T, F] (or [T, F]) window tensor.

    Padding timesteps are all-zero (see align_and_slice densification) and are
    preserved as zero AFTER normalization so the encoder's LSTM still sees a
    clean zero pad instead of a constant -mean/std offset. A genuine real window
    that is exactly zero across all F channels is not physically possible, so
    treating all-zero rows as padding is safe.
    """
    w = np.asarray(windows, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.where(std == 0, 1.0, std)

    pad = np.all(w == 0, axis=-1, keepdims=True)
    out = (w - mean) / std
    out = np.where(pad, 0.0, out)
    return out.astype(np.float32)


def normalize_windows(features: np.ndarray, mode: str = "zscore"):
    """
    input: [N_samples, N_features]
    """
    features = np.asarray(features, dtype=float)

    # safety check — return array-typed stats (consistent with the normal branch)
    # so callers never have to special-case scalar vs vector mean/std.
    if features.size == 0 or features.ndim < 2:
        w = features.shape[-1] if features.ndim >= 1 and features.shape[-1] > 0 else 1
        return features, {"mean": np.zeros(w, dtype=float), "std": np.ones(w, dtype=float)}

    if mode == "zscore":
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)

        std[std == 0] = 1.0 
        
        normed = (features - mean) / std
        return normed, {"mean": mean, "std": std}
    
    return features, {}