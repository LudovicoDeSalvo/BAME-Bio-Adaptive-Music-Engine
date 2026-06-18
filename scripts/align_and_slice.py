import os
import logging
import pandas as pd
import numpy as np
import torchaudio
import torch
from tqdm import tqdm
import warnings
from data.windows import compute_window_features
from utils.common import ensure_dir, resolve_path, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
CHUNK_SIZE_SEC = 30
STRIDE_SEC = 15
AUDIO_TRIM_START = 0
SAMPLING_RATES = {"EDA": 4, "BVP": 64, "TEMP": 4, "HR": 1, "IBI": 1}
TARGET_SR = 24000

# paths
ROOT = resolve_path("data/raw/HKU956")
OUT_AUDIO = resolve_path("data/processed/audio_clips")
OUT_PHYSIO = resolve_path("data/processed/physio_cache.npz")
OUT_PROFILE = resolve_path("data/processed/profile_cache.csv")

warnings.filterwarnings("ignore")

def align_and_process():

    print(f"--- ALIGNMENT ---")
    
    ensure_dir(OUT_AUDIO)
    
    ratings = pd.read_csv(os.path.join(ROOT, "3. AV_ratings.csv"))
    physio_dir = os.path.join(ROOT, "1. physiological_signals")
    audio_dir = os.path.join(ROOT, "2. audio_files")
    
    data_cache = {
        "window_features": [],
        "valence": [], "arousal": [],
        "song_ids": [], "clip_ids": [], "participant_ids": [], "song_nos": []
    }
    
    stats = {"processed": 0, "skipped_short": 0, "missing_files": 0, "generated": 0}
    
    for idx, row in tqdm(ratings.iterrows(), total=len(ratings), desc="Aligning"):

        pid = str(row['participant_id'])
        sid = str(row['song_id'])
        sno = str(row['song_no'])
        
        # determine duration from EDA
        eda_path = os.path.join(physio_dir, pid, "EDA", f"{sno}_{sid}.csv")
        
        if not os.path.exists(eda_path) or os.path.getsize(eda_path) == 0:
            stats["missing_files"] += 1
            continue
            
        try:
            eda_data = np.loadtxt(eda_path, delimiter=",")
            if eda_data.ndim == 0: eda_data = eda_data.reshape(1)

        except (ValueError, OSError) as e:
            logger.warning("Failed to read EDA %s: %s", eda_path, e)
            continue
            
        physio_duration = len(eda_data) / SAMPLING_RATES["EDA"]
        
        # check if too short
        if physio_duration < CHUNK_SIZE_SEC:
            stats["skipped_short"] += 1
            continue

        # load audio
        audio_path = os.path.join(audio_dir, f"{sid}.mp3")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_dir, f"{sid}.wav")
            
        if not os.path.exists(audio_path):
            continue
            
        try:
            waveform, sr = torchaudio.load(audio_path)
        except (RuntimeError, OSError) as e:
            logger.warning("Failed to load audio %s: %s", audio_path, e)
            continue
            
        # resample
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
            
        # mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # crop audio to match physio
        max_audio_samples = int(physio_duration * TARGET_SR)
        if waveform.shape[1] > max_audio_samples:
            waveform = waveform[:, :max_audio_samples]
            
        # Load each physio signal ONCE per trial (was reloaded for every chunk
        # inside the slicing loop -> O(n_chunks) re-reads of the same CSV). EDA is
        # already loaded above for the duration check, so reuse it.
        full_signals = {"EDA": eda_data}
        for sig_type in SAMPLING_RATES:
            if sig_type == "EDA":
                continue
            p = os.path.join(physio_dir, pid, sig_type, f"{sno}_{sid}.csv")
            try:
                fs = np.loadtxt(p, delimiter=",")
                if fs.ndim == 0:
                    fs = fs.reshape(1)
            except (ValueError, OSError) as e:
                logger.debug("Missing/bad %s signal %s: %s", sig_type, p, e)
                fs = np.array([])
            full_signals[sig_type] = fs

        # sliding window slicing
        num_samples = waveform.shape[1]
        chunk_samples = int(CHUNK_SIZE_SEC * TARGET_SR)
        stride_samples = int(STRIDE_SEC * TARGET_SR)

        current_start = 0
        chunk_idx = 0
        
        while current_start + chunk_samples <= num_samples:

            t_start = current_start / TARGET_SR
            t_end = t_start + CHUNK_SIZE_SEC

            clip_wave = waveform[:, current_start : current_start + chunk_samples]

            # physio processing (slice the once-loaded full signals)
            window_signals = {}

            for sig_type in SAMPLING_RATES:
                full_sig = full_signals[sig_type]

                sr_sig = SAMPLING_RATES[sig_type]
                idx_s = int(t_start * sr_sig)
                idx_e = int(t_end * sr_sig)

                if idx_s < len(full_sig):
                    window_signals[sig_type] = full_sig[idx_s : min(idx_e, len(full_sig))]
                else:
                    window_signals[sig_type] = np.array([])

            try:
                sub_windows = compute_window_features(window_signals)
            except (ValueError, RuntimeError) as e:
                logger.warning("compute_window_features failed (%s_%s chunk %d): %s",
                               sid, pid, chunk_idx, e)
                current_start += stride_samples
                continue

            # Only commit the clip (audio + cache + chunk_idx) once the physio
            # chunk is accepted, so clip_id <-> audio file <-> chunk_idx never drift.
            if not sub_windows:
                current_start += stride_samples
                continue

            # Unique ID: Song_User_Trial_Chunk. song_no (sno) is REQUIRED: a
            # participant may replay the same song_id under a different song_no,
            # and chunk_idx resets per trial. Without sno the clip_ids of the two
            # plays collide -> the second .wav overwrites the first and a single
            # MERT embedding is shared across two distinct physio trials.
            clip_id = f"{sid}_{pid}_{sno}_{chunk_idx}"
            out_name = os.path.join(OUT_AUDIO, f"{clip_id}.wav")
            torchaudio.save(out_name, clip_wave, TARGET_SR)

            stack = np.stack(sub_windows)

            data_cache["window_features"].append(stack)
            data_cache["valence"].append(row["valence_rating"])
            data_cache["arousal"].append(row["arousal_rating"])
            # Store the RAW song_id string. The old int(sid)-or-0 coercion
            # collapsed every non-numeric id to 0 (collisions) and disagreed with
            # the raw str(song_id) the ratings-keyed lookups use downstream.
            data_cache["song_ids"].append(sid)
            data_cache["clip_ids"].append(clip_id)
            data_cache["participant_ids"].append(pid)
            # song_no is the per-playback trial id: a participant may hear the
            # same song_id more than once (distinct session position + physio),
            # so (participant_id, song_no) — not (participant_id, song_id) — is
            # the unique trial key downstream uses to avoid cross-trial mixing.
            data_cache["song_nos"].append(sno)

            stats["generated"] += 1
            chunk_idx += 1

            current_start += stride_samples
            
        stats["processed"] += 1

    # save
    print(f"\n--- ALIGNMENT COMPLETE ---")
    print(f"Processed trials: {stats['processed']}")
    print(f"Generated matched clips: {stats['generated']}")
    print(f"Skipped (too short): {stats['skipped_short']}")
    
    if stats["generated"] == 0:
        print("!!! Error: no data generated")
        return
        
    # pack
    w_list = data_cache["window_features"]
    max_len = max(w.shape[0] for w in w_list)
    feat_dim = w_list[0].shape[1]
    dense_windows = np.zeros((len(w_list), max_len, feat_dim))
    for i, w in enumerate(w_list):
        dense_windows[i, :w.shape[0], :] = w

    # Per-channel z-score stats for the RAW per-window features the encoder
    # actually consumes (window_features). EDA/BVP/TEMP/HR/IBI live on wildly
    # different scales, so without this the encoder is dominated by the
    # large-magnitude channels. Stats are computed over real (non-padding)
    # windows only — padding rows are all-zero and excluded.
    #
    # NOTE: these cached stats are now only a LEGACY FALLBACK. The encoder refits
    # its own normalizer on the holdout-excluded split (true LOSO refits per fold)
    # and persists it next to the embeddings. We still exclude the config holdout
    # here so the fallback is not leaky either.
    holdout = load_config().get("training", {}).get("holdout_user", None)
    pid_arr = np.array([str(p) for p in data_cache["participant_ids"]])
    clip_mask = (pid_arr != str(holdout)) if holdout else np.ones(len(pid_arr), dtype=bool)
    stat_windows = dense_windows[clip_mask] if np.any(clip_mask) else dense_windows

    flat = stat_windows.reshape(-1, feat_dim)
    valid = ~np.all(flat == 0, axis=1)
    valid_rows = flat[valid] if np.any(valid) else flat
    win_mean = valid_rows.mean(axis=0)
    win_std = valid_rows.std(axis=0)
    win_std[win_std == 0] = 1.0

    np.savez(OUT_PHYSIO,
        window_features=dense_windows,
        valence=np.array(data_cache["valence"]),
        arousal=np.array(data_cache["arousal"]),
        song_ids=np.array(data_cache["song_ids"]),
        clip_ids=np.array(data_cache["clip_ids"]),
        participant_ids=np.array(data_cache["participant_ids"]),
        song_nos=np.array(data_cache["song_nos"]),
        win_norm_mean=win_mean,
        win_norm_std=win_std
    )
    
    try:
        pd.read_csv(os.path.join(ROOT, "4. participant_personality.csv")).to_csv(OUT_PROFILE, index=False)
    except (FileNotFoundError, OSError) as e:
        logger.warning("Could not copy personality CSV: %s", e)
    
    print(f" Saved aligned dataset to {OUT_PHYSIO}")
    print(f" Saved audio clips to {OUT_AUDIO}")

if __name__ == "__main__":
    align_and_process()