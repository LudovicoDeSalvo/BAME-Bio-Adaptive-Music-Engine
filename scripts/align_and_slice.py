import os
import pandas as pd
import numpy as np
import torchaudio
import torch
from tqdm import tqdm
import warnings
from data.windows import compute_window_features, summarize_windows, normalize_windows
from utils.common import ensure_dir

# --- Configuration ---
CHUNK_SIZE_SEC = 30
STRIDE_SEC = 15
AUDIO_TRIM_START = 0
SAMPLING_RATES = {"EDA": 4, "BVP": 64, "TEMP": 4, "HR": 1, "IBI": 1}
TARGET_SR = 24000

# paths
ROOT = "data/raw/HKU956"
OUT_AUDIO = "data/processed/audio_clips"
OUT_PHYSIO = "data/processed/physio_cache.npz"
OUT_PROFILE = "data/processed/profile_cache.csv"

warnings.filterwarnings("ignore")

def align_and_process():

    print(f"--- ALIGNMENT ---")
    
    ensure_dir(OUT_AUDIO)
    
    ratings = pd.read_csv(os.path.join(ROOT, "3. AV_ratings.csv"))
    physio_dir = os.path.join(ROOT, "1. physiological_signals")
    audio_dir = os.path.join(ROOT, "2. audio_signals")
    
    data_cache = {
        "features": [], "window_features": [], 
        "valence": [], "arousal": [], 
        "song_ids": [], "clip_ids": [], "participant_ids": []
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

        except:
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
        except:
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
            
        # sliding window slicing      
        num_samples = waveform.shape[1]
        chunk_samples = int(CHUNK_SIZE_SEC * TARGET_SR)
        stride_samples = int(STRIDE_SEC * TARGET_SR)
        
        current_start = 0
        chunk_idx = 0
        
        while current_start + chunk_samples <= num_samples:

            t_start = current_start / TARGET_SR
            t_end = t_start + CHUNK_SIZE_SEC
            
            # save clip
            clip_wave = waveform[:, current_start : current_start + chunk_samples]
            clip_id = f"{sid}_{pid}_{chunk_idx}" # Unique ID: Song_User_Chunk
            
            # save raw audio
            out_name = os.path.join(OUT_AUDIO, f"{clip_id}.wav")
            torchaudio.save(out_name, clip_wave, TARGET_SR)
            
            # physio processing
            window_signals = {}
            valid_physio = True
            
            for sig_type in SAMPLING_RATES:

                p = os.path.join(physio_dir, pid, sig_type, f"{sno}_{sid}.csv")

                try:
                    full_sig = np.loadtxt(p, delimiter=",")
                    if full_sig.ndim == 0: full_sig = full_sig.reshape(1)
                except:
                    full_sig = np.array([])
                
                sr_sig = SAMPLING_RATES[sig_type]
                idx_s = int(t_start * sr_sig)
                idx_e = int(t_end * sr_sig)
                
                if idx_s < len(full_sig):
                    window_signals[sig_type] = full_sig[idx_s : min(idx_e, len(full_sig))]
                else:
                    window_signals[sig_type] = np.array([])
            
            try:
                sub_windows = compute_window_features(window_signals)
                if not sub_windows: 
                    current_start += stride_samples
                    continue
                    
                summary = summarize_windows(sub_windows)
                stack = np.stack(sub_windows)
                
                data_cache["features"].append(summary)
                data_cache["window_features"].append(stack)
                data_cache["valence"].append(row["valence_rating"])
                data_cache["arousal"].append(row["arousal_rating"])
                data_cache["song_ids"].append(int(sid) if sid.isdigit() else 0)
                data_cache["clip_ids"].append(clip_id) # Using the unique ID
                data_cache["participant_ids"].append(pid)
                
                stats["generated"] += 1
                chunk_idx += 1
                
            except Exception:
                pass
            
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
        
    # normalize
    feats_arr = np.array(data_cache["features"])
    norm_feats, norm_stats = normalize_windows(feats_arr, mode="zscore")
    
    np.savez(OUT_PHYSIO,
        features=norm_feats,
        window_features=dense_windows,
        valence=np.array(data_cache["valence"]),
        arousal=np.array(data_cache["arousal"]),
        song_ids=np.array(data_cache["song_ids"]),
        clip_ids=np.array(data_cache["clip_ids"]),
        participant_ids=np.array(data_cache["participant_ids"]),
        norm_mean=norm_stats["mean"],
        norm_std=norm_stats["std"]
    )
    
    try:
        pd.read_csv(os.path.join(ROOT, "4. participant_personality.csv")).to_csv(OUT_PROFILE, index=False)
    except: pass
    
    print(f" Saved aligned dataset to {OUT_PHYSIO}")
    print(f" Saved audio clips to {OUT_AUDIO}")

if __name__ == "__main__":
    align_and_process()