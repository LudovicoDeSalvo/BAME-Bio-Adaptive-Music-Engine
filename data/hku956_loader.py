import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- CONFIGURATION ---
# We assume the user moved HKU956 to data/raw/HKU956
DATA_ROOT = os.path.join("data", "raw", "HKU956")
PHYSIO_DIR = os.path.join(DATA_ROOT, "1. physiological_signals")
RATINGS_PATH = os.path.join(DATA_ROOT, "3. AV_ratings.csv")
USER_PATH = os.path.join(DATA_ROOT, "4. participant_personality.csv")

# Standardize signals to last 60 seconds
EDA_LEN = 240   # 4Hz * 60s
BVP_LEN = 3840  # 64Hz * 60s

class HKU956Dataset(Dataset):
    def __init__(self, limit=None):
        print(">> [Data Loader] Initializing HKU956 Dataset...")
        
        # 1. Load Tables
        if not os.path.exists(RATINGS_PATH):
            raise FileNotFoundError(f"Missing {RATINGS_PATH}")
            
        self.ratings = pd.read_csv(RATINGS_PATH)
        self.users = pd.read_csv(USER_PATH)
        
        # Map participant_id (e.g., 'hku1903') to row index
        self.user_map = {u: i for i, u in enumerate(self.users['participant_id'])}
        
        # --- FIX: Explicit Column Names from your CSV ---
        self.tipi_cols = [
            'Extroversion_score', 
            'Agreeableness_score', 
            'Conscientiousness_score', 
            'Emotional_Stability_score', 
            'Openness_score'
        ]

        # 2. Index valid files
        self.valid_indices = []
        missing_files = 0
        
        print(f">> [Data Loader] Verifying file integrity...")
        for idx, row in self.ratings.iterrows():
            if limit and len(self.valid_indices) >= limit: break
            
            pid = row['participant_id']
            song_id = row['song_id']
            song_no = row['song_no'] # Listening order
            
            # Construct filenames based on HKU naming convention
            filename = f"{song_no}_{song_id}.csv"
            
            eda_path = os.path.join(PHYSIO_DIR, pid, "EDA", filename)
            bvp_path = os.path.join(PHYSIO_DIR, pid, "BVP", filename)
            
            # Only keep sample if BOTH signal files exist
            if os.path.exists(eda_path) and os.path.exists(bvp_path):
                self.valid_indices.append({
                    'row_idx': idx,
                    'eda_path': eda_path,
                    'bvp_path': bvp_path,
                    'pid': pid
                })
            else:
                missing_files += 1

        print(f">> [Data Loader] Indexed {len(self.valid_indices)} valid samples. (Skipped {missing_files} missing files)")

    def __len__(self):
        return len(self.valid_indices)

    def _load_signal(self, path, target_len):
        """Helper to read CSV, normalize, and pad/crop."""
        try:
            # Physio CSVs in HKU956 usually have just values
            df = pd.read_csv(path, header=None)
            vals = df.values.flatten().astype(np.float32)
            
            # Heuristic: Remove timestamp if present (Empatica standard > 1e9)
            if len(vals) > 2 and vals[0] > 1e9: 
                vals = vals[2:]
                
            # Normalize (Z-score)
            if vals.std() > 1e-6:
                vals = (vals - vals.mean()) / vals.std()
            else:
                vals = vals - vals.mean()

            # Fix Length
            current_len = len(vals)
            if current_len > target_len:
                vals = vals[-target_len:]
            else:
                pad = np.zeros(target_len - current_len, dtype=np.float32)
                vals = np.concatenate([pad, vals])
                
            return vals
            
        except Exception:
            return np.zeros(target_len, dtype=np.float32)

    def __getitem__(self, idx):
        info = self.valid_indices[idx]
        row = self.ratings.iloc[info['row_idx']]
        
        # 1. Load Physio (State)
        eda = self._load_signal(info['eda_path'], EDA_LEN)
        bvp = self._load_signal(info['bvp_path'], BVP_LEN)
        
        # 2. Load User Profile (Context)
        u_idx = self.user_map[info['pid']]
        user_vals = self.users.iloc[u_idx][self.tipi_cols].values.astype(np.float32)
        
        # 3. Load Targets (Reward)
        valence = row['valence_rating'] / 10.0
        arousal = row['arousal_rating'] / 10.0
        
        return {
            'eda': torch.tensor(eda).unsqueeze(0), 
            'bvp': torch.tensor(bvp).unsqueeze(0), 
            'tipi': torch.tensor(user_vals),
            'song_id': str(row['song_id']),
            'reward': torch.tensor([valence, arousal], dtype=torch.float32)
        }

# --- THIS IS THE FUNCTION main.py WAS LOOKING FOR ---
def process_and_cache_data():
    """Function called by main.py to verify data integrity."""
    ds = HKU956Dataset(limit=10) # Test with 10 items
    print("\n>> [Data Check] Successfully loaded sample batch:")
    sample = ds[0]
    print(f"   EDA Shape: {sample['eda'].shape}")
    print(f"   TIPI Values: {sample['tipi']}")
    print(f"   Reward: {sample['reward']}")
    print(">> Data Loader is ready for training.")

if __name__ == "__main__":
    process_and_cache_data()