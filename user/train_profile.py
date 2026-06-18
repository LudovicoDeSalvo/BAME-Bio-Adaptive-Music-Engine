import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from user.dcn_profile import DCNProfile
from utils.common import ensure_dir, load_config, resolve_path, get_device, set_seed

# --- Configuration ---
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.005

class UserDataset(Dataset):
    def __init__(self, csv_path, holdout_user=None):

        df = pd.read_csv(csv_path)
        
        if holdout_user:
            print(f">> [Dataset] Excluding user: {holdout_user}")
            df = df[df['participant_id'].astype(str) != str(holdout_user)]
        
        self.feat_cols = [c for c in df.columns if c.endswith('_score')]
        
        if not self.feat_cols:
            print(" !!! Warning: No '_score' columns found. Trying hardcoded fallback")
            self.feat_cols = [
                'Extroversion', 'Agreeableness', 'Conscientiousness', 
                'Emotional_Stability', 'Openness'
            ]
            
        # normalization
        raw_vals = df[self.feat_cols].values.astype(np.float32)
        
        self.min_vals = raw_vals.min(axis=0)
        self.max_vals = raw_vals.max(axis=0) + 1e-6
        
        self.features = (raw_vals - self.min_vals) / (self.max_vals - self.min_vals)
        self.features = torch.tensor(self.features)
        
        self.ids = df['participant_id'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def train_user_model(config_path="configs/config.yaml", holdout_user=None):

    config = load_config(config_path)
    paths = config.get("paths", {})
    set_seed(int(config.get("training", {}).get("seed", 42)))

    csv_path = resolve_path(paths.get("personality_csv", "data/raw/HKU956/4. participant_personality.csv"))
    model_path = resolve_path("user/checkpoints/profile_model.pth")
    pool_path = resolve_path("data/processed/user_embeddings.npz")

    ensure_dir(os.path.dirname(model_path))
    ensure_dir(os.path.dirname(pool_path))

    print(f">> [User] Loading profiles from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    # Exclude the holdout user from the profiler fit + pool so the LOSO
    # evaluation user never leaks into training (the min/max scaler included).
    # An explicit holdout (per-fold LOSO) overrides the config default.
    holdout = holdout_user if holdout_user is not None else config.get("training", {}).get("holdout_user", None)
    dataset = UserDataset(csv_path, holdout_user=holdout)

    device = get_device()
    seed = int(config.get("training", {}).get("seed", 42))

    # Deterministic 85/15 train/val split for best-checkpoint selection (H2). The
    # pool is still generated over the FULL holdout-excluded dataset afterwards.
    n_total = len(dataset)
    n_val = max(1, int(round(0.15 * n_total))) if n_total > 1 else 0
    if n_val > 0:
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_total - n_val, n_val], generator=gen)
    else:
        train_set, val_set = dataset, None
    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) if val_set is not None else None

    emb_dim = config.get("model", {}).get("profile_embedding_dim", 32)
    input_dim = len(dataset.feat_cols)

    model = DCNProfile(input_dim=input_dim, embedding_dim=emb_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f">> [User] Training DCN on {n_total - n_val} samples (val {n_val})...")

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            emb, recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = None
        if val_loader is not None:
            model.eval()
            v = 0.0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    _, recon = model(x)
                    v += criterion(recon, x).item()
            val_loss = v / max(1, len(val_loader))
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            msg = f"   Epoch {epoch+1}/{EPOCHS} | train MSE: {total_loss / max(1, len(dataloader)):.5f}"
            if val_loss is not None:
                msg += f" | val MSE: {val_loss:.5f} (best {best_val:.5f})"
            print(msg)

    model.load_state_dict(best_state)
    model.save(model_path)
    print(f">> [User] Model saved to {model_path}")

    generate_pool(model, dataset, pool_path, device)

def generate_pool(model, dataset, path, device):

    print(f">> [User] Generating user embeddings pool...")
    model.eval()
    
    full_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_embs = []
    
    with torch.no_grad():
        for x in full_loader:
            x = x.to(device)
            emb, _ = model(x)
            all_embs.append(emb.cpu().numpy())
            
    final_pool = np.vstack(all_embs)

    # Persist the fitted min-max scaler so inference uses the SAME normalizer
    # (single source of truth) instead of the old hardcoded /10.0.
    np.savez(path,
             embeddings=final_pool,
             participant_ids=dataset.ids,
             norm_min=dataset.min_vals,
             norm_max=dataset.max_vals,
             feat_cols=np.array(dataset.feat_cols))

    print(f">> [User] Saved {final_pool.shape} embeddings + scaler to {path}")

if __name__ == "__main__":
    train_user_model()