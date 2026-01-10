import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from user.dcn_profile import DCNProfile
from utils.common import ensure_dir, load_config, resolve_path

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

def train_user_model(config_path="configs/config.yaml"):

    config = load_config(config_path)
    paths = config.get("paths", {})
    
    csv_path = resolve_path(paths.get("personality_csv", "data/raw/HKU956/4. participant_personality.csv"))
    model_path = resolve_path("user/checkpoints/profile_model.pth")
    pool_path = resolve_path("data/processed/user_embeddings.npz")
    
    ensure_dir(os.path.dirname(model_path))
    ensure_dir(os.path.dirname(pool_path))
    
    print(f">> [User] Loading profiles from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
        
    dataset = UserDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    emb_dim = config.get("model", {}).get("profile_embedding_dim", 32)
    input_dim = len(dataset.feat_cols)
    
    model = DCNProfile(input_dim=input_dim, embedding_dim=emb_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f">> [User] Training DCN...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x in dataloader:

            x = x.to(device)
            optimizer.zero_grad()
            emb, recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | MSE Loss: {total_loss / len(dataloader):.5f}")

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
    
    np.savez(path, 
             embeddings=final_pool, 
             participant_ids=dataset.ids)
             
    print(f">> [User] Saved {final_pool.shape} embeddings to {path}")

if __name__ == "__main__":
    train_user_model()