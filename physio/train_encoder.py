import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from physio.encoder import DualStreamEncoder
from utils.common import ensure_dir, load_config, resolve_path

class PhysioDataset(Dataset):
    def __init__(self, cache_path, holdout_user=None):
        data = np.load(cache_path, allow_pickle=True)
        
        # data loading
        all_feats = torch.tensor(data['window_features'], dtype=torch.float32)
        v = torch.tensor(data['valence'], dtype=torch.float32)
        a = torch.tensor(data['arousal'], dtype=torch.float32)
        all_targets = torch.stack([v, a], dim=1)
        all_pids = data['participant_ids']
        all_clip_ids = data['clip_ids'] 
        all_song_ids = data['song_ids']

        # filtering
        if holdout_user:
            print(f">> [Dataset] Excluding user: {holdout_user}")
            mask = [str(pid) != str(holdout_user) for pid in all_pids]
            mask = np.array(mask)
        else:
            mask = np.ones(len(all_pids), dtype=bool)

        # mask
        self.windows = all_feats[mask]
        self.targets = all_targets[mask]
        self.participants = all_pids[mask]
        self.ids = all_song_ids[mask]
        self.clip_ids = all_clip_ids[mask]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # stream 1
        eda = self.windows[idx, :, 0:6]
        temp = self.windows[idx, :, 12:18]
        dermal_stream = torch.cat([eda, temp], dim=-1) 
        
        # stream 2
        bvp = self.windows[idx, :, 6:12]
        hr = self.windows[idx, :, 18:24]
        ibi = self.windows[idx, :, 24:30]
        cardio_stream = torch.cat([bvp, hr, ibi], dim=-1)
        
        return dermal_stream, cardio_stream, self.targets[idx]

def train_physio_model(epochs=20, config_path="configs/config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})
    
    cache_path = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    model_path = resolve_path("physio/checkpoints/physio_encoder.pth")
    embeddings_out = resolve_path("data/processed/physio_embeddings.npz")
    
    ensure_dir(os.path.dirname(model_path))
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing {cache_path}. Run Option [1] first.")
        
    holdout_user=config['training'].get('holdout_user')
    dataset = PhysioDataset(cache_path, holdout_user)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DualStreamEncoder(embedding_dim=64).to(device)
    head = nn.Linear(64, 2).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f">> [Physio] Training on {len(dataset)} samples...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for d_in, c_in, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            d_in, c_in, target = d_in.to(device), c_in.to(device), target.to(device)
            
            optimizer.zero_grad()
            emb = model(d_in, c_in)
            pred = head(emb)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

    model.save(model_path)
    print(f">> [Physio] model saved!")
    
    generate_embeddings(model, dataset, embeddings_out, device)

def generate_embeddings(model, dataset, path, device):
    print(f">> [Physio] generating embeddings...")
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embs = []
    
    with torch.no_grad():
        for d_in, c_in, _ in loader:
            d_in, c_in = d_in.to(device), c_in.to(device)
            emb = model(d_in, c_in)
            all_embs.append(emb.cpu().numpy())
            
    final_pool = np.vstack(all_embs)
    
    # save clip_ids
    np.savez(path, 
             embeddings=final_pool, 
             song_ids=dataset.ids, 
             participant_ids=dataset.participants,
             clip_ids=dataset.clip_ids)
             
    print(f">> [Physio] saved {final_pool.shape} embeddings to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    train_physio_model(epochs=args.epochs)