import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from context.sequence_model import ContextTransformer
from utils.common import ensure_dir, load_config, resolve_path

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
SEQ_LEN = 5 # history window size

class SessionDataset(Dataset):
    def __init__(self, ratings_path, embedding_path, id_map_path, seq_len=5, holdout_user=None):
        
        # raw data loading
        self.embeddings = np.load(embedding_path) 
        self.raw_ids = np.load(id_map_path)       
        self.ratings = pd.read_csv(ratings_path)
        
        print(">> [Context] Mapping ratings to audio clips...")
        if holdout_user:
             print(f">> Excluding user: {holdout_user}")
             self.ratings = self.ratings[self.ratings['participant_id'].astype(str) != str(holdout_user)]
        
        # build smart lookup table : mapping (songID, participantID) -> embedding_index  
        
        clip_groups = {}
        
        for idx, clip_name in enumerate(self.raw_ids):
            # parse ID: "{sID}_{pID}_{chunk_idx}" (example 101_hku1901_0)
            try:
                parts = clip_name.split('_')
                if len(parts) < 3: continue
                
                chunk_idx = int(parts[-1])
                pid = parts[-2]
                sid = "_".join(parts[:-2])
                
                key = (str(sid), str(pid))
                if key not in clip_groups:
                    clip_groups[key] = []
                clip_groups[key].append((chunk_idx, idx))
            except:
                continue
                
        # resolve best representative for each song
        self.lookup = {}
        for key, candidates in clip_groups.items():
            candidates.sort(key=lambda x: x[0])
            best_embedding_idx = candidates[0][1]
            self.lookup[key] = best_embedding_idx
            
        print(f" >> Mapped {len(self.lookup)} unique Song-User pairs to embeddings")

        self.sequences = []
        self.targets = []
        
        # group ratings by user to form histories
        user_groups = self.ratings.groupby('participant_id')
        
        for pid, group in user_groups:
            sorted_group = group.sort_values('song_no')
            song_ids = sorted_group['song_id'].astype(str).values
            
            # convert song IDs to embedding indices
            indices = []
            for sid in song_ids:
                key = (str(sid), str(pid))
                if key in self.lookup:
                    indices.append(self.lookup[key])
            
            if len(indices) < seq_len + 1:
                continue
                
            # sliding window logic
            for i in range(len(indices) - seq_len):
                hist_idx = indices[i : i+seq_len]
                target_idx = indices[i+seq_len]
                
                self.sequences.append(self.embeddings[hist_idx])
                self.targets.append(self.embeddings[target_idx])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32), 
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

def train_context_model(config_path="configs/config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})
    
    emb_path = resolve_path("data/processed/song_embeddings.npy")
    id_map_path = resolve_path("data/processed/song_id_map.npy")
    ratings_path = resolve_path(paths.get("ratings_csv", "data/raw/HKU956/3. AV_ratings.csv"))
    save_path = resolve_path("context/checkpoints/context_model.pth")
    
    ensure_dir(os.path.dirname(save_path))
    
    if not os.path.exists(emb_path):
        print(" !!! Audio embeddings not found")
        return

    dataset = SessionDataset(ratings_path, emb_path, id_map_path, SEQ_LEN)
    
    if len(dataset) == 0:
        print(" !!! Dataset empty")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model setup (inpu 1024 (mert) -> output 128 (context))
    model = ContextTransformer(input_dim=1024, hidden_dim=128).to(device)
    
    # predictor head ( 128 (Context) -> 1024 (predicted song embedding))
    predictor = nn.Linear(128, 1024).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=LR)
    criterion = nn.MSELoss()
    
    print(f">> [Context] Training on {len(dataset)} sequences...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # forward
            ctx_vector = model(x)
            pred_song = predictor(ctx_vector)
            
            loss = criterion(pred_song, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.5f}")
        
    # save ONLY the encoder
    torch.save(model.state_dict(), save_path)
    print(f">> [Context] model saved to {save_path}")

if __name__ == "__main__":
    train_context_model()