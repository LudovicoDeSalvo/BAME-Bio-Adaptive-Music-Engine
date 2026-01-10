import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

from simulator.world_model import WorldModel
from context.sequence_model import ContextTransformer
from utils.common import ensure_dir, load_config, resolve_path

class TransitionDataset(Dataset):
    def __init__(self, physio_path, audio_emb_path, audio_id_path, user_pool_path, ratings_path, context_model_path, device, holdout_user=None):
        print(">> [Simulator] Loading datasets...")
        
        # data loading
        p_data = np.load(physio_path, allow_pickle=True)
        audio_embs = np.load(audio_emb_path, allow_pickle=True)
        audio_ids = np.load(audio_id_path, allow_pickle=True)
        u_data = np.load(user_pool_path, allow_pickle=True)
        ratings = pd.read_csv(ratings_path)

        if holdout_user:
            print(f">> [Dataset] Holding out user: {holdout_user}")
            ratings = ratings[ratings['participant_id'].astype(str) != str(holdout_user)]
        
        # audio
        self.audio_map = {str(k): v for k, v in zip(audio_ids, audio_embs)}
        
        # user
        self.user_map = {str(k): v for k, v in zip(u_data['participant_ids'], u_data['embeddings'])}
        
        # physio
        physio_lookup = defaultdict(list)
        p_clip_ids = p_data['clip_ids']
        

        p_features = p_data['embeddings'] if 'embeddings' in p_data else p_data['features']
        p_song_ids = p_data['song_ids']
        p_participants = p_data['participant_ids']
        
        for idx, (cid, pid) in enumerate(zip(p_clip_ids, p_participants)):
            try:
                sid = str(p_song_ids[idx])
                parts = str(cid).split('_')
                chunk_idx = int(parts[-1])
                
                physio_lookup[(str(pid), sid)].append({
                    'chunk_idx': chunk_idx,
                    'physio': p_features[idx],
                    'clip_id': str(cid)
                })
            except: continue

        # context
        self.ctx_model = ContextTransformer(input_dim=1024, hidden_dim=128).to(device)
        self.ctx_model.load_state_dict(torch.load(context_model_path, map_location=device))
        self.ctx_model.eval()
        self.device = device
        
        self.transitions = []
        
        user_groups = ratings.groupby('participant_id')
        
        for pid, group in tqdm(user_groups):
            pid = str(pid)
            if pid not in self.user_map: continue
            
            u_vec = self.user_map[pid]
            sorted_songs = group.sort_values('song_no')
            history_buffer = [] 
            
            for _, row in sorted_songs.iterrows():
                sid = str(row['song_id'])
                
                if (pid, sid) not in physio_lookup:
                    continue

                clips = physio_lookup[(pid, sid)]
                clips.sort(key=lambda x: x['chunk_idx'])
                
                for i in range(len(clips) - 1):
                    curr = clips[i]
                    nxt = clips[i+1]
                    
                    if nxt['chunk_idx'] != curr['chunk_idx'] + 1: continue
                    
                    # causality check
                    action_id = curr['clip_id'] 
                    if action_id not in self.audio_map: continue
                    action_vec = self.audio_map[action_id]
                    
                    with torch.no_grad():
                        if len(history_buffer) == 0:
                            context_vec = np.zeros(128, dtype=np.float32)
                        else:
                            seq = np.array(history_buffer[-5:]) 
                            inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                            c_out = self.ctx_model(inp)
                            context_vec = c_out.cpu().numpy()[0]

                    # construct state
                    state_t = np.concatenate([curr['physio'], u_vec, context_vec])
                    target_physio = nxt['physio']
                    
                    self.transitions.append((state_t, action_vec, target_physio))
                    
                    history_buffer.append(action_vec)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s, a, np_ = self.transitions[idx]
        return (
            torch.tensor(s, dtype=torch.float32), 
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(np_, dtype=torch.float32)
        )

def train_world_model(epochs=70, config_path="configs/config.yaml"):
    config = load_config(config_path)
    paths = config.get("paths", {})
    
    # load embeddings
    physio_path = resolve_path(paths.get("physio_embeddings", "data/processed/physio_embeddings.npz"))
    audio_emb = resolve_path("data/processed/song_embeddings.npy")
    audio_ids = resolve_path("data/processed/song_id_map.npy")
    user_pool = resolve_path(paths.get("user_embeddings"))
    ratings = resolve_path(paths.get("ratings_csv"))
    ctx_model_path = resolve_path("context/checkpoints/context_model.pth")
    
    save_path = resolve_path("simulator/checkpoints/world_model.pth")
    ensure_dir(os.path.dirname(save_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(ctx_model_path):
        print(" !!! Error: context model not found")
        return

    try:
        # check for holdout user config
        holdout = config.get('training', {}).get('holdout_user', None)
        
        dataset = TransitionDataset(physio_path, audio_emb, audio_ids, user_pool, ratings, ctx_model_path, device, holdout)
    except Exception as e:
        print(f" !!! Dataset Error: {e}")
        return
    
    if len(dataset) == 0:
        print(" !!! No transitions created. Check data alignment")
        return

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = WorldModel(state_dim=224, action_dim=1024, physio_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f">> [Simulator] Training World Model on {len(dataset)} transitions...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for state, action, target in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            state, action, target = state.to(device), action.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(state, action)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.5f}")
            
    model.save(save_path)
    print(f">> [Simulator] Model saved to {save_path}")

if __name__ == "__main__":
    train_world_model()