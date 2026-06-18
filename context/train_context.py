import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from context.sequence_model import ContextTransformer
from utils.common import ensure_dir, load_config, resolve_path, set_seed, parse_clip_id

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
            # parse ID: "{sID}_{pID}_{song_no}_{chunk_idx}" (e.g. 101_hku1901_5_0)
            try:
                sid, pid, sno, chunk_idx = parse_clip_id(clip_name)

                # Key by the per-playback trial (sid, pid, song_no) — NOT (sid,
                # pid) — so replays of the same song stay separate, matching the
                # trial key the simulator and inference use.
                key = (str(sid), str(pid), str(sno))
                if key not in clip_groups:
                    clip_groups[key] = []
                clip_groups[key].append((chunk_idx, idx))
            except (ValueError, IndexError):
                continue

        print(f" >> Mapped {len(clip_groups)} unique Song-User-Trial groups to embeddings")

        self.sequences = []
        self.targets = []

        # group ratings by user to form histories
        user_groups = self.ratings.groupby('participant_id')

        for pid, group in user_groups:
            sorted_group = group.sort_values('song_no')

            # Build a CHUNK-level sequence in (song_no, chunk_idx) order. The
            # runtime context model consumes a stream of per-chunk MERT vectors
            # (train_simulator history / gym applied actions are per chunk), so we
            # train on the SAME granularity instead of one representative per song
            # — otherwise the context model's train and runtime inputs disagree.
            # Keyed by (sid, pid, song_no) so each replay contributes its own
            # ordered chunk run instead of being merged with the other play.
            indices = []
            for _, r in sorted_group.iterrows():
                key = (str(r['song_id']), str(pid), str(r['song_no']))
                if key in clip_groups:
                    for _, emb_idx in sorted(clip_groups[key], key=lambda x: x[0]):
                        indices.append(emb_idx)

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

def train_context_model(config_path="configs/config.yaml", holdout_user=None):
    config = load_config(config_path)
    paths = config.get("paths", {})
    set_seed(int(config.get("training", {}).get("seed", 42)))

    emb_path = resolve_path("data/processed/song_embeddings.npy")
    id_map_path = resolve_path("data/processed/song_id_map.npy")
    ratings_path = resolve_path(paths.get("ratings_csv", "data/raw/HKU956/3. AV_ratings.csv"))
    save_path = resolve_path("context/checkpoints/context_model.pth")
    
    ensure_dir(os.path.dirname(save_path))
    
    if not os.path.exists(emb_path):
        print(" !!! Audio embeddings not found")
        return

    # Exclude the holdout user so the LOSO eval subject never leaks into the
    # context model (physio/user/simulator already exclude it). An explicit
    # holdout (per-fold LOSO) overrides the config default.
    holdout = holdout_user if holdout_user is not None else config.get("training", {}).get("holdout_user", None)
    dataset = SessionDataset(ratings_path, emb_path, id_map_path,
                             seq_len=SEQ_LEN, holdout_user=holdout)
    
    if len(dataset) == 0:
        print(" !!! Dataset empty")
        return

    # Deterministic 85/15 train/val split for best-checkpoint selection (H2).
    seed = int(config.get("training", {}).get("seed", 42))
    n_total = len(dataset)
    n_val = max(1, int(round(0.15 * n_total))) if n_total > 1 else 0
    if n_val > 0:
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_total - n_val, n_val], generator=gen)
    else:
        train_set, val_set = dataset, None
    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) if val_set is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model setup (inpu 1024 (mert) -> output 128 (context))
    model = ContextTransformer(input_dim=1024, hidden_dim=128).to(device)

    # predictor head ( 128 (Context) -> 1024 (predicted song embedding))
    predictor = nn.Linear(128, 1024).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=LR)
    criterion = nn.MSELoss()

    print(f">> [Context] Training on {n_total - n_val} sequences (val {n_val})...")

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train(); predictor.train()

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

        val_loss = None
        if val_loader is not None:
            model.eval(); predictor.eval()
            v = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    v += criterion(predictor(model(x)), y).item()
            val_loss = v / max(1, len(val_loader))
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            msg = f"   Epoch {epoch+1}/{EPOCHS} | train: {total_loss/max(1,len(dataloader)):.5f}"
            if val_loss is not None:
                msg += f" | val: {val_loss:.5f} (best {best_val:.5f})"
            print(msg)

    # save ONLY the encoder (best-val weights)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f">> [Context] model saved to {save_path}")

if __name__ == "__main__":
    train_context_model()