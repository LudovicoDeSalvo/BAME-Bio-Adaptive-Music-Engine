import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from physio.encoder import DualStreamEncoder, split_streams
from data.windows import apply_window_norm
from utils.common import ensure_dir, load_config, resolve_path, set_seed

class PhysioDataset(Dataset):
    def __init__(self, cache_path, holdout_user=None):
        data = np.load(cache_path, allow_pickle=True)

        # Load the RAW per-window features (align_and_slice stores them
        # un-normalized). We fit the z-score normalizer HERE, on the
        # holdout-excluded split only, so the evaluation subject never leaks into
        # the normalizer (true LOSO refits this per fold). The fitted stats are
        # persisted alongside the embeddings so inference applies the exact same
        # normalizer.
        raw_windows = np.asarray(data['window_features'], dtype=np.float32)
        v = torch.tensor(data['valence'], dtype=torch.float32)
        a = torch.tensor(data['arousal'], dtype=torch.float32)
        all_targets = torch.stack([v, a], dim=1)
        all_pids = data['participant_ids']
        all_clip_ids = data['clip_ids']
        all_song_ids = data['song_ids']
        # song_no = per-playback trial id. Carry it through the encoder so the
        # world model can key transitions by (participant, song_id, song_no) and
        # never merge distinct replays. Legacy caches lack it -> None.
        all_song_nos = data['song_nos'] if 'song_nos' in data else None

        # filtering
        if holdout_user:
            print(f">> [Dataset] Excluding user: {holdout_user}")
            mask = [str(pid) != str(holdout_user) for pid in all_pids]
            mask = np.array(mask)
        else:
            mask = np.ones(len(all_pids), dtype=bool)

        masked_windows = raw_windows[mask]

        # Fit per-channel z-score stats over the masked (training) split, using
        # real (non-padding) windows only. Padding rows are all-zero and excluded
        # so they don't bias the mean/std toward zero.
        flat = masked_windows.reshape(-1, masked_windows.shape[-1])
        valid = ~np.all(flat == 0, axis=1)
        rows = flat[valid] if np.any(valid) else flat
        win_mean = rows.mean(axis=0)
        win_std = rows.std(axis=0)
        win_std[win_std == 0] = 1.0
        self.win_mean = win_mean.astype(np.float32)
        self.win_std = win_std.astype(np.float32)

        normed = apply_window_norm(masked_windows, self.win_mean, self.win_std)

        # Per-sample mean over NON-padding windows (30-dim). This is the C3
        # auxiliary reconstruction target: forcing the 64-dim latent to decode
        # back to real signal statistics keeps it from collapsing onto the 2
        # supervised affect dims. Padding rows are all-zero post-norm, so we
        # average over valid timesteps only.
        pad_mask = np.all(normed == 0, axis=-1)                     # [N, T]
        valid_counts = np.maximum((~pad_mask).sum(axis=1, keepdims=True), 1)
        mean_windows = normed.sum(axis=1) / valid_counts            # [N, 30]
        self.mean_windows = torch.tensor(mean_windows, dtype=torch.float32)

        # mask
        self.windows = torch.tensor(normed, dtype=torch.float32)
        self.targets = all_targets[mask]
        self.participants = all_pids[mask]
        self.ids = all_song_ids[mask]
        self.clip_ids = all_clip_ids[mask]
        self.song_nos = all_song_nos[mask] if all_song_nos is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        dermal_stream, cardio_stream = split_streams(self.windows[idx])
        return dermal_stream, cardio_stream, self.targets[idx], self.mean_windows[idx]

def train_physio_model(epochs=20, config_path="configs/config.yaml", holdout_user=None,
                       patience=20):
    config = load_config(config_path)
    paths = config.get("paths", {})
    set_seed(int(config.get("training", {}).get("seed", 42)))

    cache_path = resolve_path(paths.get("physio_cache", "data/processed/physio_cache.npz"))
    model_path = resolve_path("physio/checkpoints/physio_encoder.pth")
    embeddings_out = resolve_path("data/processed/physio_embeddings.npz")

    ensure_dir(os.path.dirname(model_path))

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing {cache_path}. Run Option [1] first.")

    # Explicit holdout (per-fold LOSO) overrides the config default.
    if holdout_user is None:
        holdout_user = config['training'].get('holdout_user')
    dataset = PhysioDataset(cache_path, holdout_user)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("training", {}).get("seed", 42))

    # Deterministic 85/15 train/val split for best-checkpoint selection. The split
    # governs ONLY model selection; embeddings are generated over the FULL
    # holdout-excluded dataset afterwards.
    n_total = len(dataset)
    n_val = max(1, int(round(0.15 * n_total))) if n_total > 1 else 0
    if n_val > 0:
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_total - n_val, n_val], generator=gen)
    else:
        train_set, val_set = dataset, None
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False) if val_set is not None else None

    model = DualStreamEncoder(embedding_dim=64).to(device)
    head = nn.Linear(64, 2).to(device)
    # C3: decoder reconstructs the per-sample mean window vector (30-dim) from the
    # 64-dim latent. Trained jointly but NOT persisted (the encoder checkpoint
    # shape stays identical for inference).
    decoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 30)).to(device)
    RECON_W = 0.5

    params = list(model.parameters()) + list(head.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=5e-4)
    criterion = nn.MSELoss()

    def _loss(d_in, c_in, target, mean_win):
        emb = model(d_in, c_in)
        return criterion(head(emb), target) + RECON_W * criterion(decoder(emb), mean_win)

    print(f">> [Physio] Training on {n_total - n_val} samples (val {n_val})...")

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train(); head.train(); decoder.train()
        total_loss = 0
        for d_in, c_in, target, mean_win in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            d_in, c_in, target, mean_win = (d_in.to(device), c_in.to(device),
                                            target.to(device), mean_win.to(device))
            optimizer.zero_grad()
            loss = _loss(d_in, c_in, target, mean_win)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation + best-checkpoint selection (encoder weights only).
        val_loss = None
        if val_loader is not None:
            model.eval(); head.eval(); decoder.eval()
            v = 0.0
            with torch.no_grad():
                for d_in, c_in, target, mean_win in val_loader:
                    d_in, c_in, target, mean_win = (d_in.to(device), c_in.to(device),
                                                    target.to(device), mean_win.to(device))
                    v += _loss(d_in, c_in, target, mean_win).item()
            val_loss = v / max(1, len(val_loader))
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            msg = f"   Epoch {epoch+1}: train = {total_loss / max(1, len(train_loader)):.4f}"
            if val_loss is not None:
                msg += f" | val = {val_loss:.4f} (best {best_val:.4f})"
            print(msg)

        # Early stop: val plateaued for `patience` epochs -> stop, keep best weights.
        if val_loader is not None and patience and epochs_no_improve >= patience:
            print(f">> [Physio] early stop at epoch {epoch+1} "
                  f"(no val improvement for {patience} epochs; best epoch {best_epoch}, "
                  f"best val {best_val:.4f})")
            break

    # Restore best-val weights before saving + generating embeddings, so the
    # checkpoint + embeddings reflect the best-generalizing model, not the last
    # (overfit) epoch.
    model.load_state_dict(best_state)
    model.save(model_path)
    print(f">> [Physio] model saved! (best epoch {best_epoch}, val {best_val:.4f})")

    # Overwrites embeddings_out fully (np.savez truncates) -> this train run's
    # embeddings are the ones downstream uses.
    generate_embeddings(model, dataset, embeddings_out, device)

def generate_embeddings(model, dataset, path, device):
    print(f">> [Physio] generating embeddings...")
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embs = []

    with torch.no_grad():
        for d_in, c_in, _, _ in loader:
            d_in, c_in = d_in.to(device), c_in.to(device)
            emb = model(d_in, c_in)
            all_embs.append(emb.cpu().numpy())

    final_pool = np.vstack(all_embs)

    # save clip_ids + song_nos so downstream (world model) can key transitions by
    # the unique (participant, song_id, song_no) playback trial. Omit song_nos
    # only for legacy caches that never had it.
    save_kwargs = dict(
        embeddings=final_pool,
        song_ids=dataset.ids,
        participant_ids=dataset.participants,
        clip_ids=dataset.clip_ids,
        # Persist the window normalizer fitted on this (holdout-excluded) split so
        # inference normalizes the holdout subject with the exact same stats the
        # encoder trained on — and never with stats that saw the holdout.
        win_norm_mean=dataset.win_mean,
        win_norm_std=dataset.win_std,
        # M3: persist the per-dim latent std so the env's scale-aware reward/eval
        # geometry has a single source of truth (env loads this instead of
        # recomputing). Fixed by the encoder config -> runs are comparable only
        # within the same config.
        physio_std=np.std(final_pool, axis=0).astype(np.float32),
    )
    if dataset.song_nos is not None:
        save_kwargs["song_nos"] = dataset.song_nos

    np.savez(path, **save_kwargs)

    print(f">> [Physio] saved {final_pool.shape} embeddings to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train_physio_model(epochs=args.epochs, patience=args.patience)