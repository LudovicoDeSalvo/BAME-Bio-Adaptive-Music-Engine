import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from context.sequence_model import ContextTransformer

# --- Config ---
BATCH_SIZE = 32
EPOCHS = 10
SEQ_LEN = 5 # Look at last 5 songs to predict the 6th
LR = 1e-3
EMBEDDING_PATH = "data/processed/song_embeddings.npy"
ID_MAP_PATH = "data/processed/song_id_map.npy"

# --- Dataset ---
class SessionDataset(Dataset):
    def __init__(self, seq_len=5):
        """
        Creates sequences from the static HKU956 listening order.
        Since HKU956 is small, we simulate sessions by sliding a window 
        over the song embeddings in the order they appear in the file (just for pre-training).
        """
        if not os.path.exists(EMBEDDING_PATH):
            raise FileNotFoundError("Run 'process-audio' first to generate embeddings.")
            
        self.embeddings = np.load(EMBEDDING_PATH) # [N_songs, 768]
        self.data = []
        
        # Create sliding windows: [0,1,2,3,4] -> Predict [5]
        # In a real scenario, we would group by User Session.
        # Here we just treat the dataset as one long 'global' radio stream for pre-training.
        num_songs = len(self.embeddings)
        for i in range(num_songs - seq_len):
            window = self.embeddings[i : i+seq_len]
            target = self.embeddings[i+seq_len]
            self.data.append((window, target))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Training Function ---
def train_context_model():
    print(f">> [Context] Loading Embeddings from {EMBEDDING_PATH}...")
    
    try:
        dataset = SessionDataset(SEQ_LEN)
    except FileNotFoundError:
        print("❌ Error: Audio embeddings not found. Please run Option [5] first.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Check the actual dimension of loaded data
    sample_dim = dataset.embeddings.shape[1] 
    print(f">> [Context] Detected Embedding Dimension: {sample_dim}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContextTransformer(input_dim=sample_dim).to(device)
    
    # We want the Context Vector to be close to the Next Song's Embedding
    # But since dimensions differ (128 vs 768), we need a projection head for loss calc
    predictor = nn.Linear(128, sample_dim).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=LR)
    criterion = nn.MSELoss()

    print(f">> [Context] Starting Pre-training on {len(dataset)} sequences...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # 1. Get Context Vector (History)
            ctx_vector = model(x) # [32, 128]
            
            # 2. Predict Next Song Embedding
            pred_next_song = predictor(ctx_vector) # [32, 768]
            
            # 3. Loss: Did we predict the acoustic features of the next song?
            loss = criterion(pred_next_song, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save
    if not os.path.exists("context/checkpoints"):
        os.makedirs("context/checkpoints")
    torch.save(model.state_dict(), "context/checkpoints/context_model.pth")
    print(">> ✅ Context Model Saved to context/checkpoints/context_model.pth")

if __name__ == "__main__":
    train_context_model()