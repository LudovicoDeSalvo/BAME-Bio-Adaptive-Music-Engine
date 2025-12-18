import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ContextTransformer(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, n_layers=2, n_heads=4):
        """
        Args:
            input_dim: Dimension of your audio embeddings (1024 for MERT-330M)
            hidden_dim: Size of the output Context Vector
        """
        super().__init__()
        
        # 1. Project Input (e.g. 1024) to Internal Dim (128)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # 3. Output Head
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Input: [Batch, Seq_Len, Input_Dim]
        Output: [Batch, Hidden_Dim]
        """
        # Embed & Position
        x = self.embedding(x) # [B, S, H]
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        output = self.transformer(x) # [B, S, H]
        
        # Take the last vector in the sequence
        last_state = output[:, -1, :] # [B, H]
        
        return self.head(last_state)