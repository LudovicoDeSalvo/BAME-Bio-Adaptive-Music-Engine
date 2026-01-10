import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # sinusoidal Logic
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ContextTransformer(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        """
        Transformer Encoder that compresses a sequence of songs into a single Context Vector.
        
        Args:
            input_dim: 1024 (MERT Embedding Size)
            hidden_dim: 128 (Context Vector Size)
        """
        super().__init__()
        
        # 1024 -> 128
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, 
                                                 dim_feedforward=hidden_dim*4, 
                                                 dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # output head
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Input: [batch, seq_len, 1024]
        Output: [batch, 128]
        """
        # embed
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # transform
        x = self.transformer(x)
        
        # aggregation. shape: [Batch, 128]
        last_state = x[:, -1, :]
        
        # final projection
        context_vector = self.head(last_state)
        
        return context_vector