import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # standard Transformer Feed-Forward Block logic
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x

class WorldModel(nn.Module):
    def __init__(self, state_dim=224, action_dim=1024, physio_dim=64, hidden_dim=512, layers=3):
        """
        Predicts next physio state given (Current State + Action).
        """
        super(WorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.physio_dim = physio_dim
        
        input_dim = state_dim + action_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(layers)
        ])
        
        self.head = nn.Linear(hidden_dim, physio_dim)

    def forward(self, state, action):
        """
        state: [batch, 224]
        action: [batch, 1024]
        return: next_physio_state [batch, 64]
        """

        x = torch.cat([state, action], dim=1)
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x)
            
        delta_physio = self.head(x)
        
        current_physio = state[:, :self.physio_dim]
        
        next_physio = current_physio + delta_physio
        return next_physio

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device='cpu', **kwargs):

        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model