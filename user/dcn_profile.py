import torch
import torch.nn as nn

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()

        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        nn.init.xavier_uniform_(self.weight.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x0, x):

        xw = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * xw + self.bias + x

class DCNProfile(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=32, num_cross_layers=3, hidden_dim=64):
        """
        Deep & Cross Network (DCN)
        args:
            input_dim: 5 (TIPI scores)
            embedding_dim: 32
        """
        super(DCNProfile, self).__init__()
        
        self.num_cross_layers = num_cross_layers
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])
        
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        fusion_dim = input_dim + hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.Tanh() 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)
            
        x_deep = self.deep_net(x)
        combined = torch.cat([x_cross, x_deep], dim=1)
        embedding = self.encoder(combined)
        reconstruction = self.decoder(embedding)
        
        return embedding, reconstruction

    def get_embedding(self, x):
        """Helper to use in inference"""
        with torch.no_grad():
            emb, _ = self.forward(x)
        return emb

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path, input_dim=5, embedding_dim=32, device='cpu'):
        model = cls(input_dim, embedding_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model