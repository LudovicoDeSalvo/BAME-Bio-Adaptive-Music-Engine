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
    def __init__(self, state_dim=224, action_dim=1024, physio_dim=64, hidden_dim=512, layers=3,
                 max_delta_scale=1.0, physio_std=None):
        """
        Predicts next physio state given (Current State + Action).

        max_delta_scale : caps how far a single step can move physio. The head's
                          raw delta is tanh-squashed to (-1,1), then scaled by the
                          per-dim physio std and this factor, so one step moves at
                          most ~max_delta_scale std per dimension. Bounding the
                          per-step jump is the structural guard against the
                          single off-manifold steps that compound into rollout
                          divergence (the world model is otherwise unconstrained
                          and explodes to ~20 std over a 50-step rollout).
        physio_std      : per-dim std used for the bound (shape [physio_dim]).
                          Persisted as a buffer so inference uses the exact same
                          geometry the model was trained with.
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

        # Buffers travel with the checkpoint (saved in state_dict), so a loaded
        # model reproduces the exact delta bound without re-passing these.
        if physio_std is None:
            std_buf = torch.ones(physio_dim)
        else:
            std_buf = torch.as_tensor(np.asarray(physio_std), dtype=torch.float32)
        self.register_buffer("physio_std", std_buf)
        self.register_buffer("max_delta_scale", torch.tensor(float(max_delta_scale)))

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

        delta_raw = self.head(x)

        # Bounded residual: tanh -> (-1,1), scaled by per-dim std and global
        # factor. Broadcasts physio_std [physio_dim] over the batch.
        delta_physio = torch.tanh(delta_raw) * self.physio_std * self.max_delta_scale

        current_physio = state[:, :self.physio_dim]

        next_physio = current_physio + delta_physio
        return next_physio

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device='cpu', **kwargs):

        model = cls(**kwargs)
        # strict=False so a checkpoint trained before the bounded-delta buffers
        # (physio_std, max_delta_scale) still loads — the buffers keep their
        # constructor defaults. Such a checkpoint is stale (its weights were
        # trained for the OLD unbounded forward); warn so the model gets retrained.
        result = model.load_state_dict(torch.load(path, map_location=device), strict=False)
        missing = getattr(result, "missing_keys", [])
        if any(k in missing for k in ("physio_std", "max_delta_scale")):
            print(" !!! [WorldModel] stale checkpoint (pre bounded-delta). "
                  "Retrain the World Model [7] for correct rollout dynamics.")
        model.to(device)
        model.eval()
        return model