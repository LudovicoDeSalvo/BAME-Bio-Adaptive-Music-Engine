import torch
import torch.nn as nn


def split_streams(windows):
    """Split a [..., 30] per-window feature tensor into the encoder's two streams.

    Channel layout is fixed by data.windows.compute_window_features:
    EDA(0:6) BVP(6:12) TEMP(12:18) HR(18:24) IBI(24:30).
      dermal = [EDA, TEMP]            -> 12 channels
      cardio = [BVP, HR, IBI]         -> 18 channels
    Single source of truth so training and inference can never drift into a
    silent channel scramble. Works for [T, 30] or [B, T, 30] (last-axis slicing).
    """
    eda = windows[..., 0:6]
    bvp = windows[..., 6:12]
    temp = windows[..., 12:18]
    hr = windows[..., 18:24]
    ibi = windows[..., 24:30]
    dermal = torch.cat([eda, temp], dim=-1)
    cardio = torch.cat([bvp, hr, ibi], dim=-1)
    return dermal, cardio


class DualStreamEncoder(nn.Module):
    def __init__(self, hidden_dim=64, embedding_dim=64):
        super(DualStreamEncoder, self).__init__()
        
        # --- Stream 1: EDA + TEMP (12 features) ---

        self.dermal_cnn = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dermal_lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, 
                                   num_layers=1, batch_first=True, bidirectional=True)
        
        # --- Stream 2: BVP + HR + IBI (18 features) ---

        self.cardio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cardio_lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, 
                                   num_layers=1, batch_first=True, bidirectional=True)
        
        # --- Fusion ---
        fusion_dim = (hidden_dim * 2) + (hidden_dim * 2)
        
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim) # 64 output
        )

    def forward(self, dermal, cardio):
        """
        1: [batch, seq_len, 12]
        2: [batch, seq_len, 18]
        """

        d_in = dermal.transpose(1, 2)
        c_in = cardio.transpose(1, 2)
        
        d_feat = self.dermal_cnn(d_in) 
        c_feat = self.cardio_cnn(c_in)
        
        d_lstm_in = d_feat.transpose(1, 2)
        c_lstm_in = c_feat.transpose(1, 2)
        
        _, (h_n_d, _) = self.dermal_lstm(d_lstm_in)
        _, (h_n_c, _) = self.cardio_lstm(c_lstm_in)
        
        d_vec = torch.cat([h_n_d[-2], h_n_d[-1]], dim=1)
        c_vec = torch.cat([h_n_c[-2], h_n_c[-1]], dim=1)
        
        combined = torch.cat([d_vec, c_vec], dim=1)
        return self.projector(combined)

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path, device='cpu'):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        return model