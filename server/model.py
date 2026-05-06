"""
server/model.py
CNN1D model architecture + shared inference constants.
"""

import torch
import torch.nn as nn

# ── Inference constants ───────────────────────────────────────────────────────

THRESHOLD   = 0.65       # P(sound) < THRESHOLD → delaminated
DC_OFFSET   = 32768
N_SAMPLES   = 512
INFER_BATCH = 256        # max signals per forward pass (~33 MB peak)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model architecture ────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Weighted sum over time steps. Input: (B, C, T) → Output: (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)
        return (x * weights.permute(0, 2, 1)).sum(dim=2)


class CNN1D(nn.Module):
    """
    Architecture must exactly match the saved model.pth weights:
      net.0  Conv1d(1,  32, 7, padding=3)
      net.1  ReLU
      net.2  MaxPool1d(2)
      net.3  Conv1d(32, 64, 5, padding=2)
      net.4  ReLU
      net.5  MaxPool1d(2)
      net.6  Flatten  → 64 * 128 = 8192
      net.7  Linear(8192, 64)
      net.8  ReLU
      net.9  Dropout(0.3)
      net.10 Linear(64, 1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(8192, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
