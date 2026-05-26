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
    def __init__(self, in_channels=1, conv_channels=[32, 128, 128], head_hidden=128):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(c3)
        self.head = nn.Sequential(
            nn.Linear(c3, head_hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return self.head(x).squeeze(1)


class RebarDepthCNN(nn.Module):
    """Legacy 2-channel 256-sample model. Kept for backwards compat; superseded by HorizonCNN."""
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(-1)


class HorizonCNN(nn.Module):
    """
    Input:  (batch, 1, 512) — DC-removed, max-abs normalised raw waveform
    Output: (batch,) normalised depth in [0, 1]; multiply × 300 / 25.4 for inches
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.attn       = TemporalAttention(128)
        self.depth_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depth_head(self.attn(self.conv(x))).squeeze(-1)
