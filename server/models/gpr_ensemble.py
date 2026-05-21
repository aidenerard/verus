"""
gpr_ensemble.py — Inference wrappers for the three trained GPR models.
All models use standardized 512-sample normalized input.

Default checkpoint paths resolve to the same directory as this file, so the
ensemble can be constructed with no arguments once horizon_model.pth,
thickness_model.pth, and corrosion_model.pth are placed alongside it.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample as scipy_resample

MAX_REBAR_DEPTH_CM = 10.0
MAX_THICKNESS_CM   = 35.0

MODELS_DIR     = os.path.dirname(os.path.abspath(__file__))
HORIZON_PATH   = os.path.join(MODELS_DIR, "horizon_model.pth")
THICKNESS_PATH = os.path.join(MODELS_DIR, "thickness_model.pth")
CORROSION_PATH = os.path.join(MODELS_DIR, "corrosion_model.pth")


def preprocess_trace(raw_trace: np.ndarray, target_samples: int = 512) -> np.ndarray:
    if len(raw_trace) != target_samples:
        raw_trace = scipy_resample(raw_trace, target_samples)
    raw_trace = raw_trace - raw_trace.mean()
    max_abs = np.abs(raw_trace).max()
    if max_abs > 0:
        raw_trace = raw_trace / max_abs
    return raw_trace.astype(np.float32)


def _preprocess_batch(raw_traces: np.ndarray, target_samples: int = 512) -> np.ndarray:
    if raw_traces.shape[1] != target_samples:
        raw_traces = scipy_resample(raw_traces, target_samples, axis=1)
    raw_traces = raw_traces - raw_traces.mean(axis=1, keepdims=True)
    norms = np.abs(raw_traces).max(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (raw_traces / norms).astype(np.float32)


# ── Shared backbone ───────────────────────────────────────────────────────────

class _TemporalAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x.transpose(1, 2)), dim=1)
        return (x.transpose(1, 2) * w).sum(dim=1)


def _make_conv_backbone() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1,   32,  7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32,  64,  5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
    )


# ── Model definitions (must match training notebooks) ────────────────────────

class HorizonCNN(nn.Module):
    """Rebar horizon depth regression. Output: normalized depth 0-1."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = _make_conv_backbone()
        self.attn = _TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(-1)


class ThicknessCNN(nn.Module):
    """Deck thickness regression. Output: normalized thickness 0-1."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = _make_conv_backbone()
        self.attn = _TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(-1)


class CorrosionCNN(nn.Module):
    """Corrosion risk binary classifier. Output: raw logit (apply sigmoid for probability)."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = _make_conv_backbone()
        self.attn = _TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(-1)


# ── Ensemble ──────────────────────────────────────────────────────────────────

class GPRModelEnsemble:
    """
    Loads and runs the GPR model ensemble. Horizon is required; thickness and
    corrosion are optional — pass None to skip loading either of those and the
    corresponding predict() outputs come back as None.

    Input traces are format-agnostic — resampled to 512 samples before
    inference. predict() processes traces in batches to bound peak memory on
    constrained hosts (Render free tier).
    """

    def __init__(
        self,
        horizon_path:   str         = HORIZON_PATH,
        thickness_path: str | None = THICKNESS_PATH,
        corrosion_path: str | None = CORROSION_PATH,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._horizon   = self._load(HorizonCNN(), horizon_path)
        self._thickness = self._load(ThicknessCNN(), thickness_path) if thickness_path else None
        self._corrosion = self._load(CorrosionCNN(), corrosion_path) if corrosion_path else None

    def _load(self, model: nn.Module, path: str) -> nn.Module:
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def predict(self, traces: np.ndarray, batch_size: int = 256) -> dict:
        """
        traces: np.ndarray shape (N, any_length) — raw traces, any format.

        Returns dict:
            rebar_depth_cm    — np.ndarray (N,)
            rebar_depth_in    — np.ndarray (N,)
            deck_thickness_cm — np.ndarray (N,) or None if thickness model absent
            deck_thickness_in — np.ndarray (N,) or None
            corrosion_risk    — np.ndarray (N,) prob 0-1, or None if absent
        """
        horizon_chunks, thick_chunks, corr_chunks = [], [], []
        n = len(traces)

        for start in range(0, n, batch_size):
            batch = _preprocess_batch(traces[start:start + batch_size], target_samples=512)
            x = torch.from_numpy(batch).unsqueeze(1).to(self.device)
            with torch.no_grad():
                horizon_chunks.append(self._horizon(x).cpu().numpy())
                if self._thickness is not None:
                    thick_chunks.append(self._thickness(x).cpu().numpy())
                if self._corrosion is not None:
                    corr_chunks.append(self._corrosion(x).cpu().numpy())

        depth_norm = np.concatenate(horizon_chunks)
        depth_cm   = depth_norm * MAX_REBAR_DEPTH_CM

        out: dict = {
            "rebar_depth_cm": depth_cm,
            "rebar_depth_in": depth_cm / 2.54,
        }

        if thick_chunks:
            thick_cm = np.concatenate(thick_chunks) * MAX_THICKNESS_CM
            out["deck_thickness_cm"] = thick_cm
            out["deck_thickness_in"] = thick_cm / 2.54
        else:
            out["deck_thickness_cm"] = None
            out["deck_thickness_in"] = None

        if corr_chunks:
            corr_logit = np.concatenate(corr_chunks)
            out["corrosion_risk"] = 1.0 / (1.0 + np.exp(-corr_logit))
        else:
            out["corrosion_risk"] = None

        return out
