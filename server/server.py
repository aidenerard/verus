"""
server/server.py
FastAPI inference server wrapping the CNN1D GPR delamination classifier.

Startup behaviour
-----------------
  - If MODEL_PATH does not exist, download it from the URL in the
    MODEL_GDRIVE_URL environment variable using gdown.
  - Load CNN1D + TemporalAttention onto CPU (or CUDA if available).
  - Print "Model loaded successfully".

Endpoints
---------
  GET  /health    → liveness check
  POST /analyze   → multipart CSV upload → JSON + base64 C-scan PNG
"""

import os
import io
import base64
import shutil
import tempfile
import threading
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Configuration ─────────────────────────────────────────────────────────────
def _resolve_model_path() -> Path:
    """
    Resolve model path in order of priority:
    1. MODEL_PATH env var (explicit override)
    2. models/ directory — pick the highest-versioned .pth file
    3. model.pth in current directory (legacy fallback)
    """
    if env := os.environ.get("MODEL_PATH"):
        return Path(env)
    models_dir = Path("models")
    if models_dir.is_dir():
        candidates = sorted(models_dir.glob("*.pth"), reverse=True)
        if candidates:
            print(f"Auto-selected model: {candidates[0]}", flush=True)
            return candidates[0]
    return Path("model.pth")

MODEL_PATH = _resolve_model_path()
THRESHOLD  = 0.65        # P(sound) < THRESHOLD → delaminated
DC_OFFSET  = 32768
N_SAMPLES  = 512
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)

# ── Model architecture (must match cnn.py exactly) ────────────────────────────

class TemporalAttention(nn.Module):
    """Weighted sum over time steps. Input: (B, C, T) → Output: (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)
        return (x * weights.permute(0, 2, 1)).sum(dim=2)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,  128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        return self.head(x).squeeze(1)


# ── Data loading (identical normalisation to cnn.py / run.py) ─────────────────

def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def load_csv(fpath: Path) -> np.ndarray:
    """
    Auto-detecting CSV parser for GPR A-scan data.
    Returns normalised signals as (n_signals, 512).

    Handles multiple formats:
    1. Transposed format: each column = one A-scan (Amplitude_0, Amplitude_1, ...)
    2. Row format: each row = one A-scan
    3. Various metadata/header configurations
    """

    # Try different delimiters
    for delimiter in [',', '\t', ';']:
        try:
            df = pd.read_csv(fpath, header=None, delimiter=delimiter, low_memory=False)
            if df.shape[1] > 1:  # Valid parse
                break
        except Exception:
            continue
    else:
        raise ValueError("Could not parse CSV file with standard delimiters")

    # ── Step 1: Find where numeric data starts ────────────────────────────────
    data_start_row = None

    for idx in range(min(20, len(df))):
        row_values = df.iloc[idx].astype(str).values

        # Check if this looks like a header row
        has_amplitude = any('amplitude' in str(val).lower() for val in row_values)
        has_signal    = any('signal' in str(val).lower() for val in row_values)
        if has_amplitude or has_signal:
            data_start_row = idx + 1
            break

        # Check if this row contains mostly numeric data (>80%)
        numeric_count = sum(
            1 for val in row_values[1:]
            if _is_float(val)
        )
        if numeric_count > 0.8 * (len(row_values) - 1):
            data_start_row = idx
            break

    if data_start_row is None:
        raise ValueError("Could not find numeric data in CSV file")

    # ── Step 2: Extract numeric data ─────────────────────────────────────────
    raw_data     = df.iloc[data_start_row:].copy()
    numeric_data = raw_data.apply(pd.to_numeric, errors='coerce')
    numeric_data = numeric_data.dropna(axis=1, how='all')
    numeric_data = numeric_data.dropna(axis=0, how='all')

    if numeric_data.empty:
        raise ValueError("No numeric data found in CSV")

    data_array = numeric_data.values.astype(np.float32)

    # ── Step 3: Auto-detect orientation ──────────────────────────────────────
    rows, cols = data_array.shape

    # If rows ≈ 512 and cols << rows  → transposed (columns are A-scans)
    # If cols ≈ 512 and rows << cols  → correct orientation (rows are A-scans)
    if 400 <= rows <= 600 and cols < rows / 2:
        amps = data_array.T       # each column was an A-scan
    elif 400 <= cols <= 600 and rows < cols / 2:
        amps = data_array         # each row is already an A-scan
    elif rows > cols:
        amps = data_array.T
    else:
        amps = data_array

    n_signals, n_samples = amps.shape
    if n_signals == 0:
        raise ValueError("No A-scan signals found in CSV")

    # ── Step 4: Normalise sample count to N_SAMPLES (512) ────────────────────
    if n_samples > N_SAMPLES:
        amps = amps[:, :N_SAMPLES]
    elif n_samples < N_SAMPLES:
        amps = np.pad(amps, ((0, 0), (0, N_SAMPLES - n_samples)), mode='constant')

    # ── Step 5: Remove DC offset and apply Hann taper ────────────────────────
    if np.abs(amps.mean()) > 1000:   # data has DC offset (~32768)
        amps = (amps - DC_OFFSET) * _TAPER[np.newaxis, :]
    else:
        amps = amps * _TAPER[np.newaxis, :]

    # ── Step 6: Spatial averaging (radius=2) ─────────────────────────────────
    amps_avg = np.empty_like(amps)
    for i in range(n_signals):
        amps_avg[i] = amps[max(0, i - 2):i + 3].mean(axis=0)

    # ── Step 7: Z-score normalisation ────────────────────────────────────────
    std = amps_avg.std()
    if std < 1e-8:
        raise ValueError("Signal has no variation (constant values)")
    amps_avg = (amps_avg - amps_avg.mean()) / std

    return amps_avg  # (n_signals, 512)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        preds — int array, 1=sound / 0=delaminated, shape (n,)
        confs — float array, confidence in predicted class, shape (n,)
    """
    tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
    dl     = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=False)

    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            probs_list.append(model(xb.to(DEVICE)).sigmoid().cpu().numpy())

    probs = np.concatenate(probs_list)
    preds = (probs >= THRESHOLD).astype(int)
    confs = np.where(preds == 1, probs, 1.0 - probs)
    return preds, confs


# ── C-scan image (ASTM-style, returns PNG bytes) ──────────────────────────────

def render_cscan(
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
    file_names: list[str],
    bridge_name: str = "Bridge Deck",
) -> bytes:
    """
    Render a gaussian-smoothed RdYlGn delamination map with ASTM header box,
    bridge outline, lane markings, and legend.  Returns raw PNG bytes.
    """
    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    # Build P(delam) grid
    prob_grid = np.full((max_sigs, n_files), np.nan)
    for col, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for row, (pred, conf) in enumerate(zip(preds, confs)):
            prob_grid[row, col] = conf if pred == 0 else 1.0 - conf

    nan_mask = np.isnan(prob_grid)
    filled   = np.where(nan_mask, 0.0, prob_grid)
    smoothed = gaussian_filter(filled, sigma=(1.8, 1.8))
    masked   = np.ma.array(smoothed, mask=nan_mask)

    # Stats
    all_preds  = np.concatenate(file_preds)
    total_sigs = len(all_preds)
    n_delam    = int((all_preds == 0).sum())
    pct_delam  = n_delam / total_sigs * 100 if total_sigs else 0.0
    deck_area  = n_files * max_sigs
    delam_area = deck_area * pct_delam / 100.0

    # Span split at midpoint
    half  = n_files // 2
    spans = [(0, half, "Span 1"), (half, n_files, "Span 2")]

    fig = plt.figure(figsize=(24, 10), facecolor="white")
    ax  = fig.add_axes([0.06, 0.14, 0.70, 0.72])

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(
        masked,
        cmap=cmap, norm=norm,
        aspect="auto", origin="upper",
        extent=[-0.5, n_files - 0.5, max_sigs - 0.5, -0.5],
        interpolation="bilinear",
    )

    # Deck outline
    ax.add_patch(Rectangle(
        (-0.5, -0.5), n_files, max_sigs,
        linewidth=2.0, edgecolor="#1A1A1A", facecolor="none", zorder=5,
    ))

    # Lane markings at 1/3 and 2/3 width
    for frac in (0.33, 0.66):
        ax.axhline(
            max_sigs * frac,
            color="#333333", linewidth=1.1,
            linestyle=(0, (8, 6)), alpha=0.75, zorder=4,
        )

    # Pier at midpoint
    ax.axvline(half, color="#222266", linewidth=1.8, linestyle="-", alpha=0.85, zorder=4)
    ax.text(half, -1.8, "PIER", ha="center", va="bottom",
            fontsize=6.5, color="#222266", fontweight="bold", clip_on=False)

    # Span labels and bracket lines
    for s_fi, e_fi, s_label in spans:
        mid = (s_fi + e_fi) / 2.0
        ax.text(mid, -3.5, s_label, ha="center", va="bottom",
                fontsize=9, color="#111111", fontweight="bold", clip_on=False)
        for xb in [s_fi, e_fi]:
            ax.plot([xb, xb], [-0.5, -2.8], color="#555555",
                    linewidth=0.8, clip_on=False, zorder=3)

    # Axis labels and ticks
    ax.set_xlabel("Distance (feet)", fontsize=10, labelpad=6)
    ax.set_ylabel("Offset (feet)",   fontsize=10, labelpad=6)

    x_ticks = np.linspace(0, n_files - 1, min(11, n_files), dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks], fontsize=8)

    y_ticks = np.linspace(0, max_sigs - 1, min(9, max_sigs), dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(v) for v in y_ticks], fontsize=8)

    ax.tick_params(direction="out", length=4, width=0.8)

    # Scale bar (~15% of deck length)
    sb_len = max(5, round(n_files * 0.15 / 5) * 5)
    sb_x0  = n_files * 0.04
    sb_y   = max_sigs * 0.96
    ax.annotate("", xy=(sb_x0 + sb_len, sb_y), xytext=(sb_x0, sb_y),
                arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.2), zorder=6)
    ax.text(sb_x0 + sb_len / 2, sb_y + max_sigs * 0.025, f"{sb_len} ft",
            ha="center", va="bottom", fontsize=7.5)

    # North arrow
    na_x, na_y = n_files * 0.96, max_sigs * 0.88
    ax.annotate("", xy=(na_x, na_y - max_sigs * 0.12), xytext=(na_x, na_y),
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4,
                                mutation_scale=10), zorder=6)
    ax.text(na_x, na_y + max_sigs * 0.01, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    # ── Header box (top-right) ────────────────────────────────────────────────
    hdr = fig.add_axes([0.775, 0.38, 0.205, 0.52])
    hdr.set_axis_off()
    hdr.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="#333333", facecolor="#F5F5F5",
        transform=hdr.transAxes,
    ))
    hdr.add_patch(Rectangle(
        (0, 0.82), 1, 0.18, transform=hdr.transAxes,
        facecolor="#1F3864", edgecolor="none",
    ))
    delam_color = "#C0392B" if pct_delam > 15 else ("#E67E22" if pct_delam > 5 else "#27AE60")
    header_items = [
        ("BRIDGE DECK CONDITION ASSESSMENT", 11, "bold",   0.95, "white"),
        ("GPR – CONCRETE DELAMINATION",       9, "bold",   0.89, "white"),
        (f"Structure:   {bridge_name}",        8, "normal", 0.79, "#111111"),
        (f"Survey Date: {datetime.date.today().strftime('%B %d, %Y')}",
                                               8, "normal", 0.72, "#111111"),
        ("Standard:    ASTM D6087",            8, "normal", 0.65, "#111111"),
        (f"Scan lines:  {n_files}",            8, "normal", 0.58, "#111111"),
        (f"Signals:     {total_sigs:,}",       8, "normal", 0.51, "#111111"),
        (f"Deck Area:   {deck_area:,.0f} ft²", 8, "normal", 0.44, "#111111"),
        ("",                                   5, "normal", 0.37, "#111111"),
        (f"Total Delam: {pct_delam:.1f}%",     9, "bold",   0.30, delam_color),
        (f"Delam Area:  {delam_area:,.0f} ft²",8, "normal", 0.22, "#111111"),
    ]
    for text, fsize, fweight, y, color in header_items:
        hdr.text(0.05, y, text, transform=hdr.transAxes,
                 fontsize=fsize, fontweight=fweight, color=color, va="top")

    # ── Per-span summary box ──────────────────────────────────────────────────
    span_ax = fig.add_axes([0.775, 0.075, 0.205, 0.28])
    span_ax.set_axis_off()
    span_ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        linewidth=1.0, edgecolor="#AAAAAA", facecolor="#FAFAFA",
        transform=span_ax.transAxes,
    ))
    span_ax.text(0.5, 0.95, "SPAN SUMMARY", transform=span_ax.transAxes,
                 fontsize=8, fontweight="bold", color="#1F3864", ha="center", va="top")
    for col_x, col_lbl in [(0.04, "Span"), (0.32, "Delam%"),
                            (0.56, "Area ft²"), (0.78, "Delam ft²")]:
        span_ax.text(col_x, 0.84, col_lbl, transform=span_ax.transAxes,
                     fontsize=7, fontweight="bold", va="top")

    row_y = 0.72
    for s_fi, e_fi, s_label in spans:
        sp_preds = np.concatenate(file_preds[s_fi:e_fi]) if s_fi < e_fi else np.array([])
        sp_n     = len(sp_preds)
        sp_del   = int((sp_preds == 0).sum()) if sp_n else 0
        sp_pct   = sp_del / sp_n * 100 if sp_n else 0.0
        sp_area  = (e_fi - s_fi) * max_sigs
        sp_da    = sp_area * sp_pct / 100.0
        col      = "#C0392B" if sp_pct > 25 else ("#E67E22" if sp_pct > 10 else "#27AE60")
        span_ax.text(0.04, row_y, s_label,           transform=span_ax.transAxes, fontsize=7, va="top")
        span_ax.text(0.32, row_y, f"{sp_pct:.1f}%",  transform=span_ax.transAxes, fontsize=7, va="top", color=col, fontweight="bold")
        span_ax.text(0.56, row_y, f"{sp_area:,.0f}", transform=span_ax.transAxes, fontsize=7, va="top")
        span_ax.text(0.78, row_y, f"{sp_da:,.0f}",   transform=span_ax.transAxes, fontsize=7, va="top")
        row_y -= 0.20

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.06, 0.055, 0.70, 0.030])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Concrete Delamination Probability", fontsize=9, labelpad=4)
    cbar.set_ticks([0.0, 0.35, 0.50, 0.65, 1.0])
    cbar.set_ticklabels(["0%  (Sound)", "35%", "50%", "65%", "100%  (Delaminated)"], fontsize=7.5)
    cbar.outline.set_linewidth(0.8)

    # Swatch row
    sw_ax = fig.add_axes([0.06, 0.005, 0.70, 0.040])
    sw_ax.set_axis_off()
    for sx, sc, sl in [
        (0.02, "#CCCCCC", "N/A"),
        (0.18, "#2ECC71", "Low  (Sound)"),
        (0.44, "#F39C12", "Medium (Uncertain)"),
        (0.68, "#E84040", "High  (Delaminated)"),
    ]:
        sw_ax.add_patch(Rectangle((sx, 0.35), 0.035, 0.55,
                                   transform=sw_ax.transAxes,
                                   facecolor=sc, edgecolor="#555555", linewidth=0.6))
        sw_ax.text(sx + 0.042, 0.62, sl, transform=sw_ax.transAxes, fontsize=7.5, va="center")

    # Footnote
    fig.text(
        0.06, 0.002,
        f"Survey performed in accordance with ASTM D6087 – Standard Test Method for Evaluating "
        f"Asphalt-Covered Concrete Bridge Decks Using Ground Penetrating Radar.  |  "
        f"Threshold: P(delamination) > {1.0 - THRESHOLD:.2f}  |  "
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=6, color="#555555", va="bottom",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── App + startup ─────────────────────────────────────────────────────────────

app = FastAPI(title="Verus GPR Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)

_model: CNN1D | None = None


def _load_model_background() -> None:
    """
    Load the model in a background thread so the server port opens immediately
    and health checks succeed while the model is still initialising.
    """
    global _model

    print(f"[startup] Looking for model at: {MODEL_PATH.resolve()}", flush=True)

    # Download if missing
    if not MODEL_PATH.exists():
        gdrive_url = os.environ.get("MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading model from {gdrive_url} …", flush=True)
            try:
                import gdown
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(gdrive_url, str(MODEL_PATH), quiet=False, fuzzy=True)
            except Exception as exc:
                print(f"[startup] ERROR: gdown download failed: {exc}", flush=True)
                return
        else:
            print(
                f"[startup] WARNING: Model not found at {MODEL_PATH} and "
                "MODEL_GDRIVE_URL is not set. /analyze will return 503.",
                flush=True,
            )
            return

    if not MODEL_PATH.exists():
        print(f"[startup] ERROR: Model still missing after download: {MODEL_PATH}",
              flush=True)
        return

    try:
        m = CNN1D().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=False))
        m.eval()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        _model = m   # atomic assignment — safe for GIL-protected reads
        print(f"[startup] Model loaded successfully  "
              f"({n_params:,} parameters, device={DEVICE})", flush=True)
    except Exception as exc:
        print(f"[startup] ERROR loading model weights: {exc}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    # Kick off model loading in the background so the port opens immediately.
    # Health checks will return model_loaded=false until the thread finishes.
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()
    print("[startup] Server ready — model loading in background.", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_path": str(MODEL_PATH),
        "message": "Ready" if loaded else "Model loading in background, please retry in 30s",
    }


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tmpdir = Path(tempfile.mkdtemp(prefix="verus_infer_"))
    try:
        t0 = time.perf_counter()

        file_preds: list[np.ndarray] = []
        file_confs: list[np.ndarray] = []
        per_file_summary: list[dict] = []
        total_sigs = 0

        for upload in files:
            dest = tmpdir / (upload.filename or "upload.csv")
            content = await upload.read()
            dest.write_bytes(content)

            try:
                signals = load_csv(dest)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not parse {upload.filename}: {exc}",
                )

            preds, confs = run_inference(_model, signals)
            n         = len(preds)
            n_delam   = int((preds == 0).sum())
            delam_pct = round(n_delam / n * 100, 2) if n else 0.0

            file_preds.append(preds)
            file_confs.append(confs)
            per_file_summary.append({
                "filename":  upload.filename,
                "signals":   n,
                "delam_pct": delam_pct,
            })
            total_sigs += n

        # Aggregate stats
        all_preds       = np.concatenate(file_preds)
        n_del_total     = int((all_preds == 0).sum())
        delam_pct_total = round(n_del_total / total_sigs * 100, 2) if total_sigs else 0.0
        sound_pct_total = round(100.0 - delam_pct_total, 2)

        # Render C-scan PNG → base64
        png_bytes = render_cscan(
            file_preds, file_confs,
            [f.filename or "" for f in files],
        )
        cscan_b64 = base64.b64encode(png_bytes).decode("utf-8")

        elapsed = round(time.perf_counter() - t0, 3)

        return JSONResponse({
            "signals_analyzed":  total_sigs,
            "delamination_pct":  delam_pct_total,
            "sound_pct":         sound_pct_total,
            "analysis_time_sec": elapsed,
            "cscan_image":       cscan_b64,
            "per_file_summary":  per_file_summary,
        })

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
