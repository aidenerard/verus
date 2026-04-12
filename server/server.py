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
  GET  /memory    → RAM usage via psutil
  POST /analyze   → multipart CSV upload → JSON + base64 C-scan PNG

Memory budget (Render free tier = 512 MB RAM)
---------------------------------------------
  Python runtime + uvicorn + torch idle:  ~150 MB
  Model weights (CNN1D, 86 k params):     ~  1 MB
  One SDNET file loaded (16 383 × 512 × 4 bytes float32):  ~33 MB
  Spatial averaging peak (amps + amps_avg simultaneously): ~66 MB
  Inference batch (INFER_BATCH = 1 000 signals):           ~ 2 MB
  C-scan grid downsampled to 500 × 100:                    ~  0.4 MB
  C-scan PNG at 72 DPI:                                    ~ 5 MB
  ─────────────────────────────────────────────────────────────────
  Peak total (processing one file at a time):              ~225 MB
  Headroom to 512 MB limit:                                ~287 MB ✓
"""

import gc
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# All model / inference / rendering logic lives in run.py
from run import (
    CNN1D,
    load_csv,
    run_inference,
    render_cscan_b64,
    INFER_BATCH,
    MAX_GRID_ROWS,
    MAX_GRID_COLS,
    DEVICE,
)

# ── Server-specific configuration ────────────────────────────────────────────


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


MODEL_PATH   = _resolve_model_path()
MAX_FILE_MB  = 50    # per-file upload limit
MAX_TOTAL_MB = 500   # total upload limit across all files


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


@app.get("/memory")
def memory() -> dict:
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb":  round(vm.used      / 1024 ** 2, 1),
        "ram_total_mb": round(vm.total     / 1024 ** 2, 1),
        "ram_percent":  vm.percent,
        "ram_free_mb":  round(vm.available / 1024 ** 2, 1),
    }


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)) -> JSONResponse:
    print(f"[analyze] Received request: {len(files)} file(s)", flush=True)
    if _model is None:
        print("[analyze] Model not loaded yet — returning 503", flush=True)
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tmpdir = Path(tempfile.mkdtemp(prefix="verus_infer_"))
    try:
        t0 = time.perf_counter()
        vm = psutil.virtual_memory()
        print(f"[analyze] RAM at start: {vm.used//1024**2} MB used / "
              f"{vm.total//1024**2} MB total ({vm.percent:.1f}%)", flush=True)

        file_preds: list[np.ndarray] = []
        file_confs: list[np.ndarray] = []
        file_names: list[str]        = []
        per_file_summary: list[dict] = []
        total_sigs  = 0
        total_bytes = 0

        for i, upload in enumerate(files):
            fname = upload.filename or f"upload_{i}.csv"
            print(f"[analyze] File {i+1}/{len(files)}: {fname}", flush=True)

            content = await upload.read()
            file_mb = len(content) / 1024 ** 2

            # ── Per-file size guard ───────────────────────────────────────────
            if file_mb > MAX_FILE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"{fname} is {file_mb:.1f} MB — limit is {MAX_FILE_MB} MB per file.",
                )
            total_bytes += len(content)
            if total_bytes / 1024 ** 2 > MAX_TOTAL_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total upload exceeds {MAX_TOTAL_MB} MB. "
                           "Please upload one bridge at a time.",
                )

            dest = tmpdir / fname
            dest.write_bytes(content)
            del content                                   # free upload bytes now

            try:
                signals = load_csv(dest)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not parse {fname}: {exc}",
                )

            print(f"[analyze]   → {signals.shape[0]} signals loaded, "
                  f"running inference in batches of {INFER_BATCH} …", flush=True)
            preds, confs = run_inference(_model, signals)
            del signals                                   # free after inference
            dest.unlink(missing_ok=True)                  # delete temp file now
            gc.collect()

            n         = len(preds)
            n_delam   = int((preds == 0).sum())
            delam_pct = round(n_delam / n * 100, 2) if n else 0.0
            vm2 = psutil.virtual_memory()
            print(f"[analyze]   → {n} signals, {delam_pct}% delam | "
                  f"RAM: {vm2.used//1024**2} MB ({vm2.percent:.1f}%)", flush=True)

            file_preds.append(preds)
            file_confs.append(confs)
            file_names.append(fname)
            per_file_summary.append({
                "filename":  fname,
                "signals":   n,
                "delam_pct": delam_pct,
            })
            total_sigs += n

        # Aggregate stats
        all_preds       = np.concatenate(file_preds)
        n_del_total     = int((all_preds == 0).sum())
        del all_preds
        delam_pct_total = round(n_del_total / total_sigs * 100, 2) if total_sigs else 0.0
        sound_pct_total = round(100.0 - delam_pct_total, 2)

        # Render C-scan → base64 PNG string (in-memory, no disk I/O)
        print(f"[analyze] Rendering C-scan: {len(file_preds)} files, "
              f"{total_sigs:,} signals (grid capped at "
              f"{MAX_GRID_ROWS}×{MAX_GRID_COLS}) …", flush=True)
        try:
            cscan_b64 = render_cscan_b64(file_preds, file_confs, file_names)
            print(f"[analyze] C-scan rendered OK ({len(cscan_b64)//1024} KB b64)",
                  flush=True)
        except Exception as render_exc:
            print(f"[analyze] WARNING: C-scan render failed: {render_exc}", flush=True)
            cscan_b64 = ""

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
