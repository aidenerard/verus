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
  GET  /health          → liveness check
  GET  /memory          → RAM usage via psutil
  GET  /formats         → supported GPR file formats
  POST /analyze         → multipart upload → {job_id} (returns immediately)
  GET  /job/{job_id}    → poll job status + result

Background job pattern (fixes Render 60-second timeout)
---------------------------------------------------------
  POST /analyze reads file bytes, saves to a tmpdir, enqueues a job in a
  ThreadPoolExecutor (max_workers=2), and returns {job_id} immediately.
  The frontend polls GET /job/{job_id} every 3 s until status is
  "complete" or "failed".  Results are stored in _jobs (in-memory dict)
  and written to Supabase analysis_jobs + Storage when env vars are set.

Memory budget (Render free tier = 512 MB RAM)
---------------------------------------------
  Python runtime + uvicorn + torch idle:  ~150 MB
  Model weights (CNN1D, 86 k params):     ~  1 MB
  One SDNET file loaded (16 383 × 512 × 4 bytes float32):  ~33 MB
  Spatial averaging peak:                 ~ 66 MB
  Inference batch (INFER_BATCH = 1 000):  ~  2 MB
  C-scan PNG at 72 DPI:                   ~  5 MB
  Peak total (one file at a time):        ~225 MB ✓
"""

import gc
import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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

# Format detection and binary-to-CSV conversion
from ingest import (
    detect_and_convert,
    SUPPORTED_EXTENSIONS,
    COMPANION_EXTENSIONS,
    FORMAT_INFO,
)

# ── Supabase client (optional — graceful if env vars not set) ─────────────────

_supabase = None

def _init_supabase():
    global _supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")  # service role key for server
    if url and key:
        try:
            from supabase import create_client
            _supabase = create_client(url, key)
            print("[startup] Supabase client initialised.", flush=True)
        except Exception as exc:
            print(f"[startup] WARNING: Supabase init failed: {exc}", flush=True)
    else:
        print(
            "[startup] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — "
            "DB writes and auth verification disabled.",
            flush=True,
        )


# ── JWT auth (optional) ───────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """
    Verify the Supabase JWT and return the user_id string, or None if:
      - no token was provided
      - Supabase client is not configured
    Raises HTTP 401 only on an invalid/expired token.
    """
    if credentials is None:
        return None
    if _supabase is None:
        return None

    token = credentials.credentials
    try:
        resp = _supabase.auth.get_user(token)
        return resp.user.id
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


# ── Server-specific configuration ─────────────────────────────────────────────

MODEL_PATH   = Path(os.environ.get("MODEL_PATH", "model.pth"))
MAX_FILE_MB  = 50
MAX_TOTAL_MB = 2000

# Background job state
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── App + startup ─────────────────────────────────────────────────────────────

app = FastAPI(title="Verus GPR Inference Server", version="2.0.0")

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
    global _model
    print(f"[startup] Looking for model at: {MODEL_PATH.resolve()}", flush=True)

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
        print(f"[startup] ERROR: Model still missing after download: {MODEL_PATH}", flush=True)
        return

    try:
        m = CNN1D().to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        m.eval()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        _model = m
        print(f"[startup] Model loaded  ({n_params:,} params, device={DEVICE})", flush=True)
    except Exception as exc:
        print(f"[startup] ERROR loading model weights: {exc}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    _init_supabase()
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()
    print("[startup] Server ready — model loading in background.", flush=True)


# ── Background job worker ─────────────────────────────────────────────────────

def run_analysis_job(
    job_id: str,
    file_data: list[tuple[str, bytes]],   # [(filename, bytes), ...]
    user_id: Optional[str],
    tmpdir: Path,
) -> None:
    """
    Runs in a ThreadPoolExecutor worker thread.
    Saves files, runs inference, writes results to Supabase, updates _jobs.
    """
    _jobs[job_id]["status"] = "processing"
    t0 = time.perf_counter()

    try:
        # Save uploaded bytes to tmpdir
        saved_paths: list[tuple[str, Path]] = []
        for fname, content in file_data:
            dest = tmpdir / fname
            dest.write_bytes(content)
            saved_paths.append((fname, dest))
            print(f"[job:{job_id}] Saved {fname} ({len(content)/1024:.1f} KB)", flush=True)

        # ── Inference (same logic as original /analyze) ───────────────────────
        file_preds: list[np.ndarray] = []
        file_confs: list[np.ndarray] = []
        file_names: list[str]        = []
        per_file_summary: list[dict] = []
        total_sigs = 0

        for fname, dest in saved_paths:
            ext = dest.suffix.lower()
            if ext in COMPANION_EXTENSIONS:
                print(f"[job:{job_id}] Skipping companion: {fname}", flush=True)
                continue
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"[job:{job_id}] Unsupported ext {ext!r} — skipping", flush=True)
                continue

            print(f"[job:{job_id}] Processing: {fname}", flush=True)
            try:
                csv_path, gps = detect_and_convert(dest, upload_dir=tmpdir)
            except Exception as exc:
                print(f"[job:{job_id}] Conversion failed for {fname}: {exc}", flush=True)
                continue

            try:
                signals = load_csv(csv_path)
            except Exception as exc:
                print(f"[job:{job_id}] load_csv failed for {fname}: {exc}", flush=True)
                continue

            print(f"[job:{job_id}]   → {signals.shape[0]} signals, running inference…", flush=True)
            preds, confs = run_inference(_model, signals)
            del signals
            if csv_path != dest:
                csv_path.unlink(missing_ok=True)
            gc.collect()

            n         = len(preds)
            n_delam   = int((preds == 0).sum())
            delam_pct = round(n_delam / n * 100, 2) if n else 0.0

            file_preds.append(preds)
            file_confs.append(confs)
            file_names.append(fname)
            per_file_summary.append({
                "filename":  fname,
                "signals":   n,
                "delam_pct": delam_pct,
                "gps":       gps,
            })
            total_sigs += n

        if total_sigs == 0:
            raise ValueError("No valid GPR signals found in uploaded files.")

        # Aggregate
        all_preds       = np.concatenate(file_preds)
        n_del_total     = int((all_preds == 0).sum())
        del all_preds
        delam_pct_total = round(n_del_total / total_sigs * 100, 2)
        sound_pct_total = round(100.0 - delam_pct_total, 2)

        # Render C-scan
        gc.collect()
        try:
            cscan_b64 = render_cscan_b64(file_preds, file_confs, file_names)
            print(f"[job:{job_id}] C-scan rendered ({len(cscan_b64)//1024} KB b64)", flush=True)
        except Exception as exc:
            print(f"[job:{job_id}] C-scan render failed: {exc}", flush=True)
            cscan_b64 = ""

        elapsed = round(time.perf_counter() - t0, 3)

        result = {
            "signals_analyzed":  total_sigs,
            "delamination_pct":  delam_pct_total,
            "sound_pct":         sound_pct_total,
            "analysis_time_sec": elapsed,
            "cscan_image":       cscan_b64,
            "per_file_summary":  per_file_summary,
        }

        # ── Supabase: upload C-scan PNG + write DB row ────────────────────────
        cscan_url: Optional[str] = None
        if _supabase and cscan_b64:
            try:
                import base64
                png_bytes = base64.b64decode(cscan_b64)
                uid = user_id or "anonymous"
                storage_path = f"{uid}/{job_id}.png"
                _supabase.storage.from_("cscan-images").upload(
                    storage_path,
                    png_bytes,
                    {"content-type": "image/png"},
                )
                cscan_url = _supabase.storage.from_("cscan-images").get_public_url(storage_path)
                print(f"[job:{job_id}] C-scan uploaded to storage: {storage_path}", flush=True)
            except Exception as exc:
                print(f"[job:{job_id}] Storage upload failed: {exc}", flush=True)

        if _supabase:
            try:
                row = {
                    "id":               job_id,
                    "user_id":          user_id,
                    "status":           "complete",
                    "signals_analyzed": total_sigs,
                    "delamination_pct": delam_pct_total,
                    "sound_pct":        sound_pct_total,
                    "analysis_time_sec": elapsed,
                    "cscan_url":        cscan_url,
                    "per_file_summary": per_file_summary,
                    "file_names":       file_names,
                    "completed_at":     "now()",
                }
                _supabase.table("analysis_jobs").upsert(row).execute()
                print(f"[job:{job_id}] DB row written", flush=True)
            except Exception as exc:
                print(f"[job:{job_id}] DB write failed: {exc}", flush=True)

        _jobs[job_id].update({
            "status":     "complete",
            "result":     result,
            "cscan_url":  cscan_url,
            "completed_at": time.time(),
        })
        print(f"[job:{job_id}] Done in {elapsed}s", flush=True)

    except Exception as exc:
        print(f"[job:{job_id}] FAILED: {exc}", flush=True)
        err_msg = str(exc)

        if _supabase:
            try:
                _supabase.table("analysis_jobs").upsert({
                    "id":          job_id,
                    "user_id":     user_id,
                    "status":      "failed",
                    "error_msg":   err_msg,
                    "completed_at": "now()",
                }).execute()
            except Exception:
                pass

        _jobs[job_id].update({
            "status":    "failed",
            "error":     err_msg,
            "completed_at": time.time(),
        })

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    loaded = _model is not None
    return {
        "status":       "ok",
        "model_loaded": loaded,
        "model_path":   str(MODEL_PATH),
        "message":      "Ready" if loaded else "Model loading in background, please retry in 30s",
    }


@app.get("/formats")
def formats() -> dict:
    return {
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "companion_extensions":  sorted(COMPANION_EXTENSIONS),
        "formats":               FORMAT_INFO,
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
async def analyze(
    files: list[UploadFile] = File(...),
    user_id: Optional[str]  = Depends(verify_token),
) -> JSONResponse:
    """
    Accepts file uploads, queues a background inference job, returns {job_id}
    immediately so the frontend can poll GET /job/{job_id}.
    """
    print(f"[analyze] {len(files)} file(s), user_id={user_id}", flush=True)

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Read all file bytes while still in the async request context
    file_data: list[tuple[str, bytes]] = []
    total_bytes = 0
    for i, upload in enumerate(files):
        fname   = upload.filename or f"upload_{i}.bin"
        content = await upload.read()
        file_mb = len(content) / 1024 ** 2

        if file_mb > MAX_FILE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"{fname} is {file_mb:.1f} MB — limit is {MAX_FILE_MB} MB.",
            )
        total_bytes += len(content)
        if total_bytes / 1024 ** 2 > MAX_TOTAL_MB:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload exceeds {MAX_TOTAL_MB} MB.",
            )
        file_data.append((fname, content))
        del content

    # Create job + tmpdir
    job_id = str(uuid.uuid4())
    tmpdir = Path(tempfile.mkdtemp(prefix=f"verus_{job_id}_"))

    _jobs[job_id] = {
        "status":     "pending",
        "user_id":    user_id,
        "created_at": time.time(),
    }

    # Write initial DB row so the frontend sees it in history immediately
    if _supabase:
        try:
            _supabase.table("analysis_jobs").insert({
                "id":      job_id,
                "user_id": user_id,
                "status":  "pending",
            }).execute()
        except Exception as exc:
            print(f"[analyze] DB insert failed: {exc}", flush=True)

    # Submit to thread pool (tmpdir cleaned up inside run_analysis_job)
    _executor.submit(run_analysis_job, job_id, file_data, user_id, tmpdir)
    print(f"[analyze] Queued job {job_id}", flush=True)

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/job/{job_id}")
def get_job(
    job_id: str,
    user_id: Optional[str] = Depends(verify_token),
) -> JSONResponse:
    """Poll job status. Returns status + result when complete."""
    job = _jobs.get(job_id)
    if job is None:
        # Fall back to Supabase if job was processed by another instance
        if _supabase:
            try:
                row = _supabase.table("analysis_jobs") \
                    .select("*").eq("id", job_id).single().execute()
                if row.data:
                    return JSONResponse(row.data)
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Job not found.")

    return JSONResponse(job)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
