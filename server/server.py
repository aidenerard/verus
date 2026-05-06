"""
server/server.py
FastAPI app: startup, health/memory/formats endpoints, /analyze + /job polling.
"""

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psutil
import torch

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import verify_token
from jobs import _jobs, _executor, run_analysis_job
from run import CNN1D, DEVICE
from ingest import SUPPORTED_EXTENSIONS, COMPANION_EXTENSIONS, FORMAT_INFO

# ── Supabase client (optional) ────────────────────────────────────────────────

_supabase = None


def _init_supabase() -> None:
    global _supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
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


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH   = Path(os.environ.get("MODEL_PATH", Path(__file__).parent / "model.pth"))
MAX_FILE_MB  = 50
MAX_TOTAL_MB = 2000

# ── App ───────────────────────────────────────────────────────────────────────

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
    threading.Thread(target=_load_model_background, daemon=True).start()
    print("[startup] Server ready — model loading in background.", flush=True)


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


@app.get("/memory")
def memory() -> dict:
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb":  round(vm.used      / 1024 ** 2, 1),
        "ram_total_mb": round(vm.total     / 1024 ** 2, 1),
        "ram_percent":  vm.percent,
        "ram_free_mb":  round(vm.available / 1024 ** 2, 1),
    }


@app.get("/formats")
def formats() -> dict:
    return {
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "companion_extensions":  sorted(COMPANION_EXTENSIONS),
        "formats":               FORMAT_INFO,
    }


@app.post("/analyze")
async def analyze(
    files:         list[UploadFile] = File(...),
    manufacturer:  Optional[str]    = Form(None),
    frequency_mhz: int              = Form(1600),
    project_id:    Optional[str]    = Form(None),
    user_id:       Optional[str]    = Depends(verify_token),
) -> JSONResponse:
    """Accept uploads, queue a background job, return {job_id} immediately."""
    print(f"[analyze] {len(files)} file(s), manufacturer={manufacturer!r}, "
          f"freq={frequency_mhz} MHz, user_id={user_id}", flush=True)

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

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

    job_id = str(uuid.uuid4())
    tmpdir = Path(tempfile.mkdtemp(prefix=f"verus_{job_id}_"))

    _jobs[job_id] = {
        "status":     "pending",
        "user_id":    user_id,
        "created_at": time.time(),
    }

    if _supabase:
        try:
            _supabase.table("analysis_jobs").insert({
                "id":      job_id,
                "user_id": user_id,
                "status":  "pending",
            }).execute()
        except Exception as exc:
            print(f"[analyze] DB insert failed: {exc}", flush=True)

    _executor.submit(
        run_analysis_job,
        job_id, file_data, user_id, tmpdir,
        manufacturer, frequency_mhz, _model, _supabase,
    )
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
