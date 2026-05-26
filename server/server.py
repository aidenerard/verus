"""
server/server.py
FastAPI app: startup, health/memory/formats endpoints, /analyze + /job polling.

Does NOT: contain model loading logic (see model_loader.py) or job execution
logic (see jobs.py / pipeline.py).
"""

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psutil

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from auth import verify_token
from jobs import _jobs, _executor, run_analysis_job, run_proceq_job
from run import CNN1D, HorizonCNN, DEVICE  # noqa: F401 — re-exported for type hints
from ingest import SUPPORTED_EXTENSIONS, COMPANION_EXTENSIONS, FORMAT_INFO
from model_loader import load_models_background
import interactive

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

_HERE = Path(__file__).parent
MODEL_PATH              = Path(os.environ.get("MODEL_PATH",             _HERE / "model.pth"))
REBAR_MODEL_PATH        = Path(os.environ.get("REBAR_MODEL_PATH",       _HERE / "models" / "horizon_model.pth"))
MODEL_CONFIG_PATH       = Path(os.environ.get("MODEL_CONFIG_PATH",      _HERE / "configs" / "model_config.json"))
REBAR_MODEL_CONFIG_PATH = Path(os.environ.get("REBAR_MODEL_CONFIG_PATH", _HERE / "configs" / "rebar_model_config.json"))
MAX_FILE_MB   = 50
MAX_TOTAL_MB  = 2000

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Verus GPR Inference Server", version="2.0.0")


class ForceCORSMiddleware(BaseHTTPMiddleware):
    """
    Ensures CORS headers are present on EVERY response, including 404s,
    exception responses, and edge errors that CORSMiddleware may miss.
    Short-circuits OPTIONS preflight before routing so it can never 404.
    """
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(headers={
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age":       "86400",
            })
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response


# Order note: Starlette wraps middlewares in reverse-add order, so the LAST
# add_middleware() call becomes the OUTERMOST wrapper. ForceCORSMiddleware
# is added last on purpose — it must wrap the entire app (including
# CORSMiddleware and the exception handler) so that 404/5xx responses still
# carry CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(ForceCORSMiddleware)

_model:        CNN1D         | None = None
_rebar_model:  HorizonCNN    | None = None
_model_config: dict          | None = None


def _do_load() -> None:
    global _model, _rebar_model, _model_config
    m, rm, cfg = load_models_background(
        MODEL_PATH, REBAR_MODEL_PATH,
        MODEL_CONFIG_PATH, REBAR_MODEL_CONFIG_PATH,
    )
    _model, _rebar_model, _model_config = m, rm, cfg


@app.on_event("startup")
def startup_event() -> None:
    _init_supabase()
    interactive.init(lambda: _supabase)
    threading.Thread(target=_do_load, daemon=True).start()
    print("[startup] Server ready — model loading in background.", flush=True)


app.include_router(interactive.router)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    loaded = _model is not None
    return {
        "status":             "ok",
        "model_loaded":       loaded,
        "rebar_model_loaded": _rebar_model is not None,
        "model_path":         str(MODEL_PATH),
        "message":            "Ready" if loaded else "Model loading in background, please retry in 30s",
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


@app.get("/models/status")
def models_status() -> dict:
    """Reports which optional GPR model weights are available on disk."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    return {
        "horizon":   os.path.exists(os.path.join(models_dir, "horizon_model.pth")),
        "thickness": os.path.exists(os.path.join(models_dir, "thickness_model.pth")),
        "corrosion": os.path.exists(os.path.join(models_dir, "corrosion_model.pth")),
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
                "id":         job_id,
                "user_id":    user_id,
                "status":     "pending",
                "project_id": project_id,
            }).execute()
        except Exception as exc:
            print(f"[analyze] DB insert failed: {exc}", flush=True)

    _executor.submit(
        run_analysis_job,
        job_id, file_data, user_id, tmpdir,
        manufacturer, frequency_mhz, project_id, _model, _rebar_model, _model_config, _supabase,
    )
    print(f"[analyze] Queued job {job_id}", flush=True)

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.options("/analyze")
async def options_analyze():
    return {}


@app.get("/job/{job_id}/status")
def get_job_status(
    job_id: str,
    user_id: Optional[str] = Depends(verify_token),
) -> JSONResponse:
    """Lightweight 1-second poll — returns {status, progress, stage, elapsed_seconds, error}."""
    job = _jobs.get(job_id)
    if job:
        return JSONResponse({
            "status":          job.get("status"),
            "progress":        job.get("progress", 0),
            "stage":           job.get("stage", ""),
            "elapsed_seconds": round(time.time() - job.get("created_at", time.time()), 1),
            "error":           job.get("error"),
        })
    if _supabase:
        try:
            row = _supabase.table("analysis_jobs") \
                .select("status,progress,stage,error_msg") \
                .eq("id", job_id).single().execute()
            if row.data:
                d = row.data
                return JSONResponse({
                    "status":          d.get("status", "unknown"),
                    "progress":        d.get("progress", 100 if d.get("status") == "complete" else 0),
                    "stage":           d.get("stage", ""),
                    "elapsed_seconds": 0,
                    "error":           d.get("error_msg"),
                })
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Job not found.")


@app.options("/job/{job_id}/status")
async def options_job_status(job_id: str):
    return {}


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


@app.options("/job/{job_id}")
async def options_job(job_id: str):
    return {}


@app.post("/analyze-proceq")
async def analyze_proceq(
    files:   list[UploadFile] = File(...),
    epsr:    float            = Form(9.0),
    user_id: Optional[str]   = Depends(verify_token),
) -> JSONResponse:
    """Accept Proceq .scan/.pos/.CScan uploads, queue background job, return {job_id}."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    file_data: list[tuple[str, bytes]] = []
    total_bytes = 0
    for i, upload in enumerate(files):
        fname   = upload.filename or f"upload_{i}.bin"
        content = await upload.read()
        total_bytes += len(content)
        if total_bytes / 1024 ** 2 > MAX_TOTAL_MB:
            raise HTTPException(status_code=413, detail=f"Total upload exceeds {MAX_TOTAL_MB} MB.")
        file_data.append((fname, content))
        del content

    job_id = str(uuid.uuid4())
    tmpdir = Path(tempfile.mkdtemp(prefix=f"verus_proceq_{job_id}_"))
    _jobs[job_id] = {"status": "pending", "user_id": user_id, "created_at": time.time()}

    # Insert DB row at submission so polling never 404s before the worker writes.
    # Mirrors the /analyze pattern. Best-effort — DB unavailable still lets the
    # in-memory worker run.
    if _supabase:
        try:
            _supabase.table("analysis_jobs").insert({
                "id":      job_id,
                "user_id": user_id,
                "status":  "pending",
            }).execute()
        except Exception as exc:
            print(f"[analyze-proceq] DB insert failed: {exc}", flush=True)

    _executor.submit(run_proceq_job, job_id, file_data, tmpdir, epsr, user_id, _supabase)
    print(f"[analyze-proceq] Queued job {job_id} ({len(file_data)} files)", flush=True)

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.options("/analyze-proceq")
async def options_analyze_proceq():
    return {}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
