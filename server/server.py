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
import requests
import torch


def _download_gdrive(url: str, dest: Path) -> None:
    """Download a Google Drive file, handling the virus-scan confirmation page."""
    session = requests.Session()
    response = session.get(url, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(url, params={'confirm': token}, stream=True)

    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    print(f"[download] {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth import verify_token
from jobs import _jobs, _executor, run_analysis_job
from run import CNN1D, RebarDepthCNN, DEVICE
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

MODEL_PATH        = Path(os.environ.get("MODEL_PATH",        Path(__file__).parent / "model.pth"))
REBAR_MODEL_PATH  = Path(os.environ.get("REBAR_MODEL_PATH",  Path(__file__).parent / "rebar_model.pth"))
MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_PATH", Path(__file__).parent / "model_config.json"))
MAX_FILE_MB       = 50
MAX_TOTAL_MB      = 2000

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

_model:        CNN1D         | None = None
_rebar_model:  RebarDepthCNN | None = None
_model_config: dict          | None = None

_FALLBACK_CONFIGS: list[dict] = [
    {"in_channels": 2, "conv_channels": [32, 64, 128], "head_hidden": 64},
    {"in_channels": 1, "conv_channels": [32, 128, 128], "head_hidden": 128},
    {"in_channels": 1, "conv_channels": [32, 64, 128], "head_hidden": 64},
]


def _load_model_background() -> None:
    global _model, _rebar_model, _model_config
    print(f"[startup] Looking for model at: {MODEL_PATH.resolve()}", flush=True)

    if not MODEL_PATH.exists():
        gdrive_url = os.environ.get("MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading model from {gdrive_url} …", flush=True)
            try:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                _download_gdrive(gdrive_url, MODEL_PATH)
            except Exception as exc:
                print(f"[startup] ERROR: model download failed: {exc}", flush=True)
                return
        else:
            print(f"[startup] WARNING: {MODEL_PATH} missing and MODEL_GDRIVE_URL unset — /analyze returns 503.", flush=True)
            return

    if not MODEL_PATH.exists():
        print(f"[startup] ERROR: Model still missing after download: {MODEL_PATH}", flush=True)
        return

    # Try downloading model_config.json if available
    cfg_url = os.environ.get("MODEL_CONFIG_GDRIVE_URL")
    if cfg_url and not MODEL_CONFIG_PATH.exists():
        try:
            _download_gdrive(cfg_url, MODEL_CONFIG_PATH)
        except Exception as exc:
            print(f"[startup] WARNING: model_config.json download failed: {exc}", flush=True)

    loaded_cfg: dict | None = None
    if MODEL_CONFIG_PATH.exists():
        try:
            import json
            loaded_cfg = json.loads(MODEL_CONFIG_PATH.read_text())
        except Exception as exc:
            print(f"[startup] WARNING: model_config.json parse failed: {exc}", flush=True)

    for cfg in ([loaded_cfg] if loaded_cfg else _FALLBACK_CONFIGS):
        arch = {k: cfg[k] for k in ("in_channels", "conv_channels", "head_hidden") if k in cfg}
        try:
            m = CNN1D(**arch).to(DEVICE)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
            m.eval()
            n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            _model, _model_config = m, cfg
            print(f"[startup] Model loaded ({n_p:,} params) with config: {arch}", flush=True)
            break
        except Exception:
            continue
    if _model is None:
        print("[startup] ERROR: all model configs failed to load weights.", flush=True)

    # ── Rebar model ───────────────────────────────────────────────────────────
    if not REBAR_MODEL_PATH.exists():
        gdrive_url = os.environ.get("REBAR_MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading rebar model from {gdrive_url} …", flush=True)
            try:
                REBAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                _download_gdrive(gdrive_url, REBAR_MODEL_PATH)
            except Exception as exc:
                print(f"[startup] WARNING: Rebar model download failed: {exc}", flush=True)
        else:
            print(
                "[startup] REBAR_MODEL_GDRIVE_URL not set — "
                "rebar depth will use physics fallback.",
                flush=True,
            )

    if REBAR_MODEL_PATH.exists():
        try:
            rm = RebarDepthCNN().to(DEVICE)
            rm.load_state_dict(
                torch.load(REBAR_MODEL_PATH, map_location=DEVICE, weights_only=False)
            )
            rm.eval()
            n_rp = sum(p.numel() for p in rm.parameters() if p.requires_grad)
            _rebar_model = rm
            print(f"[startup] Rebar model loaded successfully ({n_rp:,} parameters)", flush=True)
        except Exception as exc:
            print(f"[startup] WARNING: Rebar model load failed: {exc}", flush=True)


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
