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


def download_file(url: str, dest: str) -> None:
    """Download a file from any URL. Handles Drive confirm tokens;
    treats everything else as a direct download."""
    from urllib.parse import urlparse, parse_qs

    if "drive.google.com" not in url:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(32768):
                if chunk:
                    f.write(chunk)
        return

    session = requests.Session()
    r = session.get(url, stream=True, timeout=60)
    token = next(
        (v for k, v in r.cookies.items() if k.startswith("download_warning")),
        None,
    )
    if token:
        file_id = parse_qs(urlparse(url).query).get("id", [None])[0]
        r = session.get(
            "https://drive.google.com/uc",
            params={"id": file_id, "export": "download", "confirm": token},
            stream=True,
            timeout=60,
        )
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)

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

MODEL_PATH             = Path(os.environ.get("MODEL_PATH",             Path(__file__).parent / "model.pth"))
REBAR_MODEL_PATH       = Path(os.environ.get("REBAR_MODEL_PATH",       Path(__file__).parent / "rebar_model.pth"))
MODEL_CONFIG_PATH      = Path(os.environ.get("MODEL_CONFIG_PATH",      Path(__file__).parent / "model_config.json"))
REBAR_MODEL_CONFIG_PATH = Path(os.environ.get("REBAR_MODEL_CONFIG_PATH", Path(__file__).parent / "rebar_model_config.json"))
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

def _load_model_background() -> None:
    import json
    global _model, _rebar_model, _model_config
    print(f"[startup] Looking for model at: {MODEL_PATH.resolve()}", flush=True)

    if not MODEL_PATH.exists():
        gdrive_url = os.environ.get("MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading model from {gdrive_url} …", flush=True)
            try:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                download_file(gdrive_url, str(MODEL_PATH))
            except Exception as exc:
                print(f"[startup] ERROR: model download failed: {exc}", flush=True)
                return
        else:
            print(f"[startup] WARNING: {MODEL_PATH} missing and MODEL_GDRIVE_URL unset — /analyze returns 503.", flush=True)
            return

    if not MODEL_PATH.exists():
        print(f"[startup] ERROR: Model still missing after download: {MODEL_PATH}", flush=True)
        return

    print(f"[startup] model.pth size: {MODEL_PATH.stat().st_size:,} bytes", flush=True)

    cfg_url = os.environ.get("MODEL_CONFIG_URL")
    if cfg_url and not MODEL_CONFIG_PATH.exists():
        try:
            download_file(cfg_url, str(MODEL_CONFIG_PATH))
        except Exception as exc:
            print(f"[startup] WARNING: model_config.json download failed: {exc}", flush=True)

    if not MODEL_CONFIG_PATH.exists():
        raise RuntimeError(
            f"model_config.json not found at {MODEL_CONFIG_PATH} and MODEL_CONFIG_URL is not set. "
            "Set MODEL_CONFIG_URL to a raw JSON download URL."
        )

    try:
        with open(MODEL_CONFIG_PATH) as f:
            loaded_cfg = json.load(f)
    except json.JSONDecodeError as e:
        with open(MODEL_CONFIG_PATH) as f:
            head = f.read(200)
        raise RuntimeError(
            f"Config from {cfg_url!r} is not valid JSON. First 200 bytes: {head!r}"
        ) from e
    print(f"[startup] Config: {loaded_cfg}", flush=True)

    arch = {k: loaded_cfg[k] for k in ("in_channels", "conv_channels", "head_hidden") if k in loaded_cfg}
    print(f"[startup] Loading arch: {arch}", flush=True)
    try:
        m = CNN1D(**arch).to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        m.eval()
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        _model, _model_config = m, loaded_cfg
        print(f"[startup] Model loaded ({n_p:,} params) with config: {arch}", flush=True)
    except Exception as exc:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        ckpt_shapes = {k: tuple(v.shape) for k, v in list(ckpt.items())[:8]}
        raise RuntimeError(
            f"Model weights incompatible with config {arch}.\n"
            f"  Checkpoint shapes (first 8): {ckpt_shapes}\n"
            f"  Original error: {exc}"
        ) from exc

    # ── Rebar model ───────────────────────────────────────────────────────────
    if not REBAR_MODEL_PATH.exists():
        gdrive_url = os.environ.get("REBAR_MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading rebar model from {gdrive_url} …", flush=True)
            try:
                REBAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                download_file(gdrive_url, str(REBAR_MODEL_PATH))
            except Exception as exc:
                print(f"[startup] WARNING: Rebar model download failed: {exc}", flush=True)
        else:
            print(
                "[startup] REBAR_MODEL_GDRIVE_URL not set — "
                "rebar depth will use physics fallback.",
                flush=True,
            )

    if REBAR_MODEL_PATH.exists():
        rebar_cfg_url = os.environ.get("REBAR_MODEL_CONFIG_URL")
        if rebar_cfg_url and not REBAR_MODEL_CONFIG_PATH.exists():
            try:
                download_file(rebar_cfg_url, str(REBAR_MODEL_CONFIG_PATH))
            except Exception as exc:
                print(f"[startup] WARNING: rebar_model_config.json download failed: {exc}", flush=True)

        rebar_in_ch = 2
        if REBAR_MODEL_CONFIG_PATH.exists():
            try:
                with open(REBAR_MODEL_CONFIG_PATH) as f:
                    rebar_cfg = json.load(f)
                rebar_in_ch = rebar_cfg.get("in_channels", 2)
                print(f"[startup] Rebar config: {rebar_cfg}", flush=True)
            except json.JSONDecodeError as e:
                with open(REBAR_MODEL_CONFIG_PATH) as f:
                    head = f.read(200)
                raise RuntimeError(
                    f"Rebar config from {rebar_cfg_url!r} is not valid JSON. First 200 bytes: {head!r}"
                ) from e

        try:
            rm = RebarDepthCNN(in_channels=rebar_in_ch).to(DEVICE)
            rm.load_state_dict(
                torch.load(REBAR_MODEL_PATH, map_location=DEVICE, weights_only=False)
            )
            rm.eval()
            n_rp = sum(p.numel() for p in rm.parameters() if p.requires_grad)
            _rebar_model = rm
            print(f"[startup] Rebar model loaded ({n_rp:,} params, in_channels={rebar_in_ch})", flush=True)
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
                .select("status,progress,stage").eq("id", job_id).single().execute()
            if row.data:
                return JSONResponse(row.data)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Job not found.")


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
