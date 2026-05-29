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
from jobs import _jobs, _executor, run_analysis_job, run_proceq_job, run_proceq_zip_job
from run import CNN1D, RebarDepthCNN, DEVICE  # noqa: F401 — re-exported for type hints
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


def _company_from_profile(user_id: Optional[str]) -> str:
    """Read profiles.company for this user. Empty string on any failure or
    when Supabase isn't configured. Used as a fallback when the upload form
    didn't carry an explicit company."""
    if not _supabase or not user_id:
        return ""
    try:
        row = _supabase.table("profiles").select("company").eq("id", user_id).single().execute()
        return (row.data or {}).get("company") or ""
    except Exception:
        return ""


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH              = Path(os.environ.get("MODEL_PATH",             Path(__file__).parent / "model.pth"))
REBAR_MODEL_PATH        = Path(os.environ.get("REBAR_MODEL_PATH",       Path(__file__).parent / "rebar_model.pth"))
MODEL_CONFIG_PATH       = Path(os.environ.get("MODEL_CONFIG_PATH",      Path(__file__).parent / "model_config.json"))
REBAR_MODEL_CONFIG_PATH = Path(os.environ.get("REBAR_MODEL_CONFIG_PATH", Path(__file__).parent / "rebar_model_config.json"))
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
_rebar_model:  RebarDepthCNN | None = None
_model_config: dict          | None = None


def _do_load() -> None:
    global _model, _rebar_model, _model_config
    m, rm, cfg = load_models_background(
        MODEL_PATH, REBAR_MODEL_PATH,
        MODEL_CONFIG_PATH, REBAR_MODEL_CONFIG_PATH,
    )
    _model, _rebar_model, _model_config = m, rm, cfg


def _download_horizon_model_if_missing() -> bool:
    """
    Pull horizon_model.pth from Supabase public storage if it's not on disk.
    Sync + requests (already in requirements). Best-effort — server still
    starts on failure; ensemble inference falls back to signal processing.

    Assumes a public bucket named 'models' with the file uploaded at the
    expected path: $SUPABASE_URL/storage/v1/object/public/models/horizon_model.pth
    """
    model_path = Path(__file__).parent / "models" / "horizon_model.pth"
    if model_path.exists():
        print(f"[startup] horizon_model.pth already present "
              f"({model_path.stat().st_size:,} bytes)", flush=True)
        return True

    supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_url:
        print("[startup] SUPABASE_URL not set — skipping model download", flush=True)
        return False

    url = f"{supabase_url}/storage/v1/object/public/models/horizon_model.pth"
    print(f"[startup] Downloading horizon_model.pth from Supabase public storage…", flush=True)
    try:
        import requests
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f"[startup] Model download failed: HTTP {r.status_code}", flush=True)
            return False
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(r.content)
        print(f"[startup] horizon_model.pth downloaded "
              f"({len(r.content):,} bytes)", flush=True)
        return True
    except Exception as exc:
        print(f"[startup] Model download error: {exc}", flush=True)
        return False


@app.on_event("startup")
def startup_event() -> None:
    _download_horizon_model_if_missing()
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
    files:          list[UploadFile] = File(...),
    manufacturer:   Optional[str]    = Form(None),
    frequency_mhz:  int              = Form(1600),
    project_id:     Optional[str]    = Form(None),
    analysis_name:  str              = Form("Untitled Analysis"),
    analysis_notes: str              = Form(""),
    company:        str              = Form(""),
    project:        str              = Form(""),
    user_id:        Optional[str]    = Depends(verify_token),
) -> JSONResponse:
    """Accept uploads, queue a background job, return {job_id} immediately."""
    name_clean    = (analysis_name or "").strip() or "Untitled Analysis"
    notes_clean   = (analysis_notes or "").strip()
    company_clean = (company or "").strip() or _company_from_profile(user_id)
    project_clean = (project or "").strip()
    print(f"[analyze] {len(files)} file(s), manufacturer={manufacturer!r}, "
          f"freq={frequency_mhz} MHz, user_id={user_id}, name={name_clean!r}, "
          f"company={company_clean!r}, project={project_clean!r}", flush=True)

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
                "id":             job_id,
                "user_id":        user_id,
                "status":         "pending",
                "project_id":     project_id,
                "analysis_name":  name_clean,
                "analysis_notes": notes_clean,
                "company":        company_clean,
                "project":        project_clean,
            }).execute()
        except Exception as exc:
            print(f"[analyze] DB insert failed: {exc}", flush=True)

    _executor.submit(
        run_analysis_job,
        job_id, file_data, user_id, tmpdir,
        manufacturer, frequency_mhz, project_id, _model, _rebar_model, _model_config, _supabase,
        "Bridge Deck", 1.0, company_clean, project_clean, name_clean,
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
    files:          list[UploadFile] = File(...),
    epsr:           float            = Form(9.0),
    analysis_name:  str              = Form("Untitled Analysis"),
    analysis_notes: str              = Form(""),
    company:        str              = Form(""),
    project:        str              = Form(""),
    user_id:        Optional[str]   = Depends(verify_token),
) -> JSONResponse:
    """Accept Proceq .scan/.pos/.CScan uploads, queue background job, return {job_id}."""
    name_clean    = (analysis_name or "").strip() or "Untitled Analysis"
    notes_clean   = (analysis_notes or "").strip()
    company_clean = (company or "").strip() or _company_from_profile(user_id)
    project_clean = (project or "").strip()
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

    is_zip = len(file_data) == 1 and file_data[0][0].lower().endswith(".zip")
    initial_stage = "Extracting zip" if is_zip else "Queued"

    job_id = str(uuid.uuid4())
    tmpdir = Path(tempfile.mkdtemp(prefix=f"verus_proceq_{job_id}_"))
    _jobs[job_id] = {
        "status":     "pending",
        "progress":   0,
        "stage":      initial_stage,
        "user_id":    user_id,
        "created_at": time.time(),
    }

    # Insert DB row at submission so polling never 404s before the worker writes.
    # Mirrors the /analyze pattern. Best-effort — DB unavailable still lets the
    # in-memory worker run.
    if _supabase:
        try:
            _supabase.table("analysis_jobs").insert({
                "id":             job_id,
                "user_id":        user_id,
                "status":         "pending",
                "progress":       0,
                "stage":          initial_stage,
                "analysis_name":  name_clean,
                "analysis_notes": notes_clean,
                "company":        company_clean,
                "project":        project_clean,
            }).execute()
        except Exception as exc:
            print(f"[analyze-proceq] DB insert failed: {exc}", flush=True)

    _executor.submit(
        run_proceq_job, job_id, file_data, tmpdir, epsr, user_id, _supabase,
        company_clean, project_clean, name_clean,
    )
    print(f"[analyze-proceq] Queued job {job_id} ({len(file_data)} files, zip={is_zip})", flush=True)

    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.options("/analyze-proceq")
async def options_analyze_proceq():
    return {}


@app.post("/analyze-proceq-zip")
async def analyze_proceq_zip(
    storage_path:   str   = Form(...),
    epsr:           float = Form(9.0),
    analysis_name:  str   = Form("Untitled Analysis"),
    analysis_notes: str   = Form(""),
    company:        str   = Form(""),
    project:        str   = Form(""),
    user_id:        Optional[str] = Depends(verify_token),
) -> JSONResponse:
    """Process a Proceq zip already uploaded to the 'uploads' Supabase bucket.
    Frontend uploads directly to storage to bypass Render's body size limit."""
    name_clean    = (analysis_name or "").strip() or "Untitled Analysis"
    notes_clean   = (analysis_notes or "").strip()
    company_clean = (company or "").strip() or _company_from_profile(user_id)
    project_clean = (project or "").strip()

    job_id = str(uuid.uuid4())
    print(f"[analyze-proceq-zip] {job_id} path={storage_path!r} user={user_id}", flush=True)

    if _supabase:
        try:
            _supabase.table("analysis_jobs").insert({
                "id":             job_id,
                "user_id":        user_id,
                "status":         "pending",
                "progress":       0,
                "stage":          "Downloading zip from storage",
                "analysis_name":  name_clean,
                "analysis_notes": notes_clean,
                "company":        company_clean,
                "project":        project_clean,
            }).execute()
        except Exception as exc:
            print(f"[analyze-proceq-zip] DB insert failed: {exc}", flush=True)

    _jobs[job_id] = {
        "status":     "pending",
        "progress":   0,
        "stage":      "Downloading zip from storage",
        "user_id":    user_id,
        "created_at": time.time(),
    }

    _executor.submit(
        run_proceq_zip_job, job_id, storage_path, epsr, user_id, _supabase,
        company_clean, project_clean, name_clean,
    )
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.options("/analyze-proceq-zip")
async def options_analyze_proceq_zip():
    return {}


# ── Interactive picks API ────────────────────────────────────────────────────
# Schema notes:
#   The picks table (migration 008) uses `trace_index` (not `trace_idx`) and
#   `is_edited` (not `is_manual`). The frontend JSON shape uses `trace_idx` /
#   `is_manual` per Anthropic's spec, so we translate at the API boundary.

def _picks_db_row(p: dict, job_id: str, user_id: Optional[str]) -> dict:
    """Translate frontend pick JSON → picks-table column shape. Writes the
    existing-schema columns AND the new ones from the pending migration
    (sample_idx / swath_idx / is_manual); insert fails gracefully via the
    surrounding try if the migration hasn't been applied yet."""
    swath_idx = int(p.get("swath_idx", 0))
    is_manual = bool(p.get("is_manual") or p.get("is_edited", False))
    return {
        "job_id":       job_id,
        "scan_line_id": str(swath_idx),
        "trace_index":  int(p.get("trace_idx") or p.get("trace_index") or 0),
        "sample_idx":   int(p.get("sample_idx", 0)),
        "depth_in":     float(p.get("depth_in", 0.0)),
        "amplitude":    float(p.get("amplitude", 0.0)),
        "confidence":   float(p.get("confidence", 1.0)),
        "is_edited":    is_manual,
        "is_manual":    is_manual,
        "swath_idx":    swath_idx,
    }


def _pick_to_frontend(row: dict) -> dict:
    """DB row → frontend JSON shape. Frontend uses trace_idx / swath_idx /
    is_manual; DB schema uses trace_index / scan_line_id / is_edited."""
    return {
        "id":         row.get("id"),
        "trace_idx":  row.get("trace_index", 0),
        "sample_idx": row.get("sample_idx", 0),
        "depth_in":   row.get("depth_in", 0.0),
        "confidence": row.get("confidence", 1.0),
        "is_manual":  bool(row.get("is_manual", row.get("is_edited", False))),
        "swath_idx":  int(row.get("swath_idx",
                                  int(row.get("scan_line_id", "0") or 0))),
    }


@app.get("/job/{job_id}/picks")
def get_picks(job_id: str, user_id: Optional[str] = Depends(verify_token)) -> JSONResponse:
    """Return all picks for a job (auto + analyst-edited)."""
    if not _supabase:
        raise HTTPException(503, "Database unavailable")
    try:
        res = _supabase.table("picks").select("*").eq("job_id", job_id) \
            .order("trace_index").execute()
        return JSONResponse({
            "picks": [_pick_to_frontend(r) for r in (res.data or [])],
        })
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/job/{job_id}/picks")
async def save_picks(
    job_id: str, request: Request,
    user_id: Optional[str] = Depends(verify_token),
) -> JSONResponse:
    """Replace all picks for this job with the posted set."""
    if not _supabase:
        raise HTTPException(503, "Database unavailable")
    body  = await request.json()
    picks = body.get("picks", [])
    try:
        _supabase.table("picks").delete().eq("job_id", job_id).execute()
        if picks:
            _supabase.table("picks").insert(
                [_picks_db_row(p, job_id, user_id) for p in picks]
            ).execute()
        return JSONResponse({"saved": len(picks)})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/job/{job_id}/regenerate")
def regenerate_depth_map(
    job_id: str, user_id: Optional[str] = Depends(verify_token),
) -> JSONResponse:
    """
    Regenerate the rebar depth map from current picks. Reconstructs a 2D
    (swath_idx × trace_index) grid from the picks rows and feeds it into
    build_unified_depth_map — the unified renderer requires either 2D
    input or 1D + GPS coords, never bare 1D depths.
    """
    if not _supabase:
        raise HTTPException(503, "Database unavailable")

    try:
        picks_res = _supabase.table("picks") \
            .select("trace_index, depth_in, scan_line_id") \
            .eq("job_id", job_id).execute()
        picks = picks_res.data or []
        if not picks:
            raise HTTPException(400, "No picks found for this job")

        job_res = _supabase.table("analysis_jobs") \
            .select("result, analysis_name").eq("id", job_id).single().execute()
        analysis_name = (job_res.data or {}).get("analysis_name") or "Analysis"

        # Reconstruct sparse 2D grid from picks: rows = swath (scan_line_id),
        # cols = trace_index. Linear-interpolate within each row to fill
        # gaps so contourf can run.
        from collections import defaultdict
        import numpy as np
        by_swath: dict[str, list[dict]] = defaultdict(list)
        for p in picks:
            by_swath[str(p.get("scan_line_id", "0"))].append(p)
        swath_ids = sorted(by_swath.keys())
        max_trace = max(int(p["trace_index"]) for p in picks)
        grid = np.full((len(swath_ids), max_trace + 1), np.nan, dtype=np.float32)
        for r, sid in enumerate(swath_ids):
            for p in by_swath[sid]:
                grid[r, int(p["trace_index"])] = float(p["depth_in"])
        for r in range(grid.shape[0]):
            row = grid[r]
            mask = ~np.isnan(row)
            if mask.sum() >= 2:
                grid[r] = np.interp(np.arange(grid.shape[1]), np.where(mask)[0], row[mask])

        import tempfile, base64
        from analysis import build_unified_depth_map, safe_filename
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, f"{safe_filename(analysis_name)}_rebar_depth.png")
            build_unified_depth_map(
                depths_in=grid, output_path=out_path, analysis_name=analysis_name,
            )
            with open(out_path, "rb") as f:
                map_b64 = base64.b64encode(f.read()).decode()

        prev_result = (job_res.data or {}).get("result") or {}
        _supabase.table("analysis_jobs").upsert({
            "id":     job_id,
            "result": {**prev_result,
                       "rebar_depth_map":   map_b64,
                       "picks_count":       len(picks),
                       "analyst_approved":  True},
        }).execute()
        return JSONResponse({
            "regenerated": True,
            "picks_used":  len(picks),
            "rebar_depth_map": map_b64,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.options("/job/{job_id}/picks")
async def options_picks(job_id: str):
    return {}


@app.options("/job/{job_id}/regenerate")
async def options_regenerate(job_id: str):
    return {}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
