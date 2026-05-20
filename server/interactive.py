"""
server/interactive.py
Routes for the Interactive analysis view: pick inspection/editing, regridding,
per-scan-line B-scan retrieval.

The /reprocess and /scene endpoints are deferred to phase 2 and return 501 —
reprocess requires raw DZT persistence to Supabase storage, scene requires
a surface elevation mesh that the current pipeline doesn't produce.
"""
from __future__ import annotations

import base64 as _b64
import io
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from auth import verify_token
from gridding import GRID_ALGORITHMS, run_gridding
from processing_state import settings_hash


router = APIRouter()

# Set by server.py at startup so route handlers can reach the supabase client
# without a circular import.
_supabase_getter: Callable[[], object] | None = None

# FIFO cache for regrid responses, keyed by (job_id, settings_hash). Capped to
# bound memory on Render's 512 MB free tier.
_GRID_CACHE_MAX = 32
_grid_cache: OrderedDict[str, dict] = OrderedDict()


def init(supabase_getter: Callable[[], object]) -> None:
    """Wire the supabase client getter from server.py."""
    global _supabase_getter
    _supabase_getter = supabase_getter


def _sb():
    if _supabase_getter is None:
        raise HTTPException(503, "Database unavailable: server not fully initialized.")
    sb = _supabase_getter()
    if sb is None:
        raise HTTPException(503, "Database unavailable: SUPABASE_* env vars not set.")
    return sb


# ── Picks: read + edit ────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/picks")
def get_picks(
    job_id: str,
    bbox: Optional[str] = Query(None, description="x1,y1,x2,y2 in feet"),
    include_deleted: bool = Query(False),
    user_id: Optional[str] = Depends(verify_token),
):
    sb = _sb()
    q = sb.table("picks").select("*").eq("job_id", job_id)
    if not include_deleted:
        q = q.eq("is_deleted", False)
    if bbox:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox.split(","))
        except Exception:
            raise HTTPException(400, f"Invalid bbox: {bbox!r}. Expected x1,y1,x2,y2.")
        q = q.gte("x_ft", x1).lte("x_ft", x2).gte("y_ft", y1).lte("y_ft", y2)
    rows = q.execute().data or []
    return JSONResponse(rows)


@router.patch("/picks/{pick_id}")
def patch_pick(
    pick_id: str,
    update: dict = Body(...),
    user_id: Optional[str] = Depends(verify_token),
):
    """Edit depth_in, x_ft, y_ft, or is_deleted on a single pick.
    Sets is_edited=True and flags the parent job for regrid."""
    sb = _sb()
    allowed = {"depth_in", "x_ft", "y_ft", "is_deleted"}
    patch = {k: v for k, v in update.items() if k in allowed}
    if not patch:
        raise HTTPException(400, f"No editable fields in body. Allowed: {sorted(allowed)}")
    patch["is_edited"] = True

    result = sb.table("picks").update(patch).eq("id", pick_id).execute()
    if not result.data:
        raise HTTPException(404, "Pick not found.")
    pick = result.data[0]

    job_id = pick["job_id"]
    try:
        row = sb.table("analysis_jobs").select("processing_state") \
            .eq("id", job_id).single().execute()
        state = (row.data or {}).get("processing_state") or {}
        state["needs_regrid"] = True
        sb.table("analysis_jobs").update({"processing_state": state}) \
            .eq("id", job_id).execute()
    except Exception as exc:
        print(f"[interactive] needs_regrid flag failed for {job_id}: {exc}", flush=True)

    return JSONResponse(pick)


# ── Regrid ────────────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/regrid")
def regrid(
    job_id: str,
    settings: dict = Body(...),
    user_id: Optional[str] = Depends(verify_token),
):
    sb = _sb()
    algorithm = settings.get("algorithm", "nearest")
    if algorithm not in GRID_ALGORITHMS:
        raise HTTPException(400, f"Unknown algorithm: {algorithm!r}. "
                                 f"Supported: {sorted(GRID_ALGORITHMS)}")

    cache_key = f"{job_id}:{settings_hash(settings)}"
    if cache_key in _grid_cache:
        _grid_cache.move_to_end(cache_key)
        return JSONResponse(_grid_cache[cache_key])

    rows = sb.table("picks").select("x_ft,y_ft,depth_in") \
        .eq("job_id", job_id).eq("is_deleted", False).execute().data or []
    if len(rows) < 3:
        raise HTTPException(400, f"Need at least 3 picks for gridding; have {len(rows)}.")

    xs = np.array([r["x_ft"]    for r in rows], dtype=np.float64)
    ys = np.array([r["y_ft"]    for r in rows], dtype=np.float64)
    zs = np.array([r["depth_in"] for r in rows], dtype=np.float64)

    try:
        result = run_gridding(
            algorithm, xs, ys, zs,
            cell_size_ft=     settings.get("cell_size_ft", 0.5),
            search_radius_ft= settings.get("search_radius_ft", 10.0),
            edge_clip=        settings.get("edge_clip", True),
            anisotropy_angle= settings.get("anisotropy_angle", 0.0),
            anisotropy_ratio= settings.get("anisotropy_ratio", 1.0),
            edge_polygon=     settings.get("edge_polygon"),
        )
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        print(f"[regrid] {job_id} algorithm={algorithm} failed: {exc}", flush=True)
        raise HTTPException(500, f"Gridding failed: {exc}")

    png_b64 = _render_grid_png(result["grid"], result["extent"])
    png_url = _upload_grid_png(sb, png_b64, user_id, job_id, algorithm)

    grid = result["grid"]
    response = {
        "algorithm":    algorithm,
        "extent":       list(result["extent"]),
        "cell_size_ft": result["cell_size_ft"],
        "grid_data":    np.where(np.isnan(grid), None, grid).tolist(),
        "png_url":      png_url,
        "stats": {
            "n_picks":     int(len(rows)),
            "depth_mean":  float(np.nanmean(grid)) if np.isfinite(np.nanmean(grid)) else 0.0,
            "depth_min":   float(np.nanmin(grid))  if np.isfinite(np.nanmin(grid))  else 0.0,
            "depth_max":   float(np.nanmax(grid))  if np.isfinite(np.nanmax(grid))  else 0.0,
        },
    }

    _grid_cache[cache_key] = response
    if len(_grid_cache) > _GRID_CACHE_MAX:
        _grid_cache.popitem(last=False)

    try:
        row   = sb.table("analysis_jobs").select("processing_state") \
            .eq("id", job_id).single().execute()
        state = (row.data or {}).get("processing_state") or {}
        state["gridding"]     = settings
        state["needs_regrid"] = False
        sb.table("analysis_jobs").update({"processing_state": state}) \
            .eq("id", job_id).execute()
    except Exception as exc:
        print(f"[regrid] processing_state update failed for {job_id}: {exc}", flush=True)

    return JSONResponse(response)


def _render_grid_png(grid: np.ndarray, extent: tuple) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min, x_max, y_min, y_max = extent
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="upper",
                   extent=[x_min, x_max, y_max, y_min])
    plt.colorbar(im, ax=ax, label="Rebar depth (in)")
    ax.set_xlabel("Along-track (ft)")
    ax.set_ylabel("Cross-track (ft)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return _b64.b64encode(buf.getvalue()).decode()


def _upload_grid_png(sb, b64: str, user_id: Optional[str],
                     job_id: str, algorithm: str) -> Optional[str]:
    if not sb or not b64:
        return None
    try:
        uid  = user_id or "anonymous"
        path = f"{uid}/{job_id}_regrid_{algorithm}.png"
        sb.storage.from_("cscan-images").upload(
            path, _b64.b64decode(b64),
            {"content-type": "image/png", "upsert": "true"},
        )
        return sb.storage.from_("cscan-images").get_public_url(path)
    except Exception as exc:
        print(f"[regrid] PNG upload failed for {job_id}: {exc}", flush=True)
        return None


# ── Scan line B-scan retrieval ───────────────────────────────────────────────

@router.get("/jobs/{job_id}/scan_line/{scan_line_id}")
def get_scan_line(
    job_id: str,
    scan_line_id: str,
    user_id: Optional[str] = Depends(verify_token),
):
    """Return the stored B-scan blob plus picks for one scan line.

    TODO (phase 2): re-apply processing_state.filters to the raw trace data
    here, instead of returning the pre-rendered blob from per_file_summary.
    """
    sb  = _sb()
    row = sb.table("analysis_jobs").select("per_file_summary,processing_state") \
        .eq("id", job_id).single().execute()
    if not row.data:
        raise HTTPException(404, f"Job {job_id} not found.")

    per_file = row.data.get("per_file_summary") or []
    match    = next((f for f in per_file if f.get("filename") == scan_line_id), None)
    if not match:
        raise HTTPException(404, f"Scan line {scan_line_id!r} not in job {job_id}.")

    picks = sb.table("picks").select("*") \
        .eq("job_id", job_id).eq("scan_line_id", scan_line_id) \
        .eq("is_deleted", False).execute().data or []

    bscan = match.get("bscan") or {}
    return JSONResponse({
        "scan_line_id":      scan_line_id,
        "bscan_data_b64":    bscan.get("data"),
        "bscan_n_traces":    bscan.get("n_traces", 0),
        "bscan_n_samples":   bscan.get("n_samples", 0),
        "rebar_depth_array": match.get("rebar_depth_array", []),
        "twt_array":         match.get("twt_array", []),
        "peak_sample_array": match.get("peak_sample_array", []),
        "picks":             picks,
        "gps":               match.get("gps"),
    })


# ── Phase 2 stubs ─────────────────────────────────────────────────────────────

@router.post("/jobs/{job_id}/reprocess")
def reprocess_stub(job_id: str, settings: dict = Body(...)):
    """Phase 2: re-run inference with new time-zero shifts / filters / gps_latency.
    Requires raw DZT persistence to Supabase storage on /analyze ingest, which
    is not yet wired."""
    raise HTTPException(
        501,
        "Reprocess endpoint deferred to phase 2. "
        "Will require raw DZT persistence to Supabase storage on /analyze.",
    )


@router.get("/jobs/{job_id}/scene")
def scene_stub(job_id: str):
    """Phase 2: 3D scene payload (surface mesh + 3D pick positions + scan line polylines).
    Requires a surface elevation grid which the current pipeline doesn't produce."""
    raise HTTPException(
        501,
        "Scene endpoint deferred to phase 2. "
        "Will return {surface, picks_3d, scan_lines_3d} once elevation grid is derived.",
    )
