"""
server/jobs.py
Background job orchestrator: in-memory job store, progress helpers, image upload,
and the top-level run_analysis_job function that sequences pipeline calls.

Does NOT: contain inference logic (see pipeline.py) or model loading (see model_loader.py).
"""

import base64 as _b64
import os
import re
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Extensions extracted from an uploaded zip. Anything else (notably macOS
# metadata under __MACOSX/ and ._* AppleDouble files) is skipped to keep
# tmpdir clean and avoid confusing the swath grouper.
_ZIP_ALLOWED_EXTS = {".scan", ".pos", ".cscan", ".dzt", ".dt1"}


def _safe_segment(name: str, fallback: str = "unknown") -> str:
    """Lowercase + slugify to [a-z0-9-], 40-char max. Used to build readable
    Supabase Storage paths from company / project names."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower())
    s = re.sub(r"^-+|-+$", "", s)
    return (s or fallback)[:40]

from pipeline import process_files, build_result_payload
from picks_writer import persist_picks_for_job, estimate_along_track_ft_per_col
from processing_state import DEFAULT_PROCESSING_STATE

# ── Job state ─────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# Columns on analysis_jobs that _update_job is allowed to write. `error` maps
# to the `error_msg` schema column.
_DB_PERSIST_COLS = ("status", "progress", "stage", "error", "completed_at")
_DB_COL_RENAME   = {"error": "error_msg"}


def _update_job(job_id: str, updates: dict, sb=None) -> None:
    """
    Update job state in both the in-memory _jobs dict and Supabase.
    Only allowlisted polling fields (status/progress/stage/error/completed_at)
    flow to Supabase — heavier blobs go through the explicit upsert at job end.
    Best-effort: DB failure is logged and ignored so the job continues.
    """
    if job_id in _jobs:
        _jobs[job_id].update(updates)
    if not sb:
        return
    db_fields = {
        _DB_COL_RENAME.get(k, k): v
        for k, v in updates.items()
        if k in _DB_PERSIST_COLS
    }
    if not db_fields:
        return
    try:
        sb.table("analysis_jobs").upsert({"id": job_id, **db_fields}).execute()
    except Exception as exc:
        print(f"[JOBS:{job_id}] Supabase update failed (non-fatal): {exc}", flush=True)


def _set_progress(job_id: str, progress: int, stage: str, sb) -> None:
    _jobs[job_id]['progress'] = progress
    _jobs[job_id]['stage']    = stage
    if sb:
        try:
            sb.table("analysis_jobs").update(
                {"progress": progress, "stage": stage}
            ).eq("id", job_id).execute()
        except Exception as exc:
            print(f"[job:{job_id}] progress {progress}% update failed: {exc}", flush=True)


def _upload_png(sb, b64: str, company: str, project: str, job_id: str, suffix: str) -> Optional[str]:
    """Upload a base64 PNG into cscan-images at {company}/{project}/{job_id}{suffix}.png."""
    if not sb or not b64:
        return None
    try:
        path = f"{_safe_segment(company)}/{_safe_segment(project)}/{job_id}{suffix}.png"
        sb.storage.from_("cscan-images").upload(path, _b64.b64decode(b64), {"content-type": "image/png"})
        return sb.storage.from_("cscan-images").get_public_url(path)
    except Exception as exc:
        print(f"[upload] {suffix or 'cscan'} failed: {exc}", flush=True)
        return None


# ── Worker ────────────────────────────────────────────────────────────────────

def run_analysis_job(
    job_id: str,
    file_data: list[tuple[str, bytes]],
    user_id: Optional[str],
    tmpdir: Path,
    manufacturer: Optional[str],
    frequency_mhz: int,
    project_id: Optional[str],
    model,
    rebar_model,
    model_config: Optional[dict],
    supabase_client,
    structure_name: str = "Bridge Deck",
    swath_spacing_ft: float = 1.0,
    company: str = "",
    project: str = "",
) -> None:
    """
    Runs in a ThreadPoolExecutor worker thread.
    Orchestrates pipeline, uploads images, writes results to Supabase, updates _jobs.
    """
    _jobs[job_id]["status"]   = "processing"
    _jobs[job_id]["progress"] = 0
    _jobs[job_id]["stage"]    = "Starting"
    t0 = time.perf_counter()

    try:
        # Wait for model (up to 120 s on cold start)
        if model is None:
            print(f"[job:{job_id}] Model not yet loaded — waiting up to 120s…", flush=True)
            deadline = time.time() + 120
            while model is None and time.time() < deadline:
                time.sleep(2)
            if model is None:
                raise RuntimeError("Model failed to load within 120 seconds.")
            print(f"[job:{job_id}] Model ready, proceeding.", flush=True)

        # Save uploaded bytes to tmpdir
        saved_paths: list[tuple[str, Path]] = []
        for fname, content in file_data:
            dest = tmpdir / fname
            dest.write_bytes(content)
            saved_paths.append((fname, dest))
            print(f"[job:{job_id}] Saved {fname} ({len(content)/1024:.1f} KB)", flush=True)

        _set_progress(job_id, 5, "Ingesting", supabase_client)

        def _prog(progress: int, stage: str) -> None:
            _set_progress(job_id, progress, stage, supabase_client)

        def _prog_fast(progress: int, stage: str) -> None:
            _jobs[job_id]['progress'] = progress
            _jobs[job_id]['stage']    = stage

        (
            file_preds, file_confs, file_names,
            file_peak_idxs, file_peak_amps,
            rebar_depth_arrs, rebar_twt_arrs, rebar_peak_arrs,
            per_file_summary, total_sigs,
        ) = process_files(
            job_id, saved_paths, manufacturer,
            model, rebar_model, model_config, frequency_mhz,
            _prog, _prog_fast,
        )

        if total_sigs == 0:
            raise ValueError("No valid GPR signals found in uploaded files.")

        _set_progress(job_id, 80, "Building grids", supabase_client)

        result, cscan_b64, rebar_cscan_b64, amp_b64 = build_result_payload(
            job_id,
            file_preds, file_confs, file_names,
            file_peak_idxs, file_peak_amps,
            rebar_depth_arrs, rebar_twt_arrs, rebar_peak_arrs,
            per_file_summary, total_sigs,
            frequency_mhz, manufacturer, rebar_model,
            swath_spacing_ft, structure_name,
        )

        elapsed = round(time.perf_counter() - t0, 3)
        result["analysis_time_sec"] = elapsed

        # Persist per-trace rebar picks for the Interactive view. Best-effort —
        # a write failure here must not fail the whole job.
        try:
            along_track_ft_per_col = estimate_along_track_ft_per_col(
                per_file_summary,
                grid_cols=result.get("twt_grid_cols") or len(file_names) or 1,
            )
            n_picks = persist_picks_for_job(
                supabase_client, job_id,
                file_names, rebar_depth_arrs, file_peak_amps, file_confs,
                per_file_summary, swath_spacing_ft, along_track_ft_per_col,
            )
            print(f"[job:{job_id}] persisted {n_picks} picks", flush=True)
        except Exception as exc:
            print(f"[job:{job_id}] picks persistence failed: {exc}", flush=True)

        _set_progress(job_id, 95, "Finalizing", supabase_client)
        cscan_url       = _upload_png(supabase_client, cscan_b64,       company, project, job_id, "")
        rebar_cscan_url = _upload_png(supabase_client, rebar_cscan_b64, company, project, job_id, "_rebar")
        amplitude_url   = _upload_png(supabase_client, amp_b64,         company, project, job_id, "_amplitude")

        if supabase_client:
            try:
                supabase_client.table("analysis_jobs").upsert({
                    "id": job_id, "user_id": user_id, "status": "complete", "completed_at": "now()",
                    "signals_analyzed": total_sigs,
                    "delamination_pct": result["delamination_pct"],
                    "sound_pct": result["sound_pct"],
                    "analysis_time_sec": elapsed,
                    "file_names": file_names, "per_file_summary": per_file_summary,
                    "cscan_url": cscan_url, "rebar_cscan_image_url": rebar_cscan_url,
                    "amplitude_image_url": amplitude_url,
                    "manufacturer": manufacturer, "frequency_mhz": frequency_mhz,
                    "rebar_model_used": rebar_model is not None,
                    "prob_grid": result["prob_grid"],
                    "prob_grid_rows": result["prob_grid_rows"],
                    "prob_grid_cols": result["prob_grid_cols"],
                    "otsu_threshold": result["otsu_threshold"],
                    "twt_grid": result["twt_grid"],
                    "twt_grid_rows": result["twt_grid_rows"],
                    "twt_grid_cols": result["twt_grid_cols"],
                    "model_confidence_pct": result["model_confidence_pct"],
                    "depth_accuracy_in": result["depth_accuracy_in"],
                    "signal_quality": result["signal_quality"],
                    "project_id": project_id,
                    "company": company.strip(),
                    "project": project.strip(),
                    "processing_state": DEFAULT_PROCESSING_STATE,
                    "result": result,
                }).execute()
                print(f"[job:{job_id}] DB row written", flush=True)
            except Exception as exc:
                print(f"[job:{job_id}] DB write failed: {exc}", flush=True)

            if project_id and manufacturer:
                try:
                    supabase_client.table("projects").update({
                        "manufacturer": manufacturer, "frequency_mhz": frequency_mhz,
                    }).eq("id", project_id).execute()
                except Exception as exc:
                    print(f"[job:{job_id}] Projects update failed: {exc}", flush=True)

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

        if supabase_client:
            try:
                supabase_client.table("analysis_jobs").upsert({
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


def _extract_zip_into(tmpdir: Path, zip_path: Path) -> int:
    """Extract allowed GPR files out of a zip into tmpdir (flat). Skips
    __MACOSX, ._* AppleDouble entries, and .DS_Store. Returns extracted count."""
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = Path(member).name
            if not basename or basename.startswith("._") or basename == ".DS_Store":
                continue
            if Path(basename).suffix.lower() not in _ZIP_ALLOWED_EXTS:
                continue
            print(f"[ZIP] extracting member: {member} -> {basename}", flush=True)
            target = tmpdir / basename
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def run_proceq_job(
    job_id: str,
    file_data: Optional[list[tuple[str, bytes]]],
    tmpdir: Path,
    epsr: float,
    user_id: Optional[str] = None,
    supabase_client = None,
    company: str = "",
    project: str = "",
) -> None:
    """When file_data is None the files are assumed to already exist in tmpdir."""
    import base64
    from analysis import process_proceq_dataset
    from models import get_ensemble

    _update_job(job_id, {
        "status": "processing",
        "stage":  "Running Proceq analysis",
        "progress": 5,
    }, supabase_client)
    t0 = time.perf_counter()

    try:
        if file_data is not None:
            for fname, content in file_data:
                dest = tmpdir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)

        # If a zip was uploaded, extract its allowed entries into tmpdir
        # and remove the archive to reclaim disk.
        zip_files = list(tmpdir.glob("*.zip"))
        if zip_files:
            zip_path = zip_files[0]
            print(f"[job:{job_id}] extracting zip: {zip_path.name}", flush=True)
            _update_job(job_id, {"stage": "Extracting zip", "progress": 8}, supabase_client)
            n_extracted = _extract_zip_into(tmpdir, zip_path)
            zip_path.unlink()
            print(f"[job:{job_id}] extracted {n_extracted} files from zip", flush=True)
            _update_job(job_id, {"stage": "Processing", "progress": 10}, supabase_client)

        # Inner try so a pipeline error (e.g. no GPS-valid traces -> griddata
        # 'Number of rows must be positive') still lets the job complete with
        # whatever PNGs the pipeline wrote to tmpdir before failing.
        proceq_error_detail: Optional[str] = None
        try:
            stats = process_proceq_dataset(data_dir=str(tmpdir), output_dir=str(tmpdir), epsr=epsr)
        except Exception as exc:
            proceq_error_detail = str(exc)
            print(f"[job:{job_id}] Proceq pipeline error (continuing with partial results): {exc}", flush=True)
            stats = None

        # Optional model ensemble stats — best-effort, never fails the job.
        model_stats: dict = {"model_version": "signal_processing_only", "mean_depth_inches": None}
        try:
            ensemble = get_ensemble()
            if ensemble is not None:
                from ensemble_inference import run_ensemble_stats
                ms = run_ensemble_stats(str(tmpdir), ensemble)
                if ms:
                    model_stats = ms
                    print(f"[job:{job_id}] ensemble stats: {ms}", flush=True)
            else:
                print(f"[job:{job_id}] no model weights — skipping ensemble", flush=True)
        except Exception as exc:
            print(f"[job:{job_id}] ensemble step failed (continuing): {exc}", flush=True)

        def _enc(name: str) -> str | None:
            p = tmpdir / name
            return base64.b64encode(p.read_bytes()).decode() if p.exists() else None

        elapsed = round(time.perf_counter() - t0, 3)

        # Raw B-scan trace data per swath (zlib+base64+int8 encoded by
        # analysis_bscan.encode_traces_for_frontend). Frontend canvas decodes
        # and renders interactively — no more PNG generation server-side.
        bscan_data = (stats or {}).get('bscan_data', []) if stats else []

        proceq_result = {
            "horizon_picks":         _enc("horizon_picks.png"),
            "rebar_depth_map":       _enc("rebar_depth_map.png"),
            "corrosion_map":         _enc("corrosion_map.png"),
            "bscan_data":            bscan_data,
            "bscan_count":           len(bscan_data),
            "stats":                 stats or {},
            "analysis_time_sec":     elapsed,
            "mean_depth_inches":     model_stats.get("mean_depth_inches"),
            "mean_thickness_inches": model_stats.get("mean_thickness_inches"),
            "high_risk_pct":         model_stats.get("high_risk_pct"),
            "model_version":         model_stats.get("model_version"),
            "error_detail":          proceq_error_detail,
        }
        _jobs[job_id].update({
            "status":       "complete",
            "completed_at": time.time(),
            "result":       proceq_result,
        })
        print(f"[job:{job_id}] Proceq done in {elapsed}s", flush=True)

        # Full result upsert — Proceq has no per-field columns, so the result
        # JSONB column (migration 009) is the only durable home for horizon_picks
        # /rebar_depth_map/corrosion_map PNGs + stats. Required for cross-instance
        # polling to return the result instead of 404.
        if supabase_client:
            try:
                supabase_client.table("analysis_jobs").upsert({
                    "id":                job_id,
                    "user_id":           user_id,
                    "status":            "complete",
                    "completed_at":      "now()",
                    "progress":          100,
                    "stage":             "Complete",
                    "analysis_time_sec": elapsed,
                    "company":           company.strip(),
                    "project":           project.strip(),
                    "result":            proceq_result,
                }).execute()
                print(f"[job:{job_id}] Proceq DB row + result written", flush=True)
            except Exception as exc:
                print(f"[job:{job_id}] Proceq DB write failed (non-fatal): {exc}", flush=True)

    except Exception as exc:
        print(f"[job:{job_id}] Proceq FAILED: {exc}", flush=True)
        _update_job(job_id, {
            "status":       "failed",
            "error":        str(exc),
            "completed_at": "now()",
        }, supabase_client)
        _jobs[job_id]["completed_at"] = time.time()  # in-memory: python float

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_proceq_zip_job(
    job_id: str,
    storage_path: str,
    epsr: float,
    user_id: Optional[str],
    supabase_client,
    company: str = "",
    project: str = "",
) -> None:
    """
    Stream a zipped Proceq dataset out of the 'uploads' bucket into a tmpdir,
    extract its allowed entries, drop the archive from storage, and delegate
    the pipeline to run_proceq_job.

    Used when the upload would exceed Render's request body limit — frontend
    PUTs the zip directly to Supabase Storage and only tells us the path.
    """
    import requests

    _update_job(job_id, {
        "status":   "processing",
        "progress": 5,
        "stage":    "Downloading zip from storage",
    }, supabase_client)

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not supabase_key:
        _update_job(job_id, {
            "status":       "failed",
            "error":        "Supabase storage not configured",
            "completed_at": "now()",
        }, supabase_client)
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="verus_zip_"))
    try:
        zip_url = f"{supabase_url}/storage/v1/object/uploads/{storage_path}"
        print(f"[ZIP JOB] {job_id}: downloading from {zip_url}", flush=True)
        headers = {"Authorization": f"Bearer {supabase_key}", "apikey": supabase_key}

        zip_path = tmpdir / "dataset.zip"
        with requests.get(zip_url, headers=headers, timeout=300, stream=True) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Storage download failed: HTTP {r.status_code}")
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        zip_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"[ZIP JOB] {job_id}: downloaded {zip_mb:.1f} MB", flush=True)
        _update_job(job_id, {
            "progress": 20,
            "stage":    f"Extracting zip ({zip_mb:.0f}MB)",
        }, supabase_client)

        n_extracted = _extract_zip_into(tmpdir, zip_path)
        zip_path.unlink()
        print(f"[ZIP JOB] {job_id}: extracted {n_extracted} files", flush=True)

        extracted_files = os.listdir(tmpdir)
        print(f"[ZIP] files in tmpdir after extraction: {extracted_files}", flush=True)

        # Free the storage object early — pipeline failures shouldn't leak GB-scale
        # zips in the bucket. Best-effort; processing continues either way.
        try:
            supabase_client.storage.from_("uploads").remove([storage_path])
        except Exception as exc:
            print(f"[ZIP JOB] {job_id}: storage remove failed (non-fatal): {exc}", flush=True)

        _update_job(job_id, {
            "progress": 25,
            "stage":    f"Processing {n_extracted} files",
        }, supabase_client)

        # Hand off to the standard Proceq pipeline. tmpdir already has the
        # extracted files; run_proceq_job's finally cleans tmpdir up.
        run_proceq_job(
            job_id=job_id, file_data=None, tmpdir=tmpdir, epsr=epsr,
            user_id=user_id, supabase_client=supabase_client,
            company=company, project=project,
        )

    except Exception as exc:
        print(f"[ZIP JOB] {job_id}: fatal: {exc}", flush=True)
        _update_job(job_id, {
            "status":       "failed",
            "error":        str(exc),
            "completed_at": "now()",
        }, supabase_client)
        shutil.rmtree(tmpdir, ignore_errors=True)
