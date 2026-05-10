"""
server/jobs.py
Background job orchestrator: in-memory job store, progress helpers, image upload,
and the top-level run_analysis_job function that sequences pipeline calls.

Does NOT: contain inference logic (see pipeline.py) or model loading (see model_loader.py).
"""

import base64 as _b64
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pipeline import process_files, build_result_payload

# ── Job state ─────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)


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


def _upload_png(sb, b64: str, uid: str, job_id: str, suffix: str) -> Optional[str]:
    if not sb or not b64:
        return None
    try:
        path = f"{uid}/{job_id}{suffix}.png"
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

        _set_progress(job_id, 95, "Finalizing", supabase_client)
        uid             = user_id or "anonymous"
        cscan_url       = _upload_png(supabase_client, cscan_b64,       uid, job_id, "")
        rebar_cscan_url = _upload_png(supabase_client, rebar_cscan_b64, uid, job_id, "_rebar")
        amplitude_url   = _upload_png(supabase_client, amp_b64,         uid, job_id, "_amplitude")

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
