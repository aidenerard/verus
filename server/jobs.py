"""
server/jobs.py
Background analysis job runner + in-memory job store.
"""

import base64 as _b64
import gc
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from ingest import detect_and_convert, SUPPORTED_EXTENSIONS, COMPANION_EXTENSIONS
from run import (
    load_csv, run_inference, run_rebar_inference, render_cscan_b64,
    extract_bscan_b64, extract_peak_info,
    build_prob_grid, build_extra_grids, build_rebar_grids, grid_to_list,
    render_rebar_depth_b64, render_rebar_cscan_b64, render_amplitude_b64,
    compute_confidence_metrics,
)

# ── Job state ─────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── Worker ────────────────────────────────────────────────────────────────────

def run_analysis_job(
    job_id: str,
    file_data: list[tuple[str, bytes]],
    user_id: Optional[str],
    tmpdir: Path,
    manufacturer: Optional[str],
    frequency_mhz: int,
    model,
    rebar_model,
    model_config: Optional[dict],
    supabase_client,
) -> None:
    """
    Runs in a ThreadPoolExecutor worker thread.
    Saves files, runs inference, writes results to Supabase, updates _jobs.
    """
    _jobs[job_id]["status"] = "processing"
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

        # ── Inference ─────────────────────────────────────────────────────────
        file_preds:      list[np.ndarray] = []
        file_confs:      list[np.ndarray] = []
        file_names:      list[str]        = []
        file_peak_idxs:  list[np.ndarray] = []
        file_peak_amps:  list[np.ndarray] = []
        rebar_depth_arrs: list[np.ndarray] = []
        rebar_twt_arrs:   list[np.ndarray] = []
        per_file_summary: list[dict]       = []
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
                csv_path, gps = detect_and_convert(dest, upload_dir=tmpdir,
                                                    manufacturer=manufacturer)
            except Exception as exc:
                print(f"[job:{job_id}] Conversion failed for {fname}: {exc}", flush=True)
                continue

            try:
                signals = load_csv(csv_path)
            except Exception as exc:
                print(f"[job:{job_id}] load_csv failed for {fname}: {exc}", flush=True)
                continue

            print(f"[job:{job_id}]   → {signals.shape[0]} signals, running inference…", flush=True)
            bscan_info         = extract_bscan_b64(signals)
            peak_idx, peak_amp = extract_peak_info(signals)
            preds, confs       = run_inference(model, signals, model_config=model_config)
            depth_arr, twt_arr = run_rebar_inference(rebar_model, signals, frequency_mhz)
            del signals
            if csv_path != dest:
                csv_path.unlink(missing_ok=True)
            gc.collect()

            n         = len(preds)
            n_delam   = int((preds == 0).sum())
            delam_pct = round(n_delam / n * 100, 2) if n else 0.0
            rdm  = round(float(np.nanmean(depth_arr)), 3)
            rdmn = round(float(np.nanmin(depth_arr)),  3)
            rdmx = round(float(np.nanmax(depth_arr)),  3)

            file_preds.append(preds)
            file_confs.append(confs)
            file_names.append(fname)
            file_peak_idxs.append(peak_idx)
            file_peak_amps.append(peak_amp)
            rebar_depth_arrs.append(depth_arr)
            rebar_twt_arrs.append(twt_arr)
            per_file_summary.append({
                "filename":         fname,
                "signals":          n,
                "delam_pct":        delam_pct,
                "gps":              gps,
                "bscan":            bscan_info,
                "rebar_depth_mean": rdm,
                "rebar_depth_min":  rdmn,
                "rebar_depth_max":  rdmx,
                "rebar_depth_array": depth_arr[:512].tolist(),
                "twt_array":         twt_arr[:512].tolist(),
            })
            total_sigs += n

        if total_sigs == 0:
            raise ValueError("No valid GPR signals found in uploaded files.")

        all_preds       = np.concatenate(file_preds)
        n_del_total     = int((all_preds == 0).sum())
        del all_preds
        delam_pct_total = round(n_del_total / total_sigs * 100, 2)
        sound_pct_total = round(100.0 - delam_pct_total, 2)

        # ── Render images and build extra grids ──────────────────────────────
        gc.collect()
        try:
            cscan_b64 = render_cscan_b64(file_preds, file_confs, file_names)
            print(f"[job:{job_id}] C-scan rendered ({len(cscan_b64)//1024} KB b64)", flush=True)
        except Exception as exc:
            print(f"[job:{job_id}] C-scan render failed: {exc}", flush=True)
            cscan_b64 = ""

        try:
            prob_grid, otsu_T = build_prob_grid(file_preds, file_confs)
            prob_b64  = _b64.b64encode(prob_grid.tobytes()).decode()
            pg_rows, pg_cols = prob_grid.shape
            del prob_grid
        except Exception as exc:
            print(f"[job:{job_id}] prob_grid failed: {exc}", flush=True)
            prob_b64 = ""; pg_rows = pg_cols = 0; otsu_T = 0.65

        rebar_cscan_b64    = ""
        rebar_depth_grid_j: list = []
        rebar_twt_grid_j:   list = []
        try:
            rebar_dg, rebar_tg = build_rebar_grids(rebar_depth_arrs, rebar_twt_arrs)
            rebar_cscan_b64    = render_rebar_cscan_b64(rebar_dg, frequency_mhz, rebar_model is not None)
            rebar_depth_grid_j = grid_to_list(rebar_dg)
            rebar_twt_grid_j   = grid_to_list(rebar_tg)
            del rebar_dg, rebar_tg
        except Exception as exc:
            print(f"[job:{job_id}] Rebar grid render failed: {exc}", flush=True)

        rebar_b64 = amp_b64 = twt_b64 = ""
        twt_rows = twt_cols = 0
        conf_pct = depth_acc_in = 0.0
        sig_quality = "Fair"
        try:
            depth_grid, amp_grid, twt_grid = build_extra_grids(
                file_peak_idxs, file_peak_amps, frequency_mhz
            )
            rebar_b64 = render_rebar_depth_b64(depth_grid)
            amp_b64   = render_amplitude_b64(amp_grid)
            twt_b64   = _b64.b64encode(twt_grid.tobytes()).decode()
            twt_rows, twt_cols = twt_grid.shape

            all_confs_flat = np.concatenate(file_confs)
            conf_pct, depth_acc_in, sig_quality = compute_confidence_metrics(
                all_confs_flat, amp_grid, frequency_mhz
            )
            del depth_grid, amp_grid, twt_grid, all_confs_flat
            print(f"[job:{job_id}] Extra grids rendered. Confidence={conf_pct:.1f}%", flush=True)
        except Exception as exc:
            print(f"[job:{job_id}] Extra grids failed: {exc}", flush=True)

        elapsed = round(time.perf_counter() - t0, 3)

        result = {
            "signals_analyzed":    total_sigs,
            "delamination_pct":    delam_pct_total,
            "sound_pct":           sound_pct_total,
            "analysis_time_sec":   elapsed,
            "cscan_image":         cscan_b64,
            "per_file_summary":    per_file_summary,
            "rebar_model_used":    rebar_model is not None,
            "rebar_cscan_image":   rebar_cscan_b64,
            "rebar_depth_grid":    rebar_depth_grid_j,
            "rebar_twt_grid":      rebar_twt_grid_j,
            "rebar_depth_image":   rebar_b64,
            "amplitude_image":     amp_b64,
            "prob_grid":           prob_b64,
            "prob_grid_rows":      pg_rows,
            "prob_grid_cols":      pg_cols,
            "otsu_threshold":      round(float(otsu_T), 4),
            "twt_grid":            twt_b64,
            "twt_grid_rows":       twt_rows,
            "twt_grid_cols":       twt_cols,
            "frequency_mhz":       frequency_mhz,
            "model_confidence_pct": conf_pct,
            "depth_accuracy_in":   depth_acc_in,
            "signal_quality":      sig_quality,
        }

        # ── Supabase: storage + DB row ────────────────────────────────────────
        cscan_url: Optional[str] = None
        if supabase_client and cscan_b64:
            try:
                png_bytes    = _b64.b64decode(cscan_b64)
                uid          = user_id or "anonymous"
                storage_path = f"{uid}/{job_id}.png"
                supabase_client.storage.from_("cscan-images").upload(
                    storage_path, png_bytes, {"content-type": "image/png"},
                )
                cscan_url = supabase_client.storage.from_("cscan-images").get_public_url(storage_path)
                print(f"[job:{job_id}] C-scan uploaded: {storage_path}", flush=True)
            except Exception as exc:
                print(f"[job:{job_id}] Storage upload failed: {exc}", flush=True)

        if supabase_client:
            try:
                supabase_client.table("analysis_jobs").upsert({
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
                }).execute()
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
