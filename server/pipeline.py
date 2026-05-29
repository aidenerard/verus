"""
server/pipeline.py — per-file inference loop and result-payload construction.
Does NOT manage job state, touch Supabase, start threads, or upload images.
"""
import base64 as _b64, gc
from math import radians, cos, sin, sqrt, atan2
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from ingest import detect_and_convert, SUPPORTED_EXTENSIONS, COMPANION_EXTENSIONS
from data import load_csv
from inference import run_inference, run_rebar_inference, extract_bscan_b64
from grids import (
    extract_peak_info, build_prob_grid, build_extra_grids,
    build_rebar_grids, build_peak_grid, grid_to_list,
)
from render import (
    render_cscan_b64, render_rebar_depth_b64, render_amplitude_b64,
    render_rebar_cscan_b64, compute_confidence_metrics,
)


def process_files(
    job_id: str,
    saved_paths: list[tuple[str, Path]],
    manufacturer: Optional[str],
    model,
    rebar_model,
    model_config: Optional[dict],
    frequency_mhz: int,
    set_progress_fn: Callable[[int, str], None],
    set_progress_fast_fn: Callable[[int, str], None],
) -> tuple:
    """
    Run ingestion + inference for every uploaded file.
    set_progress_fn triggers Supabase; set_progress_fast_fn is memory-only (tight loop).
    Returns: (file_preds, file_confs, file_names, file_peak_idxs, file_peak_amps,
              rebar_depth_arrs, rebar_twt_arrs, rebar_peak_arrs, per_file_summary, total_sigs)
    """
    file_preds:       list[np.ndarray] = []
    file_confs:       list[np.ndarray] = []
    file_names:       list[str]        = []
    file_peak_idxs:   list[np.ndarray] = []
    file_peak_amps:   list[np.ndarray] = []
    rebar_depth_arrs: list[np.ndarray] = []
    rebar_twt_arrs:   list[np.ndarray] = []
    rebar_peak_arrs:  list[np.ndarray] = []
    per_file_summary: list[dict]       = []
    total_sigs = 0

    n_data_files = sum(
        1 for _, d in saved_paths
        if d.suffix.lower() in SUPPORTED_EXTENSIONS and d.suffix.lower() not in COMPANION_EXTENSIONS
    )
    files_done = 0

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
            csv_path, gps = detect_and_convert(dest, upload_dir=dest.parent,
                                                manufacturer=manufacturer)
        except Exception as exc:
            print(f"[job:{job_id}] Conversion failed for {fname}: {exc}", flush=True)
            continue

        try:
            signals = load_csv(csv_path)
        except Exception as exc:
            print(f"[job:{job_id}] load_csv failed for {fname}: {exc}", flush=True)
            continue

        n_signals = signals.shape[0]
        print(f"[job:{job_id}]   → {n_signals} signals, running inference…", flush=True)
        set_progress_fn(
            15 + round(65 * files_done / max(n_data_files, 1)),
            f"Preprocessing file {files_done + 1}/{max(n_data_files,1)}",
        )

        bscan_info         = extract_bscan_b64(signals)
        peak_idx, peak_amp = extract_peak_info(signals)

        file_prog_start = 15 + round(65 * files_done / max(n_data_files, 1))
        file_prog_end   = 15 + round(65 * (files_done + 1) / max(n_data_files, 1))

        def _infer_cb(processed: int, total: int) -> None:
            t = processed / max(total, 1)
            p = file_prog_start + round((file_prog_end - file_prog_start) * t)
            set_progress_fast_fn(p, "Running inference")

        preds, confs                 = run_inference(model, signals, model_config=model_config,
                                                      progress_callback=_infer_cb)
        depth_arr, twt_arr, peak_arr = run_rebar_inference(rebar_model, signals, frequency_mhz)
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
        rebar_peak_arrs.append(peak_arr)
        per_file_summary.append({
            "filename":          fname,
            "signals":           n,
            "delam_pct":         delam_pct,
            "gps":               gps,
            "bscan_data":        bscan_info,
            "rebar_depth_mean":  rdm,
            "rebar_depth_min":   rdmn,
            "rebar_depth_max":   rdmx,
            "rebar_depth_array": depth_arr[:512].tolist(),
            "twt_array":         twt_arr[:512].tolist(),
            "peak_sample_array": peak_arr[:512].tolist(),
        })
        total_sigs += n
        files_done += 1
        set_progress_fn(file_prog_end, "Ingesting")

    return (
        file_preds, file_confs, file_names,
        file_peak_idxs, file_peak_amps,
        rebar_depth_arrs, rebar_twt_arrs, rebar_peak_arrs,
        per_file_summary, total_sigs,
    )


def build_result_payload(
    job_id: str,
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
    file_names: list[str],
    file_peak_idxs: list[np.ndarray],
    file_peak_amps: list[np.ndarray],
    rebar_depth_arrs: list[np.ndarray],
    rebar_twt_arrs: list[np.ndarray],
    rebar_peak_arrs: list[np.ndarray],
    per_file_summary: list[dict],
    total_sigs: int,
    frequency_mhz: int,
    manufacturer: Optional[str],
    rebar_model,
    swath_spacing_ft: float,
    structure_name: str,
) -> tuple[dict, str, str, str]:
    """Build grids, render images, assemble result dict.
    Returns (result_dict, cscan_b64, rebar_cscan_b64, amp_b64). result_dict excludes analysis_time_sec."""
    all_preds       = np.concatenate(file_preds)
    n_del_total     = int((all_preds == 0).sum())
    del all_preds
    delam_pct_total = round(n_del_total / total_sigs * 100, 2)
    sound_pct_total = round(100.0 - delam_pct_total, 2)

    gc.collect()

    try:
        cscan_b64 = render_cscan_b64(
            file_preds, file_confs, file_names,
            swath_spacing_ft=swath_spacing_ft,
            structure_name=structure_name,
        )
        print(f"[job:{job_id}] C-scan rendered ({len(cscan_b64)//1024} KB b64)", flush=True)
    except Exception as exc:
        print(f"[job:{job_id}] C-scan render failed: {exc}", flush=True)
        cscan_b64 = ""

    try:
        prob_grid, otsu_T = build_prob_grid(file_preds, file_confs)
        prob_b64         = _b64.b64encode(prob_grid.tobytes()).decode()
        prob_grid_data_j = grid_to_list(prob_grid)
        pg_rows, pg_cols = prob_grid.shape
        del prob_grid
    except Exception as exc:
        print(f"[job:{job_id}] prob_grid failed: {exc}", flush=True)
        prob_b64 = ""; prob_grid_data_j = []; pg_rows = pg_cols = 0; otsu_T = 0.65

    rebar_cscan_b64 = ""
    rebar_depth_grid_j: list = []; rebar_twt_grid_j: list = []; rebar_peak_grid_j: list = []
    try:
        rebar_dg, rebar_tg = build_rebar_grids(rebar_depth_arrs, rebar_twt_arrs)
        peak_g             = build_peak_grid(rebar_peak_arrs)
        rebar_peak_grid_j  = grid_to_list(peak_g)
        del peak_g

        along_track_ft_per_col = 1.0
        try:
            total_ft = 0.0
            for _f in per_file_summary:
                _gps = _f.get('gps')
                if _gps and len(_gps.get('coordinates', [])) >= 2:
                    _c = _gps['coordinates']
                    lat1, lon1 = radians(_c[0][0]), radians(_c[0][1])
                    lat2, lon2 = radians(_c[-1][0]), radians(_c[-1][1])
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                    total_ft += 2 * 6_371_000 * atan2(sqrt(a), sqrt(1-a)) * 3.28084
            if total_ft > 0 and rebar_dg.shape[1] > 0:
                along_track_ft_per_col = total_ft / rebar_dg.shape[1]
        except Exception:
            pass

        x_extent = rebar_dg.shape[1] * along_track_ft_per_col
        rebar_picks: list[tuple[float, float]] = []
        for fi, da in enumerate(rebar_depth_arrs):
            n = len(da)
            y_ft = fi * swath_spacing_ft
            step = max(1, n // 300)
            for si in range(0, n, step):
                rebar_picks.append((float(si) / max(n - 1, 1) * x_extent, y_ft))

        rebar_cscan_b64 = render_rebar_cscan_b64(
            rebar_dg,
            swath_spacing_ft=swath_spacing_ft,
            along_track_ft_per_col=along_track_ft_per_col,
            structure_name=structure_name,
            rebar_picks=rebar_picks,
        )
        rebar_depth_grid_j = grid_to_list(rebar_dg)
        rebar_twt_grid_j   = grid_to_list(rebar_tg)
        del rebar_dg, rebar_tg

    except Exception as exc:
        print(f"[job:{job_id}] Rebar grid render failed: {exc}", flush=True)

    amp_b64 = twt_b64 = ""
    amplitude_grid_data_j: list = []; twt_rows = twt_cols = 0
    conf_pct = depth_acc_in = 0.0; sig_quality = "Fair"
    try:
        depth_grid, amp_grid, twt_grid = build_extra_grids(
            file_peak_idxs, file_peak_amps, frequency_mhz
        )
        rebar_b64             = render_rebar_depth_b64(depth_grid)
        amp_b64               = render_amplitude_b64(amp_grid)
        amplitude_grid_data_j = grid_to_list(amp_grid)
        twt_b64               = _b64.b64encode(twt_grid.tobytes()).decode()
        twt_rows, twt_cols = twt_grid.shape

        all_confs_flat = np.concatenate(file_confs)
        conf_pct, depth_acc_in, sig_quality = compute_confidence_metrics(
            all_confs_flat, amp_grid, frequency_mhz
        )
        del depth_grid, amp_grid, twt_grid, all_confs_flat
        print(f"[job:{job_id}] Extra grids rendered. Confidence={conf_pct:.1f}%", flush=True)
    except Exception as exc:
        print(f"[job:{job_id}] Extra grids failed: {exc}", flush=True)

    # Aggregate per-file B-scan blobs into a top-level array so the frontend
    # BScanViewer (which reads result.bscan_data) finds them at the same path
    # as the Proceq path. Each entry is one swath/file's preprocessed traces.
    bscan_list = [f["bscan_data"] for f in per_file_summary if f.get("bscan_data")]

    result = {
        "signals_analyzed":    total_sigs,
        "delamination_pct":    delam_pct_total,
        "sound_pct":           sound_pct_total,
        "per_file_summary":    per_file_summary,
        "bscan_data":          bscan_list,
        "bscan_count":         len(bscan_list),
        "rebar_model_used":    rebar_model is not None,
        "prob_grid_data":      prob_grid_data_j,
        "amplitude_grid_data": amplitude_grid_data_j,
        "rebar_depth_grid":    rebar_depth_grid_j,
        "rebar_twt_grid":      rebar_twt_grid_j,
        "rebar_peak_grid":     rebar_peak_grid_j,
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
        "manufacturer":        manufacturer,
        "model_confidence_pct": conf_pct,
        "depth_accuracy_in":   depth_acc_in,
        "signal_quality":      sig_quality,
    }

    return result, cscan_b64, rebar_cscan_b64, amp_b64
