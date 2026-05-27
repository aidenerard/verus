"""
server/model_loader.py
Downloads model weights and config files from Google Drive (or any URL),
then loads CNN1D and HorizonCNN into memory.

Does NOT: set any globals, start threads, or interact with FastAPI.
"""

import json
import os
from pathlib import Path
from typing import Optional

import requests
import torch

from run import CNN1D, HorizonCNN, DEVICE


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


def load_models_background(
    model_path: Path,
    rebar_model_path: Path,
    model_config_path: Path,
    rebar_model_config_path: Path,
) -> tuple[Optional[CNN1D], Optional[HorizonCNN], Optional[dict]]:
    """
    Download (if needed) and load the delamination and rebar models.
    Returns (model, rebar_model, model_config).
    Any value may be None if loading fails for that model.
    """
    model: Optional[CNN1D] = None
    rebar_model: Optional[HorizonCNN] = None
    loaded_cfg: Optional[dict] = None

    print(f"[startup] Looking for model at: {model_path.resolve()}", flush=True)

    if not model_path.exists():
        gdrive_url = os.environ.get("MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading model from {gdrive_url} …", flush=True)
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                download_file(gdrive_url, str(model_path))
            except Exception as exc:
                print(f"[startup] ERROR: model download failed: {exc}", flush=True)
                return None, None, None
        else:
            print(
                f"[startup] WARNING: {model_path} missing and MODEL_GDRIVE_URL unset "
                "— /analyze returns 503.",
                flush=True,
            )
            return None, None, None

    if not model_path.exists():
        print(f"[startup] ERROR: Model still missing after download: {model_path}", flush=True)
        return None, None, None

    print(f"[startup] model.pth size: {model_path.stat().st_size:,} bytes", flush=True)

    cfg_url = os.environ.get("MODEL_CONFIG_URL")
    if cfg_url and not model_config_path.exists():
        try:
            download_file(cfg_url, str(model_config_path))
        except Exception as exc:
            print(f"[startup] WARNING: model_config.json download failed: {exc}", flush=True)

    if not model_config_path.exists():
        raise RuntimeError(
            f"model_config.json not found at {model_config_path} and MODEL_CONFIG_URL is not set. "
            "Set MODEL_CONFIG_URL to a raw JSON download URL."
        )

    try:
        with open(model_config_path) as f:
            loaded_cfg = json.load(f)
    except json.JSONDecodeError as e:
        with open(model_config_path) as f:
            head = f.read(200)
        raise RuntimeError(
            f"Config from {cfg_url!r} is not valid JSON. First 200 bytes: {head!r}"
        ) from e
    print(f"[startup] Config: {loaded_cfg}", flush=True)

    arch = {k: loaded_cfg[k] for k in ("in_channels", "conv_channels", "head_hidden") if k in loaded_cfg}
    print(f"[startup] Loading arch: {arch}", flush=True)
    try:
        m = CNN1D(**arch).to(DEVICE)
        m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        m.eval()
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        model = m
        print(f"[startup] Model loaded ({n_p:,} params) with config: {arch}", flush=True)
    except Exception as exc:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        ckpt_shapes = {k: tuple(v.shape) for k, v in list(ckpt.items())[:8]}
        raise RuntimeError(
            f"Model weights incompatible with config {arch}.\n"
            f"  Checkpoint shapes (first 8): {ckpt_shapes}\n"
            f"  Original error: {exc}"
        ) from exc

    # ── Horizon (rebar depth) model ───────────────────────────────────────────
    if not rebar_model_path.exists():
        gdrive_url = os.environ.get("REBAR_MODEL_GDRIVE_URL")
        if gdrive_url:
            print(f"[startup] Downloading horizon model from {gdrive_url} …", flush=True)
            try:
                rebar_model_path.parent.mkdir(parents=True, exist_ok=True)
                download_file(gdrive_url, str(rebar_model_path))
            except Exception as exc:
                print(f"[startup] WARNING: Horizon model download failed: {exc}", flush=True)
        else:
            print(
                "[startup] REBAR_MODEL_GDRIVE_URL not set — "
                "rebar depth will use physics fallback.",
                flush=True,
            )

    if rebar_model_path.exists():
        try:
            rm = HorizonCNN().to(DEVICE)
            rm.load_state_dict(
                torch.load(rebar_model_path, map_location=DEVICE, weights_only=False)
            )
            rm.eval()
            n_rp = sum(p.numel() for p in rm.parameters() if p.requires_grad)
            rebar_model = rm
            print(f"[startup] HorizonCNN loaded ({n_rp:,} params)", flush=True)
        except Exception as exc:
            print(f"[startup] WARNING: HorizonCNN load failed: {exc}", flush=True)

    return model, rebar_model, loaded_cfg
