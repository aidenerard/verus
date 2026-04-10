"""
Flask Server for Verus GPR Analysis - Render.com Optimized
This version is specifically configured to work on Render.com
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import subprocess
import tempfile
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths - Render specific
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RUN_SCRIPT = BASE_DIR.parent / "run.py"

print(f"[STARTUP] Base directory: {BASE_DIR}", flush=True)
print(f"[STARTUP] Models directory: {MODELS_DIR}", flush=True)
print(f"[STARTUP] Run script: {RUN_SCRIPT}", flush=True)
print(f"[STARTUP] Models directory exists: {MODELS_DIR.exists()}", flush=True)
print(f"[STARTUP] Run script exists: {RUN_SCRIPT.exists()}", flush=True)

if MODELS_DIR.exists():
    model_files = list(MODELS_DIR.glob("*.pth"))
    print(f"[STARTUP] Found {len(model_files)} model file(s): {[f.name for f in model_files]}", flush=True)
else:
    print("[STARTUP] WARNING: Models directory not found!", flush=True)

@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "service": "Verus GPR Analysis",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "analyze": "/analyze (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        models = list_models()
        return jsonify({
            "status": "ok",
            "models_available": models,
            "model_path": str(MODELS_DIR),
            "run_script": str(RUN_SCRIPT),
            "run_script_exists": RUN_SCRIPT.exists()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/models", methods=["GET"])
def get_models():
    """List available model files"""
    try:
        models = list_models()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run GPR analysis on uploaded CSV files

    Expected form data:
      - files: CSV file(s) to analyze
      - threshold: Detection threshold (default 0.65)
      - model_version: Model filename (default: latest)
    """
    try:
        print("[ANALYZE] Starting analysis request", flush=True)

        # Get parameters
        threshold = float(request.form.get('threshold', 0.65))
        model_version = request.form.get('model_version', get_latest_model())

        print(f"[ANALYZE] Threshold: {threshold}, Model: {model_version}", flush=True)

        # Validate model exists
        model_path = MODELS_DIR / model_version
        if not model_path.exists():
            print(f"[ANALYZE] ERROR: Model not found at {model_path}", flush=True)
            return jsonify({"error": f"Model not found: {model_version}"}), 404

        # Get uploaded files
        files = request.files.getlist('files')
        if not files:
            print("[ANALYZE] ERROR: No files uploaded", flush=True)
            return jsonify({"error": "No files uploaded"}), 400

        print(f"[ANALYZE] Received {len(files)} file(s)", flush=True)

        # Create temporary directory for input files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Save uploaded files
            for file in files:
                file_path = input_dir / file.filename
                file.save(str(file_path))
                print(f"[ANALYZE] Saved: {file.filename}", flush=True)

            # Run inference
            output_path = Path(tmpdir) / "results.json"

            cmd = [
                sys.executable,  # Use current Python interpreter
                str(RUN_SCRIPT),
                "--input", str(input_dir),
                "--model", str(model_path),
                "--threshold", str(threshold),
                "--output", str(output_path)
            ]

            print(f"[ANALYZE] Running command: {' '.join(cmd)}", flush=True)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            print(f"[ANALYZE] Command finished with return code: {result.returncode}", flush=True)

            if result.returncode != 0:
                print(f"[ANALYZE] STDERR: {result.stderr}", flush=True)
                print(f"[ANALYZE] STDOUT: {result.stdout}", flush=True)
                return jsonify({
                    "error": "Inference failed",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }), 500

            # Read results
            if not output_path.exists():
                print(f"[ANALYZE] ERROR: Output file not created", flush=True)
                return jsonify({
                    "error": "Analysis completed but no output file generated",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }), 500

            with open(output_path, 'r') as f:
                analysis_result = json.load(f)

            print(f"[ANALYZE] Success! Analyzed {analysis_result.get('signals_analyzed', 0)} signals", flush=True)

            return jsonify(analysis_result)

    except subprocess.TimeoutExpired:
        print("[ANALYZE] ERROR: Analysis timeout", flush=True)
        return jsonify({"error": "Analysis timeout (>5 minutes)"}), 500
    except Exception as e:
        print(f"[ANALYZE] ERROR: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def list_models():
    """List all .pth model files in models directory"""
    if not MODELS_DIR.exists():
        return []

    models = [f.name for f in MODELS_DIR.glob("*.pth")]

    # Sort by version number (descending)
    models.sort(key=lambda x: (
        int(m.group(1)) if (m := __import__('re').search(r'v(\d+)', x)) else 0
    ), reverse=True)

    return models

def get_latest_model():
    """Get the most recent model version"""
    models = list_models()
    return models[0] if models else "model_v13.pth"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    print("=" * 60, flush=True)
    print("Verus GPR Analysis Server - Starting", flush=True)
    print("=" * 60, flush=True)
    print(f"Port: {port}", flush=True)
    print(f"Base directory: {BASE_DIR}", flush=True)
    print(f"Models directory: {MODELS_DIR}", flush=True)
    print(f"Run script: {RUN_SCRIPT}", flush=True)

    # Validate setup
    if not RUN_SCRIPT.exists():
        print(f"ERROR: run.py not found at {RUN_SCRIPT}", flush=True)
        sys.exit(1)

    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found at {MODELS_DIR}", flush=True)
        sys.exit(1)

    models = list_models()
    if not models:
        print("WARNING: No .pth model files found in models/ directory", flush=True)
    else:
        print(f"Found models: {models}", flush=True)

    print("=" * 60, flush=True)
    print("Starting server...", flush=True)
    print("=" * 60, flush=True)

    app.run(host="0.0.0.0", port=port, debug=False)
