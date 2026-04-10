"""
Flask Server for Verus GPR Analysis
Deploy this Python server separately (Railway, Render, AWS, etc.)

Required packages:
  pip install flask torch pandas numpy matplotlib scipy flask-cors

Required files:
  - run.py (your inference script)
  - models/model_v13.pth (or your trained model)

Environment variables:
  PORT=8080 (or whatever your hosting service uses)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import subprocess
import tempfile
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths
MODELS_DIR = Path(__file__).parent / "models"
RUN_SCRIPT = Path(__file__).parent / "run.py"

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "models_available": list_models()})

@app.route("/models", methods=["GET"])
def get_models():
    """List available model files"""
    models = list_models()
    return jsonify({"models": models})

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
        # Get parameters
        threshold = float(request.form.get('threshold', 0.65))
        model_version = request.form.get('model_version', get_latest_model())

        # Validate model exists
        model_path = MODELS_DIR / model_version
        if not model_path.exists():
            return jsonify({"error": f"Model not found: {model_version}"}), 404

        # Get uploaded files
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        # Create temporary directory for input files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Save uploaded files
            for file in files:
                file_path = input_dir / file.filename
                file.save(str(file_path))

            # Run inference
            output_path = Path(tmpdir) / "results.json"

            cmd = [
                "python3",
                str(RUN_SCRIPT),
                "--input", str(input_dir),
                "--model", str(model_path),
                "--threshold", str(threshold),
                "--output", str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return jsonify({
                    "error": "Inference failed",
                    "stderr": result.stderr
                }), 500

            # Read results
            with open(output_path, 'r') as f:
                analysis_result = json.load(f)

            # Convert C-scan image to base64 if it exists
            if 'cscanImagePath' in analysis_result:
                cscan_path = Path(analysis_result['cscanImagePath'])
                if cscan_path.exists():
                    with open(cscan_path, 'rb') as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        analysis_result['cscanImageBase64'] = img_base64
                    del analysis_result['cscanImagePath']

            return jsonify(analysis_result)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Analysis timeout (>5 minutes)"}), 500
    except Exception as e:
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

    # Validate setup
    if not RUN_SCRIPT.exists():
        print(f"ERROR: run.py not found at {RUN_SCRIPT}")
        exit(1)

    if not MODELS_DIR.exists():
        print(f"WARNING: Models directory not found at {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = list_models()
    if not models:
        print("WARNING: No .pth model files found in models/ directory")
    else:
        print(f"Found models: {models}")

    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
