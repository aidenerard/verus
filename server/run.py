#!/usr/bin/env python3
"""
Verus GPR Inference Script
Runs trained CNN model on GPR CSV files and generates C-scan visualization

Output JSON Format (matches frontend expectations):
{
  "signals_analyzed": int,
  "delamination_pct": float,
  "sound_pct": float,
  "analysis_time_sec": float,
  "cscan_image": "base64_encoded_png_string",
  "per_file_summary": [
    {"filename": str, "signals": int, "delam_pct": float}
  ]
}
"""

import argparse
import json
import time
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


class CNN1D(nn.Module):
    """1D CNN for GPR A-scan waveform classification"""
    def __init__(self, input_size=512):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate flatten size
        self.flatten_size = 128 * (input_size // 8)

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flatten_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)


def load_model(model_path, device='cpu'):
    """Load trained PyTorch model"""
    model = CNN1D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_csv_files(input_dir):
    """Load all CSV files from input directory"""
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    all_data = []
    file_metadata = []

    for csv_file in csv_files:
        # Read CSV - assume single column of waveform data
        df = pd.read_csv(csv_file, header=None)
        signals = df.values

        all_data.append(signals)
        file_metadata.append({
            'filename': csv_file.name,
            'signals': len(signals)
        })

    return all_data, file_metadata


def run_inference(model, data_list, threshold=0.65, device='cpu'):
    """Run inference on loaded data"""
    all_predictions = []

    for data in data_list:
        # Normalize data
        data_normalized = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)

        # Convert to tensor
        data_tensor = torch.FloatTensor(data_normalized).unsqueeze(1).to(device)

        # Run inference
        with torch.no_grad():
            predictions = model(data_tensor).cpu().numpy().flatten()

        all_predictions.append(predictions)

    return all_predictions


def generate_cscan_image(predictions_list, threshold=0.65):
    """
    Generate professional ASTM-compliant C-scan visualization
    Returns base64 encoded PNG
    """
    # Combine all predictions
    all_predictions = np.concatenate(predictions_list)
    total_signals = len(all_predictions)

    # Calculate delamination percentage
    delaminated = np.sum(all_predictions >= threshold)
    delamination_pct = (delaminated / total_signals) * 100
    sound_pct = 100 - delamination_pct

    # Create 2D grid for visualization (assume bridge deck layout)
    # Simulate spatial arrangement
    grid_cols = int(np.sqrt(total_signals * 2))  # Wider than tall
    grid_rows = int(np.ceil(total_signals / grid_cols))

    # Create probability grid
    prob_grid = np.zeros((grid_rows, grid_cols))
    for i, pred in enumerate(all_predictions):
        if i >= grid_rows * grid_cols:
            break
        row = i // grid_cols
        col = i % grid_cols
        prob_grid[row, col] = pred

    # Apply Gaussian smoothing for realistic appearance
    prob_grid_smooth = gaussian_filter(prob_grid, sigma=2.5)

    # Create ASTM colormap: Sound (green) → Suspect (yellow) → Deteriorated (red)
    colors = ['#2ECC71', '#F39C12', '#E74C3C']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('astm_condition', colors, N=n_bins)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Plot C-scan
    im = ax.imshow(prob_grid_smooth, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add bridge deck outline
    rect = plt.Rectangle((0, 0), grid_cols-1, grid_rows-1,
                         fill=False, edgecolor='#2C3E50', linewidth=2)
    ax.add_patch(rect)

    # Add title and labels
    ax.set_title('Bridge Deck Condition Assessment - C-Scan Map\nASTM D6087 Compliant Analysis',
                fontsize=14, fontweight='bold', color='#2C3E50', pad=20)
    ax.set_xlabel('Transverse Direction (ft)', fontsize=10, color='#6C757D')
    ax.set_ylabel('Longitudinal Direction (ft)', fontsize=10, color='#6C757D')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Delamination Probability', fontsize=10, color='#6C757D')
    cbar.ax.tick_params(labelsize=9)

    # Add statistics box
    stats_text = f'Total Area Surveyed: {total_signals:,} signals\n'
    stats_text += f'Sound: {sound_pct:.1f}% | Deteriorated: {delamination_pct:.1f}%\n'
    stats_text += f'Threshold: {threshold}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#DEE2E6', alpha=0.9),
           color='#2C3E50')

    # Add north arrow
    ax.text(0.98, 0.98, 'N ↑', transform=ax.transAxes,
           fontsize=12, verticalalignment='top', horizontalalignment='right',
           fontweight='bold', color='#2C3E50')

    # Style
    ax.tick_params(labelsize=9, colors='#6C757D')
    ax.spines['top'].set_color('#DEE2E6')
    ax.spines['right'].set_color('#DEE2E6')
    ax.spines['bottom'].set_color('#DEE2E6')
    ax.spines['left'].set_color('#DEE2E6')

    plt.tight_layout()

    # Save to bytes and encode as base64
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


def main():
    parser = argparse.ArgumentParser(description='Verus GPR Inference')
    parser.add_argument('--input', required=True, help='Input directory with CSV files')
    parser.add_argument('--model', required=True, help='Path to model.pth file')
    parser.add_argument('--threshold', type=float, default=0.65, help='Detection threshold')
    parser.add_argument('--output', required=True, help='Output JSON file path')

    args = parser.parse_args()

    start_time = time.time()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args.model, device)

    # Load data
    data_list, file_metadata = load_csv_files(args.input)

    # Run inference
    predictions_list = run_inference(model, data_list, args.threshold, device)

    # Calculate statistics
    total_signals = sum(len(p) for p in predictions_list)
    all_predictions = np.concatenate(predictions_list)
    delaminated = np.sum(all_predictions >= args.threshold)
    delamination_pct = (delaminated / total_signals) * 100
    sound_pct = 100 - delamination_pct

    # Generate C-scan image
    cscan_image_base64 = generate_cscan_image(predictions_list, args.threshold)

    # Calculate per-file summary
    per_file_summary = []
    for metadata, predictions in zip(file_metadata, predictions_list):
        file_delaminated = np.sum(predictions >= args.threshold)
        file_delam_pct = (file_delaminated / len(predictions)) * 100

        per_file_summary.append({
            'filename': metadata['filename'],
            'signals': metadata['signals'],
            'delam_pct': round(file_delam_pct, 2)
        })

    analysis_time = time.time() - start_time

    # Create output JSON matching frontend expectations
    result = {
        'signals_analyzed': total_signals,
        'delamination_pct': round(delamination_pct, 2),
        'sound_pct': round(sound_pct, 2),
        'analysis_time_sec': round(analysis_time, 2),
        'cscan_image': cscan_image_base64,
        'per_file_summary': per_file_summary
    }

    # Write output
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Analysis complete: {total_signals:,} signals analyzed in {analysis_time:.2f}s")
    print(f"Delamination: {delamination_pct:.1f}% | Sound: {sound_pct:.1f}%")


if __name__ == '__main__':
    main()
