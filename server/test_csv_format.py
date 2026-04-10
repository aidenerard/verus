#!/usr/bin/env python3
"""
Test script to verify your CSV files are in the correct format for Verus
Run this before deploying to check your data structure
"""

import pandas as pd
import sys
from pathlib import Path

def test_csv_file(csv_path):
    """Test a single CSV file"""
    print(f"\n{'='*60}")
    print(f"Testing: {csv_path}")
    print(f"{'='*60}")

    try:
        # Try to load CSV
        df = pd.read_csv(csv_path, header=None)

        print(f"✓ File loaded successfully")
        print(f"  Shape: {df.shape} (rows × columns)")
        print(f"  Data type: {df.dtypes[0]}")

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            print(f"✗ WARNING: First column is not numeric!")
            print(f"  Type: {df.dtypes[0]}")
            return False

        print(f"✓ Data is numeric")

        # Show sample
        print(f"\nFirst 5 values:")
        print(df.head())

        # Check for NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"✗ WARNING: Found {nan_count} NaN values")
            return False

        print(f"✓ No NaN values found")

        # Estimate signals
        if df.shape[1] == 1:
            # Single column - assume each row is a sample, file has 1 signal
            signals = 1
            samples_per_signal = df.shape[0]
        else:
            # Multiple columns - assume each column is a signal
            signals = df.shape[1]
            samples_per_signal = df.shape[0]

        print(f"\nEstimated structure:")
        print(f"  Signals in file: {signals}")
        print(f"  Samples per signal: {samples_per_signal}")

        # Check signal length
        if samples_per_signal < 100:
            print(f"✗ WARNING: Very short signals ({samples_per_signal} samples)")
            print(f"  Typical GPR A-scans have 256-1024 samples")
        elif samples_per_signal > 2048:
            print(f"✗ WARNING: Very long signals ({samples_per_signal} samples)")
            print(f"  May need to downsample or adjust model input size")
        else:
            print(f"✓ Signal length looks reasonable")

        print(f"\n{'='*60}")
        print(f"✓ File format appears compatible with run.py")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"✗ ERROR loading file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_csv_format.py <csv_file_or_directory>")
        print("\nExamples:")
        print("  python test_csv_format.py test.csv")
        print("  python test_csv_format.py ./data/")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        # Test single file
        test_csv_file(path)
    elif path.is_dir():
        # Test all CSV files in directory
        csv_files = list(path.glob("*.csv"))

        if not csv_files:
            print(f"✗ No CSV files found in {path}")
            sys.exit(1)

        print(f"Found {len(csv_files)} CSV file(s)")

        success_count = 0
        for csv_file in csv_files:
            if test_csv_file(csv_file):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Summary: {success_count}/{len(csv_files)} files passed")
        print(f"{'='*60}")
    else:
        print(f"✗ Path not found: {path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
