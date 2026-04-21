"""
ingest_gpr_data.py
──────────────────
Converts raw GPR data files from external sources into a flat CSV format
compatible with cnn.py training (or further conversion to SDNET2021 format
via combine_bscan_parts.py).

Output format per file (max 1000 rows, no header):
    512 integer amplitude columns  (DC=32,768, peak+-3,000, clipped [0,65535])
    + 1 label column               (1=sound, 2=delaminated)

CLI usage:
    python ingest_gpr_data.py \\
        --company gatech_analyst \\
        --input  ~/Desktop/verus/raw_data/gatech/ \\
        --output ~/Desktop/verus/gt_bridges/ \\
        --layout layout_sheet.csv \\
        --annotations rebar_depth_maps.csv

Module usage:
    from ingest_gpr_data import GPRConverter
    converter = GPRConverter("gatech_analyst", input_dir, output_dir,
                             layout_sheet=Path("layout.csv"),
                             annotation_map=Path("annotations.csv"))
    results = converter.convert()
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
N_SAMPLES     = 512
DC_OFFSET     = 32_768
MAX_COUNTS    = 3_000
MAX_PER_FILE  = 1_000
LABEL_SOUND   = 1
LABEL_DELAM   = 2

SUPPORTED_FORMATS = {".csv", ".dzt", ".dt1", ".rd3", ".sgr", ".dat"}


# ── Shared utilities ───────────────────────────────────────────────────────────

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize a 1-D signal array to DC=32,768, peak+-3,000, clipped [0,65535].

        1. Remove DC by centering on 32,768
        2. Scale peak to +- MAX_COUNTS counts
        3. Round and clip to uint16 range
    """
    sig = signal.astype(np.float64)
    sig = sig - sig.mean() + DC_OFFSET
    peak = np.abs(sig - DC_OFFSET).max()
    if peak > 0:
        sig = (sig - DC_OFFSET) / peak * MAX_COUNTS + DC_OFFSET
    return np.clip(np.round(sig), 0, 65535).astype(np.int32)


def save_flat_csv(signals: np.ndarray, labels: np.ndarray,
                  out_dir: Path, file_prefix: str = "FILE____") -> int:
    """
    Write (signals, labels) as FILE____NNN.csv chunks of max MAX_PER_FILE rows.
    No header. 512 amplitude columns + 1 label column. Integer dtype.
    Returns the number of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for start in range(0, len(signals), MAX_PER_FILE):
        end   = min(start + MAX_PER_FILE, len(signals))
        chunk = np.column_stack([
            signals[start:end].astype(np.int32),
            labels[start:end].astype(np.int32),
        ])
        n_files += 1
        np.savetxt(out_dir / f"{file_prefix}{n_files:03d}.csv",
                   chunk, delimiter=",", fmt="%d")
    return n_files


# ── Base adapter ───────────────────────────────────────────────────────────────

class BaseAdapter:
    def __init__(self, input_dir: Path, output_dir: Path,
                 layout_sheet: Optional[Path] = None,
                 annotation_map: Optional[Path] = None):
        self.input_dir      = input_dir
        self.output_dir     = output_dir
        self.layout_sheet   = layout_sheet
        self.annotation_map = annotation_map

    def convert(self) -> dict:
        raise NotImplementedError


# ── Adapter 1: GT Bridge Data ──────────────────────────────────────────────────

class GaTechAnalystAdapter(BaseAdapter):
    """
    Handles GT bridge scan CSV files from the gatech_analyst data source.

    Layout sheet (CSV)
    ------------------
    Expected columns: channel, col_start, col_end, offset_cm
        channel   : 1 or 2 — which antenna channel
        col_start : first column index in the raw CSV (0-based)
        col_end   : last column index in the raw CSV (0-based, inclusive)
        offset_cm : antenna offset from curb (metadata only, not used in output)

    Annotation map (CSV)
    --------------------
    Expected columns: file, pos_start, pos_end, label
        file      : raw filename (basename) this row applies to
        pos_start : first signal index of the zone (0-based)
        pos_end   : last signal index of the zone (0-based, inclusive)
        label     : 'sound' | 'delaminated' | 1 | 2

    File naming convention
    ----------------------
    Filenames ending in _r (before the extension) are scanned in reverse
    direction and are flipped before processing.
    """

    def convert(self) -> dict:
        layout      = self._load_layout()
        annotations = self._load_annotations()

        input_files = sorted(
            f for f in self.input_dir.rglob("*")
            if f.suffix.lower() in SUPPORTED_FORMATS
        )
        if not input_files:
            print(f"  WARNING: no supported files found in {self.input_dir}",
                  flush=True)
            return {"error": "no input files"}

        summary = {
            "files_processed":     0,
            "files_reversed":      0,
            "ch1_sound":           0,
            "ch1_delam":           0,
            "ch2_sound":           0,
            "ch2_delam":           0,
            "manual_review_flags": 0,
            "ch1_output_dir":      str(self.output_dir / "gt_bridges_ch1"),
            "ch2_output_dir":      str(self.output_dir / "gt_bridges_ch2"),
        }

        all_ch1_sigs,   all_ch1_labels = [], []
        all_ch2_sigs,   all_ch2_labels = [], []

        for fpath in input_files:
            print(f"  Processing {fpath.name} …", flush=True)

            # ── Load raw data ──────────────────────────────────────────────────
            try:
                df = pd.read_csv(fpath, header=None)
            except Exception as exc:
                print(f"    SKIP — could not read: {exc}", flush=True)
                continue

            # ── Reverse if _r suffix ───────────────────────────────────────────
            if fpath.stem.endswith("_r"):
                df = df.iloc[::-1].reset_index(drop=True)
                summary["files_reversed"] += 1
                print(f"    Reversed (_r suffix detected)", flush=True)

            # ── Split into channel 1 and channel 2 ────────────────────────────
            ch1_raw, ch2_raw = self._split_channels(df, layout)

            # ── Extract and normalize signals ─────────────────────────────────
            ch1_sigs = self._extract_signals(ch1_raw)
            ch2_sigs = self._extract_signals(ch2_raw)

            # ── Assign labels from annotation map ─────────────────────────────
            ch1_labels, ch1_flagged = self._assign_labels(
                fpath.name, len(ch1_sigs), annotations)
            ch2_labels, ch2_flagged = self._assign_labels(
                fpath.name, len(ch2_sigs), annotations)

            summary["manual_review_flags"] += ch1_flagged + ch2_flagged

            all_ch1_sigs.append(ch1_sigs)
            all_ch1_labels.append(ch1_labels)
            all_ch2_sigs.append(ch2_sigs)
            all_ch2_labels.append(ch2_labels)

            summary["files_processed"] += 1
            flag_note = "  [flagged for review]" if (ch1_flagged or ch2_flagged) else ""
            print(f"    CH1: {len(ch1_sigs):,} signals  "
                  f"CH2: {len(ch2_sigs):,} signals{flag_note}", flush=True)

        if not all_ch1_sigs and not all_ch2_sigs:
            print("  No signals extracted.", flush=True)
            return summary

        # ── Concatenate and save ───────────────────────────────────────────────
        if all_ch1_sigs:
            X1 = np.concatenate(all_ch1_sigs)
            y1 = np.concatenate(all_ch1_labels)
            summary["ch1_sound"] = int((y1 == LABEL_SOUND).sum())
            summary["ch1_delam"] = int((y1 == LABEL_DELAM).sum())
            n1 = save_flat_csv(X1, y1, self.output_dir / "gt_bridges_ch1")
            print(f"\n  CH1 saved: {n1} file(s) → {summary['ch1_output_dir']}"
                  f"  sound={summary['ch1_sound']:,}  "
                  f"delam={summary['ch1_delam']:,}", flush=True)

        if all_ch2_sigs:
            X2 = np.concatenate(all_ch2_sigs)
            y2 = np.concatenate(all_ch2_labels)
            summary["ch2_sound"] = int((y2 == LABEL_SOUND).sum())
            summary["ch2_delam"] = int((y2 == LABEL_DELAM).sum())
            n2 = save_flat_csv(X2, y2, self.output_dir / "gt_bridges_ch2")
            print(f"  CH2 saved: {n2} file(s) → {summary['ch2_output_dir']}"
                  f"  sound={summary['ch2_sound']:,}  "
                  f"delam={summary['ch2_delam']:,}", flush=True)

        if summary["manual_review_flags"]:
            print(f"\n  *** {summary['manual_review_flags']:,} signal group(s) "
                  f"flagged for manual label review "
                  f"(no annotation found or parse error) ***", flush=True)

        return summary

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_layout(self) -> Optional[pd.DataFrame]:
        """Load channel layout sheet; warn and return None if missing."""
        if self.layout_sheet and self.layout_sheet.exists():
            df = pd.read_csv(self.layout_sheet)
            required = {"channel", "col_start", "col_end"}
            if not required.issubset(df.columns):
                print(f"  WARNING: layout sheet missing columns {required - set(df.columns)}. "
                      "Falling back to column split.", flush=True)
                return None
            print(f"  Layout sheet loaded: {self.layout_sheet.name}  "
                  f"({len(df)} channel entries)", flush=True)
            return df
        print("  WARNING: no layout sheet — assuming first half of columns = "
              "CH1, second half = CH2.", flush=True)
        return None

    def _load_annotations(self) -> Optional[pd.DataFrame]:
        """Load annotation map; warn and return None if missing."""
        if self.annotation_map and self.annotation_map.exists():
            df = pd.read_csv(self.annotation_map)
            required = {"file", "pos_start", "pos_end", "label"}
            if not required.issubset(df.columns):
                print(f"  WARNING: annotation map missing columns "
                      f"{required - set(df.columns)}. "
                      "All signals will be labelled sound.", flush=True)
                return None
            print(f"  Annotation map loaded: {self.annotation_map.name}  "
                  f"({len(df)} entries)", flush=True)
            return df
        print("  WARNING: no annotation map — all signals labelled sound (1) "
              "and flagged for manual review.", flush=True)
        return None

    def _split_channels(self, df: pd.DataFrame,
                        layout: Optional[pd.DataFrame]) -> tuple:
        """
        Split raw DataFrame rows into CH1 and CH2 column subsets.
        Uses layout sheet if available; falls back to halving columns.
        """
        if layout is not None:
            try:
                ch1_row = layout[layout["channel"] == 1].iloc[0]
                ch2_row = layout[layout["channel"] == 2].iloc[0]
                c1 = slice(int(ch1_row["col_start"]), int(ch1_row["col_end"]) + 1)
                c2 = slice(int(ch2_row["col_start"]), int(ch2_row["col_end"]) + 1)
                return df.iloc[:, c1], df.iloc[:, c2]
            except (KeyError, IndexError) as exc:
                print(f"    WARNING: layout parse error ({exc}), "
                      "falling back to column halving.", flush=True)

        mid = len(df.columns) // 2
        return df.iloc[:, :mid], df.iloc[:, mid:]

    def _extract_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Pull N_SAMPLES numeric columns from each row, normalize to DC/peak
        convention, return (n_rows, N_SAMPLES) int32.

        Rows with fewer than N_SAMPLES numeric values are zero-padded on
        the right. Rows with more are trimmed to the first N_SAMPLES.
        """
        numeric = df.select_dtypes(include=[np.number])
        arr = numeric.values.astype(np.float64)
        n_rows, n_cols = arr.shape

        if n_cols < N_SAMPLES:
            arr = np.hstack([arr,
                             np.zeros((n_rows, N_SAMPLES - n_cols),
                                      dtype=np.float64)])
        elif n_cols > N_SAMPLES:
            arr = arr[:, :N_SAMPLES]

        out = np.empty((n_rows, N_SAMPLES), dtype=np.int32)
        for i in range(n_rows):
            out[i] = normalize_signal(arr[i])
        return out

    def _assign_labels(self, filename: str, n_signals: int,
                       annotations: Optional[pd.DataFrame]) -> tuple[np.ndarray, int]:
        """
        Build per-signal label array from the annotation map.

        If no annotation rows match this file, all signals are labelled
        sound (1) and the entire file is counted as flagged for review.

        Returns (labels: np.ndarray[int32], n_flagged: int).
        """
        labels  = np.full(n_signals, LABEL_SOUND, dtype=np.int32)
        flagged = 0

        if annotations is None:
            return labels, n_signals

        file_rows = annotations[
            annotations["file"].astype(str).str.contains(
                Path(filename).name, regex=False, na=False)
        ]

        if file_rows.empty:
            # No annotation for this file — label sound, flag entire file
            return labels, n_signals

        for _, row in file_rows.iterrows():
            try:
                p0        = int(row["pos_start"])
                p1        = int(row["pos_end"]) + 1   # exclusive upper bound
                raw_label = str(row["label"]).strip().lower()
                lbl       = (LABEL_DELAM
                             if raw_label in ("delaminated", "delam", "2")
                             else LABEL_SOUND)
                p0 = max(0, p0)
                p1 = min(n_signals, p1)
                if p0 < p1:
                    labels[p0:p1] = lbl
            except (KeyError, ValueError, TypeError) as exc:
                print(f"    WARNING: annotation row parse error ({exc}), "
                      "skipping.", flush=True)
                flagged += 1

        return labels, flagged


# ── Adapter 2: Generic CSV ─────────────────────────────────────────────────────

class GenericCSVAdapter(BaseAdapter):
    """
    Accepts any CSV file where rows are signals and columns are numeric samples.

    Auto-detection heuristic: keeps columns with > 100 unique values (likely
    signal amplitude data) and discards metadata columns (IDs, flags, etc.).
    Requires at least N_SAMPLES qualifying columns.

    All output signals are labelled 1 (sound) pending manual annotation.
    """

    def convert(self) -> dict:
        input_files = sorted(self.input_dir.rglob("*.csv"))
        if not input_files:
            print(f"  WARNING: no CSV files found in {self.input_dir}",
                  flush=True)
            return {"error": "no input files"}

        summary = {
            "files_processed": 0,
            "files_skipped":   0,
            "total_signals":   0,
            "output_dir":      str(self.output_dir / "generic_csv"),
        }

        all_sigs = []

        for fpath in input_files:
            print(f"  Processing {fpath.name} …", flush=True)
            try:
                df = pd.read_csv(fpath, header=None)
            except Exception as exc:
                print(f"    SKIP — could not read: {exc}", flush=True)
                summary["files_skipped"] += 1
                continue

            # Keep only numeric columns with enough value spread to be signal
            numeric      = df.select_dtypes(include=[np.number])
            signal_cols  = [c for c in numeric.columns
                            if numeric[c].nunique() > 100]

            if len(signal_cols) < N_SAMPLES:
                print(f"    SKIP — only {len(signal_cols)} signal-like columns "
                      f"(need >= {N_SAMPLES})", flush=True)
                summary["files_skipped"] += 1
                continue

            arr  = numeric[signal_cols[:N_SAMPLES]].values.astype(np.float64)
            sigs = np.empty((len(arr), N_SAMPLES), dtype=np.int32)
            for i in range(len(arr)):
                sigs[i] = normalize_signal(arr[i])

            all_sigs.append(sigs)
            summary["files_processed"] += 1
            print(f"    {len(sigs):,} signals extracted", flush=True)

        if not all_sigs:
            print("  No signals extracted.", flush=True)
            return summary

        X   = np.concatenate(all_sigs)
        y   = np.full(len(X), LABEL_SOUND, dtype=np.int32)
        out = self.output_dir / "generic_csv"
        n   = save_flat_csv(X, y, out)

        summary["total_signals"] = len(X)
        print(f"\n  {n} file(s) saved → {out}  "
              f"({len(X):,} signals, all labelled sound — "
              "manual annotation required)", flush=True)
        return summary


# ── Adapter 3: GSSI DZT (placeholder) ─────────────────────────────────────────

class GSSIAdapter(BaseAdapter):
    """Placeholder for future GSSI hardware file support."""

    def convert(self) -> dict:
        print("  GSSI DZT format support coming in V2", flush=True)
        return {"status": "not_implemented"}


# ── Adapter registry ───────────────────────────────────────────────────────────

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "gatech_analyst": GaTechAnalystAdapter,
    "generic_csv":    GenericCSVAdapter,
    "gssi":           GSSIAdapter,
}


# ── GPRConverter — public facade ───────────────────────────────────────────────

class GPRConverter:
    """
    Routes conversion to the appropriate company-specific adapter.

    Parameters
    ----------
    company        : 'gatech_analyst' | 'generic_csv' | 'gssi'
    input_dir      : directory containing raw GPR files
    output_dir     : root output directory (subdirs created per adapter)
    layout_sheet   : (gatech_analyst) path to channel layout CSV
    annotation_map : (gatech_analyst) path to delamination annotation CSV
    """

    def __init__(self, company: str, input_dir: Path, output_dir: Path,
                 layout_sheet: Optional[Path] = None,
                 annotation_map: Optional[Path] = None):
        key = company.lower()
        if key not in ADAPTERS:
            raise ValueError(
                f"Unknown company '{company}'. "
                f"Supported: {sorted(ADAPTERS.keys())}"
            )
        self._company = key
        self._adapter = ADAPTERS[key](
            input_dir      = Path(input_dir),
            output_dir     = Path(output_dir),
            layout_sheet   = Path(layout_sheet)   if layout_sheet   else None,
            annotation_map = Path(annotation_map) if annotation_map else None,
        )

    def convert(self) -> dict:
        """Run the conversion pipeline and return a summary dict."""
        print("=" * 60, flush=True)
        print(f"GPR Ingest  company={self._company}", flush=True)
        print(f"  Input  : {self._adapter.input_dir}", flush=True)
        print(f"  Output : {self._adapter.output_dir}", flush=True)
        print("=" * 60, flush=True)
        results = self._adapter.convert()
        self._print_summary(results)
        return results

    def _auto_detect_format(self, fpath: Path) -> str:
        """
        Detect file format from extension; sniff content for ambiguous cases.
        Returns the lower-case extension string or 'unknown'.
        """
        ext = fpath.suffix.lower()
        if ext in SUPPORTED_FORMATS:
            return ext
        try:
            with open(fpath, "rb") as f:
                head = f.read(512)
            if b"," in head or b"\t" in head:
                return ".csv"
        except OSError:
            pass
        return "unknown"

    def _print_summary(self, results: dict) -> None:
        print(f"\n{'─'*60}", flush=True)
        print("CONVERSION SUMMARY", flush=True)
        print(f"{'─'*60}", flush=True)
        for key, val in results.items():
            print(f"  {key.replace('_', ' ').title():<30} {val}", flush=True)
        print(f"{'─'*60}", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert raw GPR data to flat CSV format for cnn.py training.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--company", required=True, choices=sorted(ADAPTERS.keys()),
                   help="Company adapter to use.")
    p.add_argument("--input",  required=True,
                   help="Directory containing raw GPR input files.")
    p.add_argument("--output", required=True,
                   help="Root output directory for converted files.")
    p.add_argument("--layout",
                   help="(gatech_analyst) Channel layout sheet CSV path.")
    p.add_argument("--annotations",
                   help="(gatech_analyst) Delamination annotation map CSV path.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    converter = GPRConverter(
        company        = args.company,
        input_dir      = Path(args.input).expanduser(),
        output_dir     = Path(args.output).expanduser(),
        layout_sheet   = Path(args.layout).expanduser()      if args.layout      else None,
        annotation_map = Path(args.annotations).expanduser() if args.annotations else None,
    )
    results = converter.convert()
    sys.exit(0 if "error" not in results else 1)
