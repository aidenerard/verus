/**
 * ThreeDTypes.ts
 * Shared interfaces and layout constants for the 3D bridge-deck visualisation.
 *
 * Does NOT: contain any React components, Three.js logic, or rendering code.
 */

export interface FileResult {
  filename: string;
  signals: number;
  delam_pct: number;
}

export interface ThreeDViewProps {
  perFileSummary: FileResult[];
}

export interface TooltipState {
  x: number;
  y: number;
  lines: string[];
}

export const BG_COLOR  = '#0a1628';
export const CELL_W    = 1.0;
export const CELL_H    = 0.18;
export const CELL_D    = 1.0;
export const GAP       = 0.08;
export const GRID_ROWS = 24;  // visual rows per scan line (depth axis)
export const HEIGHT_PX = 500;
