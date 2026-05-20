export interface Pick {
  id:           string;
  scan_line_id: string;
  trace_idx:    number;
  x_ft:         number;
  y_ft:         number;
  z_ft:         number;
  depth_in:     number;
  time_ns:      number;
  sample_idx:   number;
  amplitude:    number;
  confidence:   number;
  lat:          number | null;
  lon:          number | null;
  is_edited:    boolean;
  is_deleted:   boolean;
}

export interface ScanLineMeta {
  id:     string;
  label:  string;
  y_ft:   number;
  points: [number, number][];
}

export interface ScanLineTraces {
  id:               string;
  label:            string;
  n_traces:         number;
  n_samples:        number;
  samples_per_ns:   number;
  trace_spacing_ft: number;
  pick_ids:         string[];
  data:             number[][];
}

export interface Scene {
  job_id:            string;
  project_name:      string;
  deck_thickness_in: number;
  bbox:              { min: [number, number]; max: [number, number] };
  surface:           {
    texture_url:    string;
    depth_range_in: [number, number];
    subdivisions:   { nx: number; ny: number };
  };
  scan_lines:        ScanLineMeta[];
  samples_per_ns:    number;
  epsilon_r:         number;
  picks?:            Pick[];
}

export type FilterType =
  | 'bandpass' | 'background_removal' | 'gain' | 'agc' | 'hilbert';

export interface FilterStep {
  id:      string;
  type:    FilterType;
  enabled: boolean;
  params:  Record<string, number | string | boolean>;
}

export interface ProcessingConfig {
  time_zero_shifts: Record<string, number>;
  filters:          FilterStep[];
  gps_latency_ms:   number;
}

export type GriddingAlgorithm =
  | 'nearest_neighbor' | 'idw' | 'natural_neighbor'
  | 'minimum_curvature' | 'kriging';

export interface GriddingConfig {
  algorithm:        GriddingAlgorithm;
  search_radius_ft: number;
  edge_clip:        boolean;
  cell_size_ft:     number;
  anisotropy:       { angle_deg: number; ratio: number };
}

export type ViewMode = 'top' | 'three_d' | 'fixed';

export interface CameraState {
  position: [number, number, number];
  target:   [number, number, number];
  mode:     ViewMode;
}
