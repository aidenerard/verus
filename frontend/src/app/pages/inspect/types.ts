export interface BscanData { data: string; n_traces: number; n_samples: number }

export interface GpsData {
  lat_start: number; lon_start: number;
  lat_end: number;   lon_end: number;
  coordinates: [number, number][];
}

export interface FileResult {
  filename:          string;
  signals:           number;
  delam_pct:         number;
  gps?:              GpsData | null;
  bscan?:            BscanData;
  rebar_depth_mean?: number;
  rebar_depth_min?:  number;
  rebar_depth_max?:  number;
  rebar_depth_array?:  number[];
  twt_array?:          number[];
  peak_sample_array?:  number[];
}

export interface DeckQuantities {
  n_picks:               number;
  scan_spacing_ft?:      number;
  mean_cover_in:         number;
  std_cover_in?:         number;
  min_cover_in:          number;
  max_cover_in:          number;
  cover_below_2in_pct?:  number;
  deteriorated_pct:      number;
  sound_pct:             number;
  astm_method:           string;
  astm_status:           string;
  deterioration_method?: string;
  threshold_db?:         number;
  threshold_note?:       string;
  mean_dielectric?:      number;
  high_moisture_pct?:    number;
}

export interface AnalysisResult {
  signals_analyzed:    number;
  delamination_pct:    number;
  sound_pct:           number;
  analysis_time_sec:   number;
  cscan_image?:        string;
  cscan_url?:          string;
  manufacturer?:       string;
  per_file_summary:    FileResult[];
  // optional — present only from server v2+
  rebar_depth_image?:  string;
  prob_grid?:          string;
  prob_grid_rows?:     number;
  prob_grid_cols?:     number;
  otsu_threshold?:     number;
  twt_grid?:           string;
  twt_grid_rows?:      number;
  twt_grid_cols?:      number;
  frequency_mhz?:      number;
  model_confidence_pct?: number;
  depth_accuracy_in?:  number;
  signal_quality?:     string;
  // rebar model fields — present from server v3+
  rebar_model_used?:        boolean;
  rebar_cscan_image?:       string;
  rebar_cscan_image_url?:   string;
  rebar_depth_grid?:        (number | null)[][];
  rebar_twt_grid?:          (number | null)[][];
  rebar_peak_grid?:         (number | null)[][];
  // v4+ canvas grid data (JSON arrays, pre-downsampled)
  prob_grid_data?:          (number | null)[][];
  // v5+ simplified 3-panel result keys
  horizon_picks?:           string;
  rebar_depth_map?:         string;
  corrosion_map?:           string;
  bscan_data?:              Array<{      // per-swath raw trace blobs for canvas viewer
    data:      string;                    // base64 + zlib + int8
    n_traces:  number;
    n_samples: number;
    encoding:  string;
  }>;
  bscan_count?:             number;
  // v6+ large blobs offloaded to the job-results storage bucket (URLs)
  rebar_depth_map_url?:     string;
  corrosion_map_url?:       string;
  horizon_picks_url?:       string;
  rebar_depth_image_url?:   string;
  bscan_data_url?:          string;
  mean_depth_inches?:       number;
  deck_thickness_inches?:   number;
  high_risk_pct?:           number;
  high_moisture_pct?:       number | null;
  // Dielectric / moisture map — null until a metal-plate calibration scan
  // provides real per-trace dielectric (see dielectric_map_unavailable_reason).
  dielectric_map?:          string | null;
  dielectric_map_url?:      string | null;
  dielectric_map_unavailable_reason?: string;
  // ASTM — method label only, never a pass/fail compliance boolean.
  astm_method?:             string;
  astm_status?:             string;
  quantities?:              DeckQuantities | null;
  // Proceq pipeline stats sub-object
  stats?: {
    n_traces?:      number;
    mean_depth_in?: number;
    high_risk_pct?: number;
    quantities?:    DeckQuantities | null;
  };
  condition_class_pcts?: {
    sound:                number;
    monitor:              number;
    anomalous_response:   number;
    significant_anomaly:  number;
  };
  // User-supplied metadata, mirrored from the analysis_jobs row
  analysis_name?:           string;
  analysis_notes?:          string;
  company?:                 string;
  project?:                 string;
}

export interface UploadedFile { file: File; name: string }

export type OutputTab = 'condition' | 'rebar_depth' | 'amplitude' | 'gps';
