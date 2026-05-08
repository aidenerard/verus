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
  amplitude_image?:    string;
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
  amplitude_image_url?:     string;
  rebar_depth_grid?:        (number | null)[][];
  rebar_twt_grid?:          (number | null)[][];
  rebar_peak_grid?:         (number | null)[][];
  // v4+ canvas grid data (JSON arrays, pre-downsampled)
  prob_grid_data?:          (number | null)[][];
  amplitude_grid_data?:     (number | null)[][];
}

export interface UploadedFile { file: File; name: string }

export type OutputTab = 'condition' | 'rebar_depth' | 'amplitude' | 'gps';
