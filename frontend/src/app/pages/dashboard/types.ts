import type { ComponentType } from 'react';

export interface AnalysisJob {
  id: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  created_at: string;
  completed_at?: string;
  signals_analyzed?: number;
  delamination_pct?: number;
  sound_pct?: number;
  analysis_time_sec?: number;
  cscan_url?: string;
  file_names?: string[];
  error_msg?: string;
  project_id?: string;
  project_name?: string;
}

export interface InspectionModule {
  id: string;
  name: string;
  fullName: string;
  status: 'available' | 'in-development';
  icon: ComponentType<{ className?: string; style?: React.CSSProperties }>;
  standard: string;
  description: string;
}
