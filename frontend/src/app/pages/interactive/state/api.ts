import { SERVER } from '../../inspect/constants';
import type {
  GriddingConfig, Pick, ProcessingConfig, Scene, ScanLineTraces,
} from './types';

interface ReprocessAck { job_id: string; status: string }
interface RegridAck    { job_id: string; status: string; texture_url?: string }

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${SERVER}${url}`, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const interactiveFetcher = <T,>(url: string) => getJson<T>(url);

export const interactiveApi = {
  scene:      (jobId: string)        => getJson<Scene>(`/jobs/${jobId}/scene`),
  picks:      (jobId: string)        => getJson<{ picks: Pick[] }>(`/jobs/${jobId}/picks`),
  scanLine:   (jobId: string, sl: string) =>
    getJson<ScanLineTraces>(`/jobs/${jobId}/scan_line/${sl}`),
  processing: (jobId: string)        => getJson<ProcessingConfig>(`/jobs/${jobId}/processing`),
  gridding:   (jobId: string)        => getJson<GriddingConfig>(`/jobs/${jobId}/gridding`),

  patchPick: (id: string, patch: Partial<Pick>) =>
    getJson<{ pick: Pick }>(`/picks/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }),

  saveProcessing: (jobId: string, cfg: ProcessingConfig) =>
    getJson<ProcessingConfig>(`/jobs/${jobId}/processing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    }),

  saveGridding: (jobId: string, cfg: GriddingConfig) =>
    getJson<GriddingConfig>(`/jobs/${jobId}/gridding`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    }),

  reprocess: (jobId: string) =>
    getJson<ReprocessAck>(`/jobs/${jobId}/reprocess`, { method: 'POST' }),
  regrid: (jobId: string) =>
    getJson<RegridAck>(`/jobs/${jobId}/regrid`, { method: 'POST' }),
};

export const sceneKey     = (id: string)            => `/jobs/${id}/scene`;
export const picksKey     = (id: string)            => `/jobs/${id}/picks`;
export const scanLineKey  = (id: string, sl: string)=> `/jobs/${id}/scan_line/${sl}`;
export const procKey      = (id: string)            => `/jobs/${id}/processing`;
export const griddingKey  = (id: string)            => `/jobs/${id}/gridding`;
