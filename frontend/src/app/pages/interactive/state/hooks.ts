import useSWR, { mutate } from 'swr';
import { useEffect } from 'react';
import {
  griddingKey, interactiveApi, interactiveFetcher,
  picksKey, procKey, sceneKey, scanLineKey,
} from './api';
import type {
  GriddingConfig, Pick, ProcessingConfig, Scene, ScanLineTraces,
} from './types';
import { useInteractiveStore } from './useInteractiveStore';

export function useScene(jobId: string | undefined) {
  const { data, error, isLoading } = useSWR<Scene>(
    jobId ? sceneKey(jobId) : null,
    interactiveFetcher,
  );
  const setPicks = useInteractiveStore(s => s.setPicks);
  useEffect(() => { if (data?.picks) setPicks(data.picks); }, [data, setPicks]);
  return { scene: data, error, isLoading };
}

export function usePicks(jobId: string | undefined) {
  const { data, error, isLoading } = useSWR<{ picks: Pick[] }>(
    jobId ? picksKey(jobId) : null,
    interactiveFetcher,
  );
  const setPicks = useInteractiveStore(s => s.setPicks);
  useEffect(() => { if (data?.picks) setPicks(data.picks); }, [data, setPicks]);
  return { picks: data?.picks ?? [], error, isLoading };
}

export function useScanLine(jobId: string | undefined, scanLineId: string | undefined) {
  const { data, error, isLoading } = useSWR<ScanLineTraces>(
    jobId && scanLineId ? scanLineKey(jobId, scanLineId) : null,
    interactiveFetcher,
  );
  return { traces: data, error, isLoading };
}

export function useProcessing(jobId: string | undefined) {
  return useSWR<ProcessingConfig>(jobId ? procKey(jobId) : null, interactiveFetcher);
}

export function useGridding(jobId: string | undefined) {
  return useSWR<GriddingConfig>(jobId ? griddingKey(jobId) : null, interactiveFetcher);
}

export async function patchPickAndRevalidate(jobId: string, id: string, patch: Partial<Pick>) {
  const { pick } = await interactiveApi.patchPick(id, patch);
  useInteractiveStore.getState().upsertPick(pick);
  mutate(picksKey(jobId));
  return pick;
}

export async function saveProcessing(jobId: string, cfg: ProcessingConfig) {
  const next = await interactiveApi.saveProcessing(jobId, cfg);
  mutate(procKey(jobId), next, false);
  return next;
}

export async function saveGridding(jobId: string, cfg: GriddingConfig) {
  const next = await interactiveApi.saveGridding(jobId, cfg);
  mutate(griddingKey(jobId), next, false);
  return next;
}
