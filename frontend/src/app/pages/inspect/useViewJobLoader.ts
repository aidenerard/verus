/**
 * useViewJobLoader.ts — two startup-only effects that hydrate workspace state.
 * Effect 1: restore project config from a saved localStorage project_id.
 * Effect 2: load a completed job from the URL ?project_id= param.
 * Does NOT manage ongoing analysis polling — that lives in useAnalysisJob.
 */

import { useEffect } from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from '../../../lib/supabase';
import { SERVER, DEFAULT_ER } from './constants';
import type { ManufacturerKey } from './constants';
import type { AnalysisResult, OutputTab } from './types';
import type { ImperativePanelHandle } from 'react-resizable-panels';

interface SetupSetters {
  setSetupChecking: (v: boolean) => void;
  setSetupDone:     (v: boolean) => void;
  setProjectId:     (v: string | null) => void;
  setManufacturer:  (v: ManufacturerKey | '') => void;
  setFrequencyMhz:  (v: number) => void;
  setDielectricEr:  (v: number) => void;
  setProjectName:   (v: string) => void;
  setStructureName: (v: string) => void;
  setBridgeId:      (v: string) => void;
  setInspDate:      (v: string) => void;
  setNotes:         (v: string) => void;
}

interface JobSetters {
  setAnalysisResult:    (v: AnalysisResult | null) => void;
  setJobStatus:         (v: 'idle'|'pending'|'processing'|'complete'|'failed') => void;
  setOutputTab:         (v: OutputTab) => void;
  setRightIconOpen:     (v: 'properties'|'analysis'|'adjust'|null) => void;
  setActiveView:        (v: 'cscan'|'3d') => void;
  setDetectionThreshold:(v: number) => void;
}

interface UseViewJobLoaderOptions {
  viewJobId: string | null;
  session:   Session | null;
  setupSetters: SetupSetters;
  jobSetters:   JobSetters;
  bottomPanelRef: React.RefObject<ImperativePanelHandle | null>;
}

export function useViewJobLoader({
  viewJobId, session, setupSetters, jobSetters, bottomPanelRef,
}: UseViewJobLoaderOptions): void {
  const { setSetupChecking, setSetupDone, setProjectId, setManufacturer, setFrequencyMhz,
          setDielectricEr, setProjectName, setStructureName, setBridgeId, setInspDate, setNotes } = setupSetters;
  const { setAnalysisResult, setJobStatus, setOutputTab, setRightIconOpen, setActiveView,
          setDetectionThreshold } = jobSetters;

  // Restore project config from localStorage
  useEffect(() => {
    if (viewJobId) return;
    const pid = localStorage.getItem('verus_project_id');
    if (!pid) { setSetupChecking(false); return; }
    supabase.from('projects').select('*').eq('id', pid).single()
      .then(({ data }) => {
        if (data?.manufacturer) {
          const freq = data.frequency_mhz ?? 1600;
          setProjectId(data.id); setManufacturer(data.manufacturer as ManufacturerKey);
          setFrequencyMhz(freq); setDielectricEr(DEFAULT_ER[freq] ?? 6);
          if (data.name)            setProjectName(data.name);
          if (data.structure_name)  setStructureName(data.structure_name);
          if (data.bridge_id)       setBridgeId(data.bridge_id);
          if (data.inspection_date) setInspDate(data.inspection_date);
          if (data.notes)           setNotes(data.notes);
          setSetupDone(true);
        }
        setSetupChecking(false);
      })
      .catch(() => setSetupChecking(false));
  }, []); // eslint-disable-line

  // Load a completed job from the URL ?project_id= param
  useEffect(() => {
    if (!viewJobId) return;
    const headers: Record<string, string> = {};
    if (session?.access_token) headers['Authorization'] = `Bearer ${session.access_token}`;
    fetch(`${SERVER}/job/${viewJobId}`, { headers })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((job: any) => {
        const result: AnalysisResult = job.result ?? (job.signals_analyzed != null ? {
          signals_analyzed: job.signals_analyzed, delamination_pct: job.delamination_pct ?? 0,
          sound_pct: job.sound_pct ?? 0, analysis_time_sec: job.analysis_time_sec ?? 0,
          cscan_image: '', cscan_url: job.cscan_url, per_file_summary: job.per_file_summary ?? [],
          rebar_cscan_image: '', rebar_cscan_image_url: job.rebar_cscan_image_url,
          amplitude_image: '', amplitude_image_url: job.amplitude_image_url,
          prob_grid: job.prob_grid, prob_grid_rows: job.prob_grid_rows, prob_grid_cols: job.prob_grid_cols,
          otsu_threshold: job.otsu_threshold, twt_grid: job.twt_grid,
          twt_grid_rows: job.twt_grid_rows, twt_grid_cols: job.twt_grid_cols,
          frequency_mhz: job.frequency_mhz, manufacturer: job.manufacturer,
          rebar_model_used: job.rebar_model_used, model_confidence_pct: job.model_confidence_pct,
          depth_accuracy_in: job.depth_accuracy_in, signal_quality: job.signal_quality,
        } : null as unknown as AnalysisResult);
        if (result) {
          setAnalysisResult(result);
          if (job.project_id) {
            setProjectId(job.project_id);
            supabase.from('projects').select('name,structure_name,bridge_id,inspection_date,notes')
              .eq('id', job.project_id).single()
              .then(({ data }) => {
                if (data?.name)            setProjectName(data.name);
                if (data?.structure_name)  setStructureName(data.structure_name);
                if (data?.bridge_id)       setBridgeId(data.bridge_id);
                if (data?.inspection_date) setInspDate(data.inspection_date);
                if (data?.notes)           setNotes(data.notes);
              }).catch(() => {});
          }
          const mfr = result.manufacturer ?? job.manufacturer;
          if (mfr) setManufacturer(mfr as ManufacturerKey);
          const freq = result.frequency_mhz ?? job.frequency_mhz;
          if (freq) { setFrequencyMhz(freq); setDielectricEr(DEFAULT_ER[freq] ?? 6); }
          if (result.otsu_threshold) setDetectionThreshold(result.otsu_threshold);
          setJobStatus('complete'); setOutputTab('condition');
          setRightIconOpen('properties'); setActiveView('cscan');
          bottomPanelRef.current?.expand();
        }
        setSetupDone(true); setSetupChecking(false);
      }).catch(() => setSetupChecking(false));
  }, []); // eslint-disable-line
}
