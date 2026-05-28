/**
 * useAnalysisJob.ts
 * Manages the full analysis job lifecycle: server wake, file upload, polling,
 * progress tracking, and completion/failure handling.
 *
 * Does NOT: manage canvas rendering, map state, UI layout, or setup wizard state.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { Session } from '@supabase/supabase-js';

import {
  SERVER, WAKE_TIMEOUT_MS, WAKE_INTERVAL_MS, POLL_TIMEOUT_MS, DEFAULT_ER,
} from './constants';
import type { ManufacturerKey } from './constants';
import type { AnalysisResult, UploadedFile } from './types';
import { estimateAnalysisSeconds } from './utils';
import { supabase } from '../../../lib/supabase';

interface PollSnapshot {
  status:   string;
  progress: number;
  stage:    string;
  error:    string | null;
  result:   AnalysisResult | null;
}

async function pollFromSupabase(jobId: string): Promise<PollSnapshot | null> {
  const { data, error } = await supabase
    .from('analysis_jobs')
    .select('id, status, progress, stage, error_msg, result')
    .eq('id', jobId)
    .single();
  if (error || !data) return null;
  return {
    status:   data.status,
    progress: data.progress ?? 0,
    stage:    data.stage ?? '',
    error:    data.error_msg ?? null,
    result:   (data.result as AnalysisResult | null) ?? null,
  };
}

async function pollJobStatus(jobId: string, headers: Record<string, string>): Promise<PollSnapshot> {
  // Supabase first (no CORS — direct PostgREST call).
  const s = await pollFromSupabase(jobId);
  if (s && s.status) return s;

  // Fallback: Render server. result blob not included by /status; will be
  // fetched separately on completion via /job/{id} below.
  const r = await fetch(`${SERVER}/job/${jobId}/status`, { headers });
  if (!r.ok) throw new Error(`Job not found: ${r.status}`);
  const raw = await r.json();
  return {
    status:   raw.status,
    progress: raw.progress ?? 0,
    stage:    raw.stage ?? '',
    error:    raw.error ?? null,
    result:   null,
  };
}

interface UseAnalysisJobProps {
  files: UploadedFile[];
  session: Session | null;
  manufacturer: ManufacturerKey | '';
  frequencyMhz: number;
  useCustomFreq: boolean;
  customFreq: string;
  projectId: string | null;
  onComplete: (result: AnalysisResult, otsuThreshold: number | undefined, freq: number | undefined) => void;
  extraFormData?: (fd: FormData) => void;
  analysisName?:  string;
  analysisNotes?: string;
  company?:       string;
  project?:       string;
}

interface UseAnalysisJobReturn {
  jobId: string | null;
  jobStatus: 'idle' | 'pending' | 'processing' | 'complete' | 'failed';
  setJobStatus: React.Dispatch<React.SetStateAction<'idle' | 'pending' | 'processing' | 'complete' | 'failed'>>;
  errorMsg: string | null;
  setErrorMsg: React.Dispatch<React.SetStateAction<string | null>>;
  statusMsg: string;
  showConfirm: boolean;
  setShowConfirm: React.Dispatch<React.SetStateAction<boolean>>;
  estimatedSecs: number;
  setEstimatedSecs: React.Dispatch<React.SetStateAction<number>>;
  showProgressOverlay: boolean;
  jobProgress: number;
  jobStage: string;
  startAnalysis: () => Promise<void>;
  onConfirmAnalysis: () => void;
  isAnalyzing: boolean;
}

export function useAnalysisJob({
  files,
  session,
  manufacturer,
  frequencyMhz,
  useCustomFreq,
  customFreq,
  projectId,
  onComplete,
  extraFormData,
  analysisName,
  analysisNotes,
  company,
  project,
}: UseAnalysisJobProps): UseAnalysisJobReturn {
  const [jobId,               setJobId]               = useState<string | null>(null);
  const [jobStatus,           setJobStatus]           = useState<'idle'|'pending'|'processing'|'complete'|'failed'>('idle');
  const [errorMsg,            setErrorMsg]            = useState<string | null>(null);
  const [statusMsg,           setStatusMsg]           = useState('');
  const [showConfirm,         setShowConfirm]         = useState(false);
  const [estimatedSecs,       setEstimatedSecs]       = useState(15);
  const [showProgressOverlay, setShowProgressOverlay] = useState(false);
  const [jobProgress,         setJobProgress]         = useState(0);
  const [jobStage,            setJobStage]            = useState('');

  const pollRef        = useRef<ReturnType<typeof setInterval> | null>(null);
  const statusCycleRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isAnalyzing = jobStatus === 'pending' || jobStatus === 'processing';

  const startAnalysis = useCallback(async () => {
    if (!files.length || jobStatus === 'pending' || jobStatus === 'processing') return;
    setJobStatus('pending');
    setErrorMsg(null);
    setStatusMsg('Waking up server…');

    const headers: Record<string, string> = {};
    if (session?.access_token) headers['Authorization'] = `Bearer ${session.access_token}`;

    try {
      const wakeDeadline = Date.now() + WAKE_TIMEOUT_MS;
      let serverReady = false;
      let coldChecked = false;
      while (Date.now() < wakeDeadline) {
        const t0 = Date.now();
        try {
          const h = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(10000) });
          if (!coldChecked) {
            coldChecked = true;
            if (Date.now() - t0 > 2000) setEstimatedSecs(prev => prev + 30);
          }
          if (h.ok) { const hj = await h.json(); if (hj.model_loaded) { serverReady = true; break; } setStatusMsg('Loading AI model…'); }
        } catch (_e) {
          if (!coldChecked) { coldChecked = true; if (Date.now() - t0 > 2000) setEstimatedSecs(prev => prev + 30); }
        }
        await new Promise(r => setTimeout(r, WAKE_INTERVAL_MS));
      }
      if (!serverReady) { setErrorMsg('Server did not respond in time.'); setJobStatus('failed'); return; }

      const zipFile = (files ?? []).find(f => f.file.name.toLowerCase().endsWith('.zip'));
      const hasProceqFiles = (files ?? []).some(f => {
        const n = f.file.name.toLowerCase();
        return n.endsWith('.scan') || n.endsWith('.zip');
      });

      let job_id: string;
      if (zipFile) {
        // Direct-to-storage path: bypass Render's request body limit by
        // PUTting the zip straight to the 'uploads' bucket, then handing
        // the backend the storage path. Slug company/project/analysis for
        // a readable storage key.
        const slug = (s: string) =>
          (s || 'unknown').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 30);
        const storagePath =
          `${slug(company ?? '')}/${slug(project ?? '')}/${slug(analysisName ?? '')}-${Date.now()}.zip`;

        const zipMb = zipFile.file.size / 1024 / 1024;
        setStatusMsg(`Uploading zip (${zipMb.toFixed(0)} MB) to storage…`);
        const { error: uploadError } = await supabase.storage
          .from('uploads')
          .upload(storagePath, zipFile.file, { upsert: true, contentType: 'application/zip' });
        if (uploadError) {
          setErrorMsg(`Zip upload failed: ${uploadError.message}`);
          setJobStatus('failed'); return;
        }

        setStatusMsg('Queuing analysis…');
        const fd = new FormData();
        fd.append('storage_path',   storagePath);
        fd.append('epsr',           '9.0');
        fd.append('analysis_name',  (analysisName  ?? '').trim() || 'Untitled Analysis');
        fd.append('analysis_notes', (analysisNotes ?? '').trim());
        fd.append('company',        (company       ?? '').trim());
        fd.append('project',        (project       ?? '').trim());

        const res = await fetch(`${SERVER}/analyze-proceq-zip`, {
          method: 'POST', headers, body: fd, signal: AbortSignal.timeout(60000),
        });
        if (!res.ok) {
          let msg = `HTTP ${res.status}`;
          try { const j = await res.json(); msg = j.detail || j.error || msg; } catch (_e) {}
          setErrorMsg(msg); setJobStatus('failed'); return;
        }
        ({ job_id } = await res.json());
      } else {
        // Standard path: multipart through Render. Used for individual
        // .scan/.pos/.cscan or non-Proceq formats (DZT/SEG-Y/etc.).
        const endpoint = hasProceqFiles ? '/analyze-proceq' : '/analyze';
        setStatusMsg('Uploading files…');

        const formData = new FormData();
        files.forEach(f => formData.append('files', f.file));
        formData.append('analysis_name',  (analysisName  ?? '').trim() || 'Untitled Analysis');
        formData.append('analysis_notes', (analysisNotes ?? '').trim());
        formData.append('company',        (company       ?? '').trim());
        formData.append('project',        (project       ?? '').trim());
        if (hasProceqFiles) {
          formData.append('epsr', '9.0');
        } else {
          if (manufacturer) formData.append('manufacturer', manufacturer);
          const effectiveFreq = useCustomFreq ? (parseInt(customFreq) || 1600) : frequencyMhz;
          formData.append('frequency_mhz', String(effectiveFreq));
          if (projectId) formData.append('project_id', projectId);
          if (extraFormData) extraFormData(formData);
        }

        const res = await fetch(`${SERVER}${endpoint}`, {
          method: 'POST', headers, body: formData, signal: AbortSignal.timeout(60000),
        });
        if (!res.ok) {
          let msg = `HTTP ${res.status}`;
          try { const j = await res.json(); msg = j.detail || j.error || msg; } catch (_e) {}
          setErrorMsg(msg); setJobStatus('failed'); return;
        }
        ({ job_id } = await res.json());
      }

      setJobId(job_id);
      setJobStatus('processing');
      setJobProgress(0);
      setJobStage('Starting…');

      const pollDeadline = Date.now() + POLL_TIMEOUT_MS;
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        if (Date.now() > pollDeadline) {
          clearInterval(pollRef.current!);
          setErrorMsg('Analysis timed out.'); setJobStatus('failed'); return;
        }
        try {
          const s = await pollJobStatus(job_id, headers);
          setJobProgress(s.progress);
          setJobStage(s.stage);
          if (s.status === 'complete') {
            clearInterval(pollRef.current!);
            // Use the result blob from Supabase if it landed there; otherwise
            // fall back to GET /job/{id} on the server.
            let result: AnalysisResult | null = s.result;
            if (!result) {
              const jr = await fetch(`${SERVER}/job/${job_id}`, { headers });
              if (!jr.ok) { setErrorMsg(`HTTP ${jr.status}`); setJobStatus('failed'); return; }
              const job = await jr.json();
              result = job.result ?? (job.signals_analyzed != null ? job : null);
            }
            if (result) {
              setJobStatus('complete');
              setJobProgress(100);
              onComplete(result, result.otsu_threshold, result.frequency_mhz);
            }
          } else if (s.status === 'failed') {
            clearInterval(pollRef.current!);
            setErrorMsg(s.error || 'Analysis failed'); setJobStatus('failed');
          }
        } catch (_e) { /* poll silently */ }
      }, 1000);
    } catch (err) {
      clearInterval(statusCycleRef.current!);
      setErrorMsg(err instanceof Error ? err.message : 'Analysis failed');
      setJobStatus('failed');
    }
  }, [files, session, jobStatus, manufacturer, frequencyMhz, useCustomFreq, customFreq, projectId, onComplete, extraFormData, analysisName, analysisNotes, company, project]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (statusCycleRef.current) clearInterval(statusCycleRef.current);
  }, []);

  const onConfirmAnalysis = useCallback(() => {
    const estSecs = estimateAnalysisSeconds(files.map(f => f.file), manufacturer);
    setEstimatedSecs(estSecs);
    setJobProgress(0);
    setJobStage('');
    setShowConfirm(false);
    setShowProgressOverlay(true);
    startAnalysis();
  }, [files, manufacturer, startAnalysis]);

  // Hide overlay after completion/failure
  useEffect(() => {
    if (jobStatus === 'complete' || jobStatus === 'failed') {
      const t = setTimeout(() => setShowProgressOverlay(false), 800);
      return () => clearTimeout(t);
    }
  }, [jobStatus]);

  return {
    jobId,
    jobStatus,
    setJobStatus,
    errorMsg,
    setErrorMsg,
    statusMsg,
    showConfirm,
    setShowConfirm,
    estimatedSecs,
    setEstimatedSecs,
    showProgressOverlay,
    jobProgress,
    jobStage,
    startAnalysis,
    onConfirmAnalysis,
    isAnalyzing,
  };
}
