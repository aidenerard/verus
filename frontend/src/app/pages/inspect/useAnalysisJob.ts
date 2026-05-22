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

      setStatusMsg('Uploading files…');
      const hasProceqFiles = files.some(f => f.file.name.toLowerCase().endsWith('.scan'));
      const endpoint = hasProceqFiles ? '/analyze-proceq' : '/analyze';

      const formData = new FormData();
      files.forEach(f => formData.append('files', f.file));
      if (hasProceqFiles) {
        // /analyze-proceq takes epsr only; manufacturer/frequency/project_id don't apply.
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

      const { job_id } = await res.json();
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
          const sr = await fetch(`${SERVER}/job/${job_id}/status`, { headers });
          if (!sr.ok) return;
          const s = await sr.json();
          setJobProgress(s.progress ?? 0);
          setJobStage(s.stage ?? '');
          if (s.status === 'complete') {
            clearInterval(pollRef.current!);
            const jr = await fetch(`${SERVER}/job/${job_id}`, { headers });
            if (!jr.ok) { setErrorMsg(`HTTP ${jr.status}`); setJobStatus('failed'); return; }
            const job = await jr.json();
            const result: AnalysisResult = job.result ?? (job.signals_analyzed != null ? job : null);
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
  }, [files, session, jobStatus, manufacturer, frequencyMhz, useCustomFreq, customFreq, projectId, onComplete, extraFormData]);

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
