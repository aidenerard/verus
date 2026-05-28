import { useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router';
import { Radio, RotateCcw, AlertCircle } from 'lucide-react';
import { useAuth } from '../../../context/AuthContext';
import { supabase } from '../../../lib/supabase';
import { useAnalysisJob } from '../inspect/useAnalysisJob';
import ConfirmAnalysisModal from '../inspect/ConfirmAnalysisModal';
import AnalysisProgressOverlay from '../inspect/AnalysisProgressOverlay';
import type { AnalysisResult, UploadedFile } from '../inspect/types';
import type { ManufacturerKey } from '../inspect/constants';
import GPRUploadCard from './GPRUploadCard';
import GPRResults from './GPRResults';
import { useProcessingOptions } from './ProcessingOptionsContext';
import { ACCENT, BORDER, PANEL, TEXT, TEXT2, TEXT3, ACCENT_SOFT } from './tokens';

export default function GPRWorkspace() {
  const { session, user } = useAuth();
  const [searchParams] = useSearchParams();
  const viewJobId = searchParams.get('project_id');
  const { toFormData } = useProcessingOptions();

  const [files,          setFiles]          = useState<UploadedFile[]>([]);
  const [manufacturer,   setManufacturer]   = useState<ManufacturerKey | ''>('');
  const [frequencyMhz,   setFrequencyMhz]   = useState(1600);
  const [analysisName,   setAnalysisName]   = useState('');
  const [analysisNotes,  setAnalysisNotes]  = useState('');
  const [company,        setCompany]        = useState('');
  const [project,        setProject]        = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loadingJob,     setLoadingJob]     = useState(!!viewJobId);
  const [loadError,      setLoadError]      = useState<string | null>(null);

  const onComplete = useCallback((result: AnalysisResult) => {
    setAnalysisResult({
      ...result,
      analysis_name:  result.analysis_name  ?? (analysisName.trim()  || 'Untitled Analysis'),
      analysis_notes: result.analysis_notes ?? analysisNotes.trim(),
      company:        result.company        ?? company.trim(),
      project:        result.project        ?? project.trim(),
    });
  }, [analysisName, analysisNotes, company, project]);

  const {
    jobStatus, errorMsg, showConfirm, setShowConfirm,
    estimatedSecs, showProgressOverlay, onConfirmAnalysis, isAnalyzing,
  } = useAnalysisJob({
    files, session, manufacturer, frequencyMhz,
    useCustomFreq: false, customFreq: '',
    projectId: null,
    onComplete,
    extraFormData: toFormData,
    analysisName, analysisNotes,
    company, project,
  });

  // Pre-fill Company from the user's profile so analysts never re-type it.
  // Only runs when the field is still empty, so user-typed values are never
  // clobbered by an in-flight profile fetch.
  useEffect(() => {
    const uid = user?.id;
    if (!uid) return;
    let cancelled = false;
    (async () => {
      const { data: profile } = await supabase
        .from('profiles')
        .select('company')
        .eq('id', uid)
        .single();
      if (cancelled) return;
      const fromProfile = (profile?.company as string | undefined)?.trim();
      const fromMetadata = (user.user_metadata?.company as string | undefined)?.trim();
      const next = fromProfile || fromMetadata || '';
      if (next) setCompany(prev => prev || next);
    })();
    return () => { cancelled = true; };
  }, [user?.id, user?.user_metadata?.company]);

  useEffect(() => {
    if (!viewJobId) return;
    setLoadingJob(true);
    supabase.from('analysis_jobs').select('*').eq('id', viewJobId).single()
      .then(({ data, error }) => {
        if (error || !data?.result) {
          setLoadError('Could not load that job.');
        } else {
          const result = data.result as AnalysisResult;
          setAnalysisResult({
            ...result,
            analysis_name:  result.analysis_name  ?? (data.analysis_name  as string | undefined) ?? 'Untitled Analysis',
            analysis_notes: result.analysis_notes ?? (data.analysis_notes as string | undefined) ?? '',
            company:        result.company        ?? (data.company        as string | undefined) ?? '',
            project:        result.project        ?? (data.project        as string | undefined) ?? '',
          });
        }
        setLoadingJob(false);
      });
  }, [viewJobId]);

  const reset = () => {
    setAnalysisResult(null);
    setFiles([]);
    setAnalysisName('');
    setAnalysisNotes('');
    setCompany('');
    setProject('');
    setLoadError(null);
  };

  return (
    <div style={{ padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
      <Header onReset={reset} hasResult={!!analysisResult} />

      {loadError && (
        <div style={{ background: '#FFF8E6', border: `1px solid rgba(217,119,6,0.3)`, padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, color: '#92400e' }}>
          <AlertCircle size={14} /> {loadError}
        </div>
      )}

      {loadingJob ? (
        <LoadingCard />
      ) : analysisResult ? (
        <GPRResults result={analysisResult} projectId={viewJobId ?? undefined} />
      ) : (
        <GPRUploadCard
          files={files} setFiles={setFiles}
          manufacturer={manufacturer} setManufacturer={setManufacturer}
          frequencyMhz={frequencyMhz} setFrequencyMhz={setFrequencyMhz}
          analysisName={analysisName}   setAnalysisName={setAnalysisName}
          analysisNotes={analysisNotes} setAnalysisNotes={setAnalysisNotes}
          company={company}             setCompany={setCompany}
          project={project}             setProject={setProject}
          onStart={() => setShowConfirm(true)}
          busy={isAnalyzing}
        />
      )}

      {showConfirm && (
        <ConfirmAnalysisModal
          files={files} manufacturer={manufacturer} frequencyMhz={frequencyMhz}
          onConfirm={onConfirmAnalysis} onCancel={() => setShowConfirm(false)}
        />
      )}
      {showProgressOverlay && (
        <AnalysisProgressOverlay
          structureName="GPR Scan"
          estimatedSecs={estimatedSecs}
          fileCount={files.length}
          fileFormat={files[0]?.file.name.split('.').pop()?.toUpperCase() ?? 'GPR'}
          jobStatus={jobStatus as 'pending' | 'processing' | 'complete' | 'failed'}
          errorMsg={errorMsg}
        />
      )}
    </div>
  );
}

function Header({ onReset, hasResult }: { onReset: () => void; hasResult: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <div style={{ width: 38, height: 38, borderRadius: '50%', background: ACCENT_SOFT, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Radio size={18} style={{ color: ACCENT }} />
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: TEXT, letterSpacing: '-0.01em' }}>Ground-Penetrating Radar</div>
        <div style={{ fontSize: 11, color: TEXT3, fontWeight: 600, letterSpacing: '0.04em', marginTop: 2 }}>ASTM D6087 · Subsurface delamination + rebar depth</div>
      </div>
      {hasResult && (
        <button
          onClick={onReset}
          style={{ background: 'none', border: `1px solid ${BORDER}`, color: TEXT2, padding: '8px 14px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'inherit' }}
        >
          <RotateCcw size={12} /> New Scan
        </button>
      )}
    </div>
  );
}

function LoadingCard() {
  return (
    <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '48px 24px', textAlign: 'center', color: TEXT2, fontSize: 13 }}>
      <div style={{ width: 24, height: 24, border: `3px solid ${BORDER}`, borderTopColor: ACCENT, borderRadius: '50%', margin: '0 auto 12px', animation: 'spin 0.8s linear infinite' }} />
      Loading job…
    </div>
  );
}
