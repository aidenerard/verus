/**
 * GPRWorkspace.tsx — shell orchestrator. Owns top-level state, composes hooks/sub-components.
 * Does NOT contain canvas rendering, map init, analysis polling, or PanelGroup layout.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import type { ImperativePanelHandle } from 'react-resizable-panels';
import { AlertCircle } from 'lucide-react';
import { useAuth } from '../../../context/AuthContext';
import { supabase } from '../../../lib/supabase';
import { GPR_EXTS, TEXT, MANUFACTURER_EXTS, DEFAULT_ER } from './constants';
import type { ManufacturerKey, LayerId } from './constants';
import type { AnalysisResult, OutputTab, UploadedFile } from './types';
import { badgeColor } from './utils';
import { useMapbox } from './useMapbox';
import { useCanvasRenderers } from './useCanvasRenderers';
import { useAnalysisJob } from './useAnalysisJob';
import { useViewJobLoader } from './useViewJobLoader';
import { useSetupCallbacks } from './useSetupCallbacks';
import SetupWizard from './SetupWizard';
import ConfirmAnalysisModal from './ConfirmAnalysisModal';
import AnalysisProgressOverlay from './AnalysisProgressOverlay';
import WorkspaceToolbar from './WorkspaceToolbar';
import LayersSidebar from './LayersSidebar';
import CentralColumn from './CentralColumn';
import ProjectsDrawer from './ProjectsDrawer';

export default function GPRWorkspace() {
  const navigate    = useNavigate();
  const { session } = useAuth();
  const [searchParams] = useSearchParams();
  const viewJobId = searchParams.get('project_id');

  const [setupChecking,  setSetupChecking]  = useState(() => !!localStorage.getItem('verus_project_id') || !!viewJobId);
  const [setupDone,      setSetupDone]      = useState(false);
  const [setupStep,      setSetupStep]      = useState<1|2|3>(1);
  const [manufacturer,   setManufacturer]   = useState<ManufacturerKey | ''>('');
  const [frequencyMhz,   setFrequencyMhz]   = useState(1600);
  const [customFreq,     setCustomFreq]     = useState('');
  const [useCustomFreq,  setUseCustomFreq]  = useState(false);
  const [projectId,      setProjectId]      = useState<string | null>(null);
  const [inspDate,       setInspDate]       = useState(new Date().toISOString().slice(0, 10));
  const [bridgeId,       setBridgeId]       = useState('');
  const [notes,          setNotes]          = useState('');
  const [projectName,    setProjectName]    = useState('New Project');
  const [structureName,  setStructureName]  = useState('Bridge Deck');
  const [activeView,     setActiveView]     = useState<'cscan' | '3d'>('cscan');
  const [outputTab,      setOutputTab]      = useState<OutputTab>('gps');
  const [rightIconOpen,  setRightIconOpen]  = useState<'properties' | 'analysis' | 'adjust' | null>(null);
  const [bottomExpanded, setBottomExpanded] = useState(false);
  const [editingProject,   setEditingProject]   = useState(false);
  const [editingStructure, setEditingStructure] = useState(false);
  const [showSettingsMenu, setShowSettingsMenu] = useState(false);
  const [selectedLayer,    setSelectedLayer]    = useState<LayerId>('gpr');
  const [layerVis,         setLayerVis]         = useState<Record<LayerId, boolean>>({ gpr: true, condition: true, amplitude: false, satellite: true, annotations: true });
  const [conditionOpacity, setConditionOpacity] = useState(80);
  const [showAddMenu,      setShowAddMenu]      = useState(false);
  const [showExportMenu,   setShowExportMenu]   = useState(false);
  const [showProjects,     setShowProjects]     = useState(false);
  const [detectionThreshold, setDetectionThreshold] = useState(0.65);
  const [dielectricEr,       setDielectricEr]       = useState(6);
  const [ampClampMin,  setAmpClampMin]  = useState(0);
  const [ampClampMax,  setAmpClampMax]  = useState(1);
  const [useCondCanvas,  setUseCondCanvas]  = useState(false);
  const [useRebarCanvas, setUseRebarCanvas] = useState(false);
  const [useAmpCanvas,   setUseAmpCanvas]   = useState(false);
  const [files,           setFiles]           = useState<UploadedFile[]>([]);
  const [analysisResult,  setAnalysisResult]  = useState<AnalysisResult | null>(null);
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const [recentJobs,      setRecentJobs]      = useState<any[]>([]);
  const fileInputRef   = useRef<HTMLInputElement>(null);
  const bottomPanelRef = useRef<ImperativePanelHandle>(null);

  const { mapContainerRef, mapRef, mouseCoords } = useMapbox({ analysisResult, layerVis, conditionOpacity });
  const { bscanCanvasRef, condCanvasRef, rebarCanvasRef, ampCanvasRef, rebarColorbarRef } = useCanvasRenderers({
    analysisResult, selectedFileIdx, outputTab, detectionThreshold, dielectricEr, ampClampMin, ampClampMax,
  });

  const handleComplete = useCallback((result: AnalysisResult, otsuThreshold: number | undefined, freq: number | undefined) => {
    setAnalysisResult(result); setRightIconOpen('properties'); setSelectedFileIdx(0);
    setActiveView('cscan'); setOutputTab('condition');
    if (otsuThreshold) setDetectionThreshold(otsuThreshold);
    if (freq) setDielectricEr(DEFAULT_ER[freq] ?? 6);
    bottomPanelRef.current?.expand();
  }, []);

  const {
    jobStatus, setJobStatus, errorMsg, setErrorMsg, statusMsg,
    showConfirm, setShowConfirm, estimatedSecs, showProgressOverlay,
    onConfirmAnalysis, isAnalyzing,
  } = useAnalysisJob({ files, session, manufacturer, frequencyMhz, useCustomFreq, customFreq, projectId, onComplete: handleComplete });

  useViewJobLoader({
    viewJobId, session,
    setupSetters: { setSetupChecking, setSetupDone, setProjectId, setManufacturer, setFrequencyMhz,
                    setDielectricEr, setProjectName, setStructureName, setBridgeId, setInspDate, setNotes },
    jobSetters:   { setAnalysisResult, setJobStatus, setOutputTab, setRightIconOpen, setActiveView, setDetectionThreshold },
    bottomPanelRef,
  });

  const { completeSetup, newProject } = useSetupCallbacks({
    session, projectId, manufacturer, frequencyMhz, useCustomFreq, customFreq,
    structureName, projectName, bridgeId, inspDate, notes,
    setProjectId, setDielectricEr, setSetupDone, setManufacturer, setFrequencyMhz,
    setSetupStep, setFiles: setFiles as (v: []) => void,
    setJobStatus: setJobStatus as (v: 'idle') => void,
    setAnalysisResult: setAnalysisResult as (v: null) => void,
    setErrorMsg: setErrorMsg as (v: null) => void,
  });

  useEffect(() => { setDielectricEr(DEFAULT_ER[frequencyMhz] ?? 6); }, [frequencyMhz]);
  useEffect(() => { if (files.length > 0 && jobStatus === 'idle') setShowConfirm(true); }, [files]); // eslint-disable-line
  useEffect(() => {
    if (!showProjects) return;
    supabase.from('analysis_jobs').select('*').eq('status', 'complete')
      .order('created_at', { ascending: false }).limit(10)
      .then(({ data }) => setRecentJobs(data ?? []));
  }, [showProjects]);

  const onFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const accepted = Array.from(e.target.files).filter(f => GPR_EXTS.has(f.name.slice(f.name.lastIndexOf('.')).toLowerCase()));
    if (!accepted.length) return;
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...accepted.filter(f => !existing.has(f.name)).map(f => ({ file: f, name: f.name }))];
    });
    setRightIconOpen('analysis'); setShowAddMenu(false); e.target.value = '';
  }, []);

  const exportPNG = useCallback(() => {
    const a = document.createElement('a');
    const slug = projectName.replace(/\s+/g, '_');
    if (outputTab === 'gps') {
      const map = mapRef.current;
      if (map) { a.href = map.getCanvas().toDataURL(); a.download = `${slug}_map.png`; a.click(); }
      setShowExportMenu(false); return;
    }
    const canvas = ({ condition: condCanvasRef, rebar_depth: rebarCanvasRef, amplitude: ampCanvasRef } as any)[outputTab]?.current as HTMLCanvasElement | null;
    if (canvas && canvas.width > 0) {
      a.href = canvas.toDataURL('image/png'); a.download = `${slug}_${outputTab}.png`; a.click();
    } else if (analysisResult?.cscan_url && outputTab === 'condition') {
      a.href = analysisResult.cscan_url; a.download = `${slug}_condition.png`; a.click();
    }
    setShowExportMenu(false);
  }, [outputTab, projectName, analysisResult, mapRef, condCanvasRef, rebarCanvasRef, ampCanvasRef]);

  const loadJob = useCallback((job: any) => {
    if (!job.result) return;
    setAnalysisResult(job.result); setJobStatus('complete');
    setFiles([]); setSelectedFileIdx(0); setShowProjects(false);
    setRightIconOpen('properties'); setOutputTab('condition');
    if (job.result.otsu_threshold) setDetectionThreshold(job.result.otsu_threshold);
  }, [setJobStatus]);

  const hasResult  = analysisResult !== null;
  const totalFiles = analysisResult?.per_file_summary.length ?? 0;
  const fileAccept = manufacturer && MANUFACTURER_EXTS[manufacturer]
    ? MANUFACTURER_EXTS[manufacturer]
    : '.csv,.dzt,.DZT,.dt1,.DT1,.rd3,.rd7,.segy,.sgy,.dzg,.hd,.rad,.dt,.gec,.iprb,.iprh';

  const condBadge  = () => hasResult && analysisResult!.model_confidence_pct != null
    ? { text: `${analysisResult!.model_confidence_pct.toFixed(0)}%`, color: badgeColor(analysisResult!.model_confidence_pct >= 80, analysisResult!.model_confidence_pct >= 60) } : null;
  const depthBadge = () => hasResult && analysisResult!.depth_accuracy_in != null
    ? { text: `±${analysisResult!.depth_accuracy_in.toFixed(2)}"`, color: badgeColor(analysisResult!.depth_accuracy_in <= 0.25, analysisResult!.depth_accuracy_in <= 0.5) } : null;
  const ampBadge   = () => hasResult && analysisResult!.signal_quality
    ? { text: analysisResult!.signal_quality!, color: badgeColor(analysisResult!.signal_quality === 'Good', analysisResult!.signal_quality === 'Fair') } : null;

  const OUTPUT_TABS: { id: OutputTab; label: string; badge: () => {text:string;color:string}|null }[] = [
    { id: 'condition',   label: 'Condition Map', badge: condBadge  },
    { id: 'rebar_depth', label: 'Rebar Depth',   badge: depthBadge },
    { id: 'amplitude',   label: 'Amplitude',     badge: ampBadge   },
    { id: 'gps',         label: 'GPS Map',        badge: () => null },
  ];

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#F5F3EF', color: TEXT, overflow: 'hidden', fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', userSelect: 'none' }}>
      <input ref={fileInputRef} type="file" multiple accept={fileAccept} onChange={onFileInput} style={{ display: 'none' }} />

      {setupDone && !manufacturer && (
        <div onClick={() => { setSetupDone(false); setSetupStep(1); }}
          style={{ background: '#FFF8E6', borderBottom: '1px solid rgba(217,119,6,0.3)', padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, cursor: 'pointer', zIndex: 50, flexShrink: 0 }}>
          <AlertCircle size={14} style={{ color: '#d97706', flexShrink: 0 }} />
          <span style={{ color: '#92400e' }}>Equipment not configured — click here to complete setup</span>
        </div>
      )}

      {!setupDone && !setupChecking && (
        <SetupWizard
          setupStep={setupStep} setSetupStep={setSetupStep}
          manufacturer={manufacturer} setManufacturer={setManufacturer}
          frequencyMhz={frequencyMhz} setFrequencyMhz={setFrequencyMhz}
          customFreq={customFreq} setCustomFreq={setCustomFreq}
          useCustomFreq={useCustomFreq} setUseCustomFreq={setUseCustomFreq}
          structureName={structureName} setStructureName={setStructureName}
          projectName={projectName} setProjectName={setProjectName}
          bridgeId={bridgeId} setBridgeId={setBridgeId}
          inspDate={inspDate} setInspDate={setInspDate}
          notes={notes} setNotes={setNotes} onComplete={completeSetup}
        />
      )}

      <WorkspaceToolbar
        navigate={navigate}
        editingProject={editingProject} setEditingProject={setEditingProject}
        projectName={projectName} setProjectName={setProjectName}
        editingStructure={editingStructure} setEditingStructure={setEditingStructure}
        structureName={structureName} setStructureName={setStructureName}
        activeView={activeView} setActiveView={setActiveView}
        mouseCoords={mouseCoords} outputTab={outputTab} setupDone={setupDone}
        showSettingsMenu={showSettingsMenu} setShowSettingsMenu={v => setShowSettingsMenu(v as any)}
        newProject={newProject} fileInputRef={fileInputRef} setRightIconOpen={setRightIconOpen}
        showExportMenu={showExportMenu} setShowExportMenu={v => setShowExportMenu(v as any)}
        exportPNG={exportPNG} setSetupDone={setSetupDone} setSetupStep={setSetupStep}
      />

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <LayersSidebar
          selectedLayer={selectedLayer} setSelectedLayer={setSelectedLayer}
          setRightIconOpen={setRightIconOpen} layerVis={layerVis} setLayerVis={setLayerVis}
          files={files} selectedFileIdx={selectedFileIdx} setSelectedFileIdx={setSelectedFileIdx}
          isAnalyzing={isAnalyzing} jobStatus={jobStatus}
          setFiles={setFiles} setJobStatus={setJobStatus}
          setAnalysisResult={() => setAnalysisResult(null)} setErrorMsg={() => setErrorMsg(null)}
          showAddMenu={showAddMenu} setShowAddMenu={v => setShowAddMenu(v as any)}
          fileInputRef={fileInputRef}
          showProjects={showProjects} setShowProjects={v => setShowProjects(v as any)}
        />
        <CentralColumn
          mapContainerRef={mapContainerRef} activeView={activeView}
          outputTab={outputTab} setOutputTab={setOutputTab}
          hasResult={hasResult} analysisResult={analysisResult}
          condCanvasRef={condCanvasRef} rebarCanvasRef={rebarCanvasRef}
          rebarColorbarRef={rebarColorbarRef} ampCanvasRef={ampCanvasRef}
          exportPNG={exportPNG} condBadge={condBadge} depthBadge={depthBadge} ampBadge={ampBadge}
          rightIconOpen={rightIconOpen} setRightIconOpen={setRightIconOpen}
          selectedLayer={selectedLayer} selectedFileIdx={selectedFileIdx} setSelectedFileIdx={setSelectedFileIdx}
          bottomPanelRef={bottomPanelRef} conditionOpacity={conditionOpacity} setConditionOpacity={setConditionOpacity}
          manufacturer={manufacturer} frequencyMhz={frequencyMhz} bridgeId={bridgeId} inspDate={inspDate}
          setupDone={setupDone} isAnalyzing={isAnalyzing} jobStatus={jobStatus}
          statusMsg={statusMsg} errorMsg={errorMsg} files={files} fileInputRef={fileInputRef}
          detectionThreshold={detectionThreshold} setDetectionThreshold={setDetectionThreshold}
          setUseCondCanvas={setUseCondCanvas} dielectricEr={dielectricEr} setDielectricEr={setDielectricEr}
          setUseRebarCanvas={setUseRebarCanvas} ampClampMin={ampClampMin} setAmpClampMin={setAmpClampMin}
          ampClampMax={ampClampMax} setAmpClampMax={setAmpClampMax} setUseAmpCanvas={setUseAmpCanvas}
          projectId={projectId} setShowConfirm={setShowConfirm} totalFiles={totalFiles}
          bscanCanvasRef={bscanCanvasRef} bottomExpanded={bottomExpanded} setBottomExpanded={setBottomExpanded}
          outputTabs={OUTPUT_TABS}
        />
      </div>

      {showProjects && <ProjectsDrawer recentJobs={recentJobs} loadJob={loadJob} setShowProjects={setShowProjects} />}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes verus-bar { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
        .mapboxgl-ctrl-bottom-right { z-index: 1 !important; }
        .mapboxgl-ctrl-bottom-left  { z-index: 1 !important; }
      `}</style>

      {showConfirm && (
        <ConfirmAnalysisModal
          files={files} manufacturer={manufacturer} frequencyMhz={frequencyMhz}
          onConfirm={onConfirmAnalysis} onCancel={() => setShowConfirm(false)}
        />
      )}
      {showProgressOverlay && (
        <AnalysisProgressOverlay
          structureName={structureName} estimatedSecs={estimatedSecs} fileCount={files.length}
          fileFormat={files.length > 0 ? (files[0].file.name.split('.').pop()?.toUpperCase() ?? 'GPR') : 'GPR'}
          jobStatus={jobStatus as 'pending' | 'processing' | 'complete' | 'failed'} errorMsg={errorMsg}
        />
      )}
    </div>
  );
}
