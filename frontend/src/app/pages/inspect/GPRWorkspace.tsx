import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { ImperativePanelHandle } from 'react-resizable-panels';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  ArrowLeft, Eye, EyeOff, Plus, ChevronDown, ChevronLeft, ChevronRight,
  Layers, Download, X, FolderOpen, Loader2, Check, AlertCircle,
  Maximize2, Minimize2, Radio, Settings,
} from 'lucide-react';

import ThreeDView from '../../components/ThreeDView';
import VerusLogo from '../../components/VerusLogo';
import { useAuth } from '../../../context/AuthContext';
import { supabase } from '../../../lib/supabase';

import {
  SERVER, MAPBOX_TOKEN, DEFAULT_CENTER, GPR_EXTS,
  BG, PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2, ACCENT,
  STATUS_MSGS, WAKE_TIMEOUT_MS, WAKE_INTERVAL_MS, POLL_TIMEOUT_MS,
  MANUFACTURERS, MANUFACTURER_EXTS, DEFAULT_ER, LAYER_DEFS,
} from './constants';
import type { ManufacturerKey, LayerId } from './constants';
import type { AnalysisResult, OutputTab, UploadedFile } from './types';
import { decodeF32, renderConditionToCanvas, renderDepthToCanvas } from './colormaps';
import { delamColor, badgeColor } from './utils';
import SetupWizard from './SetupWizard';
import AdjustPanel from './AdjustPanel';
import OutputMaps from './OutputMaps';

export default function GPRWorkspace() {
  const navigate    = useNavigate();
  const { session } = useAuth();

  // ── Setup ─────────────────────────────────────────────────────────────────────
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

  // ── View ─────────────────────────────────────────────────────────────────────
  const [activeView,     setActiveView]     = useState<'cscan' | '3d'>('cscan');
  const [outputTab,      setOutputTab]      = useState<OutputTab>('gps');
  const [rightTab,       setRightTab]       = useState<'properties' | 'analysis'>('analysis');
  const [bottomExpanded, setBottomExpanded] = useState(false);
  const [mouseCoords,    setMouseCoords]    = useState<{ x: number; y: number } | null>(null);
  const [editingProject,    setEditingProject]    = useState(false);
  const [editingStructure,  setEditingStructure]  = useState(false);

  // ── Layers ────────────────────────────────────────────────────────────────────
  const [selectedLayer,    setSelectedLayer]    = useState<LayerId>('gpr');
  const [layerVis,         setLayerVis]         = useState<Record<LayerId, boolean>>({ gpr: true, condition: true, amplitude: false, satellite: true, annotations: true });
  const [conditionOpacity, setConditionOpacity] = useState(80);
  const [showAddMenu,      setShowAddMenu]      = useState(false);
  const [showExportMenu,   setShowExportMenu]   = useState(false);
  const [showProjects,     setShowProjects]     = useState(false);

  // ── Adjust ────────────────────────────────────────────────────────────────────
  const [detectionThreshold, setDetectionThreshold] = useState(0.65);
  const [dielectricEr,       setDielectricEr]       = useState(6);
  const [ampClampMin,        setAmpClampMin]         = useState(0);
  const [ampClampMax,        setAmpClampMax]         = useState(1);
  const [useCondCanvas,  setUseCondCanvas]  = useState(false);
  const [useRebarCanvas, setUseRebarCanvas] = useState(false);
  const [useAmpCanvas,   setUseAmpCanvas]   = useState(false);

  // ── Analysis ─────────────────────────────────────────────────────────────────
  const [files,           setFiles]           = useState<UploadedFile[]>([]);
  const [jobId,           setJobId]           = useState<string | null>(null);
  const [jobStatus,       setJobStatus]       = useState<'idle'|'pending'|'processing'|'complete'|'failed'>('idle');
  const [analysisResult,  setAnalysisResult]  = useState<AnalysisResult | null>(null);
  const [errorMsg,        setErrorMsg]        = useState<string | null>(null);
  const [statusMsg,       setStatusMsg]       = useState('');
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const [recentJobs,      setRecentJobs]      = useState<any[]>([]);

  // ── Refs ──────────────────────────────────────────────────────────────────────
  const fileInputRef    = useRef<HTMLInputElement>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef          = useRef<mapboxgl.Map | null>(null);
  const bscanCanvasRef  = useRef<HTMLCanvasElement>(null);
  const condCanvasRef   = useRef<HTMLCanvasElement>(null);
  const rebarCanvasRef  = useRef<HTMLCanvasElement>(null);
  const ampCanvasRef    = useRef<HTMLCanvasElement>(null);
  const pollRef         = useRef<ReturnType<typeof setInterval> | null>(null);
  const statusCycleRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const rightPanelRef   = useRef<ImperativePanelHandle>(null);
  const bottomPanelRef  = useRef<ImperativePanelHandle>(null);

  // ── Restore setup from localStorage ──────────────────────────────────────────
  useEffect(() => {
    const pid  = localStorage.getItem('verus_project_id');
    const mfr  = localStorage.getItem('verus_manufacturer') as ManufacturerKey | null;
    const freq = parseInt(localStorage.getItem('verus_frequency_mhz') || '1600');
    const pn   = localStorage.getItem('verus_project_name');
    const sn   = localStorage.getItem('verus_structure_name');
    if (pid && mfr) {
      setProjectId(pid); setManufacturer(mfr); setFrequencyMhz(freq);
      setDielectricEr(DEFAULT_ER[freq] ?? 6);
      if (pn) setProjectName(pn);
      if (sn) setStructureName(sn);
      setSetupDone(true);
    }
  }, []);

  useEffect(() => { setDielectricEr(DEFAULT_ER[frequencyMhz] ?? 6); }, [frequencyMhz]);

  // ── Mapbox ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current || !MAPBOX_TOKEN) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: DEFAULT_CENTER, zoom: 5,
    });
    map.addControl(new mapboxgl.NavigationControl({ visualizePitch: false }), 'bottom-right');
    map.addControl(new mapboxgl.ScaleControl({ unit: 'imperial' }), 'bottom-left');
    map.on('mousemove', (e) => {
      const c  = map.getCenter();
      const dx = (e.lngLat.lng - c.lng) * 111320 * Math.cos(c.lat * Math.PI / 180) * 3.28084;
      const dy = (e.lngLat.lat - c.lat) * 110540 * 3.28084;
      setMouseCoords({ x: Math.round(dx), y: Math.round(dy) });
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = analysisResult && mapRef.current ? mapRef.current : null;
    if (!map || !analysisResult) return;
    const add = () => {
      ['condition-fill'].forEach(id => { if (map.getLayer(id)) map.removeLayer(id); });
      if (map.getSource('condition')) map.removeSource('condition');
      const gpsFiles = analysisResult.per_file_summary.filter(f => f.gps);
      if (!gpsFiles.length) return;
      const features: GeoJSON.Feature<GeoJSON.LineString>[] = gpsFiles.map(f => ({
        type: 'Feature',
        properties: { filename: f.filename, delam_pct: f.delam_pct },
        geometry: { type: 'LineString', coordinates: f.gps!.coordinates.map(([lat, lon]) => [lon, lat]) },
      }));
      map.addSource('condition', { type: 'geojson', data: { type: 'FeatureCollection', features } });
      map.addLayer({
        id: 'condition-fill', type: 'line', source: 'condition',
        layout: { visibility: layerVis.condition ? 'visible' : 'none' },
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'delam_pct'], 0, '#22c55e', 50, '#f59e0b', 100, '#ef4444'],
          'line-width': 5, 'line-opacity': conditionOpacity / 100,
        },
      });
      const first = gpsFiles[0].gps!;
      map.flyTo({ center: [first.lon_start, first.lat_start], zoom: 17, duration: 1500 });
    };
    if (map.isStyleLoaded()) add(); else map.once('load', add);
  }, [analysisResult]); // eslint-disable-line

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getLayer('condition-fill'))
      map.setLayoutProperty('condition-fill', 'visibility', layerVis.condition ? 'visible' : 'none');
  }, [layerVis.condition]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer('condition-fill')) return;
    map.setPaintProperty('condition-fill', 'line-opacity', conditionOpacity / 100);
  }, [conditionOpacity]);

  // ── B-scan render ─────────────────────────────────────────────────────────────
  const renderBscan = useCallback(() => {
    const canvas = bscanCanvasRef.current;
    if (!canvas || !analysisResult) return;
    const fileResult = analysisResult.per_file_summary[selectedFileIdx];
    if (!fileResult?.bscan) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = 400; canvas.height = 120;
        ctx.fillStyle = PANEL; ctx.fillRect(0, 0, 400, 120);
        ctx.fillStyle = TEXT2; ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('B-scan data not available for this file', 200, 65);
      }
      return;
    }
    const { data, n_traces, n_samples } = fileResult.bscan;
    const binary = atob(data);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const floats = new Float32Array(bytes.buffer);
    let minVal = Infinity, maxVal = -Infinity;
    for (const v of floats) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
    const range = maxVal - minVal || 1;
    canvas.width = n_traces; canvas.height = n_samples;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imageData = ctx.createImageData(n_traces, n_samples);
    for (let t = 0; t < n_traces; t++) {
      for (let s = 0; s < n_samples; s++) {
        const v   = Math.round(((floats[t * n_samples + s] - minVal) / range) * 255);
        const idx = (s * n_traces + t) * 4;
        imageData.data[idx] = v; imageData.data[idx+1] = v; imageData.data[idx+2] = v; imageData.data[idx+3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
    ctx.strokeStyle = '#ff3cff'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let t = 0; t < n_traces; t++) {
      let peakS = 0, peakA = -Infinity;
      for (let s = 0; s < n_samples; s++) {
        const a = Math.abs(floats[t * n_samples + s]);
        if (a > peakA) { peakA = a; peakS = s; }
      }
      t === 0 ? ctx.moveTo(t, peakS) : ctx.lineTo(t, peakS);
    }
    ctx.stroke();
  }, [analysisResult, selectedFileIdx]);

  useEffect(() => { renderBscan(); }, [renderBscan]);

  // ── Canvas re-renders on slider change ────────────────────────────────────────
  useEffect(() => {
    if (!useCondCanvas || !condCanvasRef.current || !analysisResult?.prob_grid) return;
    renderConditionToCanvas(condCanvasRef.current, decodeF32(analysisResult.prob_grid),
      analysisResult.prob_grid_rows!, analysisResult.prob_grid_cols!, detectionThreshold);
  }, [useCondCanvas, detectionThreshold, analysisResult]);

  useEffect(() => {
    if (!useRebarCanvas || !rebarCanvasRef.current || !analysisResult?.twt_grid) return;
    renderDepthToCanvas(rebarCanvasRef.current, decodeF32(analysisResult.twt_grid),
      analysisResult.twt_grid_rows!, analysisResult.twt_grid_cols!, dielectricEr);
  }, [useRebarCanvas, dielectricEr, analysisResult]);

  // ── File handling ─────────────────────────────────────────────────────────────
  const onFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const accepted = Array.from(e.target.files).filter(f => {
      const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
      return GPR_EXTS.has(ext);
    });
    if (!accepted.length) return;
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      const fresh = accepted.filter(f => !existing.has(f.name)).map(f => ({ file: f, name: f.name }));
      return [...prev, ...fresh];
    });
    setRightTab('analysis');
    setShowAddMenu(false);
    e.target.value = '';
  }, []);

  useEffect(() => {
    if (files.length > 0 && jobStatus === 'idle') startAnalysis();
  }, [files]); // eslint-disable-line

  // ── Analysis ──────────────────────────────────────────────────────────────────
  const startAnalysis = useCallback(async () => {
    if (!files.length || jobStatus === 'pending' || jobStatus === 'processing') return;
    setJobStatus('pending');
    setErrorMsg(null);
    setStatusMsg('Waking up server…');
    setUseCondCanvas(false); setUseRebarCanvas(false); setUseAmpCanvas(false);

    const headers: Record<string, string> = {};
    if (session?.access_token) headers['Authorization'] = `Bearer ${session.access_token}`;

    try {
      const wakeDeadline = Date.now() + WAKE_TIMEOUT_MS;
      let serverReady = false;
      while (Date.now() < wakeDeadline) {
        try {
          const h = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(10000) });
          if (h.ok) { const hj = await h.json(); if (hj.model_loaded) { serverReady = true; break; } setStatusMsg('Loading AI model…'); }
        } catch { /* waking */ }
        await new Promise(r => setTimeout(r, WAKE_INTERVAL_MS));
      }
      if (!serverReady) { setErrorMsg('Server did not respond in time.'); setJobStatus('failed'); return; }

      setStatusMsg('Uploading files…');
      const formData = new FormData();
      files.forEach(f => formData.append('files', f.file));
      if (manufacturer) formData.append('manufacturer', manufacturer);
      const effectiveFreq = useCustomFreq ? (parseInt(customFreq) || 1600) : frequencyMhz;
      formData.append('frequency_mhz', String(effectiveFreq));
      if (projectId) formData.append('project_id', projectId);

      const res = await fetch(`${SERVER}/analyze`, {
        method: 'POST', headers, body: formData, signal: AbortSignal.timeout(60000),
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.detail || j.error || msg; } catch {}
        setErrorMsg(msg); setJobStatus('failed'); return;
      }

      const { job_id } = await res.json();
      setJobId(job_id);
      setJobStatus('processing');
      let msgIdx = 2;
      statusCycleRef.current = setInterval(() => {
        msgIdx = 2 + ((msgIdx - 1) % (STATUS_MSGS.length - 2));
        setStatusMsg(STATUS_MSGS[msgIdx]);
      }, 4000);

      const pollDeadline = Date.now() + POLL_TIMEOUT_MS;
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        if (Date.now() > pollDeadline) {
          clearInterval(pollRef.current!); clearInterval(statusCycleRef.current!);
          setErrorMsg('Analysis timed out.'); setJobStatus('failed'); return;
        }
        try {
          const jr = await fetch(`${SERVER}/job/${job_id}`, { headers });
          if (!jr.ok) return;
          const job = await jr.json();
          if (job.status === 'complete' && job.result) {
            clearInterval(pollRef.current!); clearInterval(statusCycleRef.current!);
            setAnalysisResult(job.result);
            setJobStatus('complete');
            setRightTab('properties');
            setSelectedFileIdx(0);
            setActiveView('cscan');
            setOutputTab('condition');
            if (job.result.otsu_threshold) setDetectionThreshold(job.result.otsu_threshold);
            if (job.result.frequency_mhz) setDielectricEr(DEFAULT_ER[job.result.frequency_mhz] ?? 6);
            bottomPanelRef.current?.expand();
          } else if (job.status === 'failed') {
            clearInterval(pollRef.current!); clearInterval(statusCycleRef.current!);
            setErrorMsg(job.error || 'Analysis failed'); setJobStatus('failed');
          }
        } catch { /* poll silently */ }
      }, 3000);
    } catch (err) {
      clearInterval(statusCycleRef.current!);
      setErrorMsg(err instanceof Error ? err.message : 'Analysis failed');
      setJobStatus('failed');
    }
  }, [files, session, jobStatus, manufacturer, frequencyMhz, useCustomFreq, customFreq, projectId]);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (statusCycleRef.current) clearInterval(statusCycleRef.current);
  }, []);

  // ── Setup completion ──────────────────────────────────────────────────────────
  const completeSetup = useCallback(async () => {
    const effectiveFreq = useCustomFreq ? (parseInt(customFreq) || 1600) : frequencyMhz;
    localStorage.setItem('verus_manufacturer',    manufacturer);
    localStorage.setItem('verus_frequency_mhz',   String(effectiveFreq));
    localStorage.setItem('verus_project_name',     projectName);
    localStorage.setItem('verus_structure_name',   structureName);

    if (session?.user?.id) {
      try {
        const { data } = await supabase.from('projects').insert({
          user_id:         session.user.id, name: projectName,
          structure_name:  structureName,   bridge_id: bridgeId || null,
          inspection_date: inspDate,        notes: notes || null,
          manufacturer:    manufacturer || null, frequency_mhz: effectiveFreq,
        }).select('id').single();
        if (data?.id) { setProjectId(data.id); localStorage.setItem('verus_project_id', data.id); }
      } catch { /* non-fatal */ }
    }
    setDielectricEr(DEFAULT_ER[effectiveFreq] ?? 6);
    setSetupDone(true);
  }, [manufacturer, frequencyMhz, useCustomFreq, customFreq, structureName, projectName, bridgeId, inspDate, notes, session]);

  const newProject = useCallback(() => {
    localStorage.removeItem('verus_project_id');
    localStorage.removeItem('verus_manufacturer');
    setManufacturer(''); setFrequencyMhz(1600); setProjectId(null);
    setSetupDone(false); setSetupStep(1);
    setFiles([]); setJobStatus('idle'); setAnalysisResult(null); setErrorMsg(null);
  }, []);

  // ── Export ────────────────────────────────────────────────────────────────────
  const exportPNG = useCallback(() => {
    if (!analysisResult) return;
    const a = document.createElement('a');
    const slug = projectName.replace(/\s+/g, '_');
    if (outputTab === 'condition') {
      a.href = `data:image/png;base64,${analysisResult.cscan_image}`;
      a.download = `${slug}_condition.png`;
    } else if (outputTab === 'rebar_depth' && analysisResult.rebar_depth_image) {
      a.href = `data:image/png;base64,${analysisResult.rebar_depth_image}`;
      a.download = `${slug}_rebar_depth.png`;
    } else if (outputTab === 'amplitude' && analysisResult.amplitude_image) {
      a.href = `data:image/png;base64,${analysisResult.amplitude_image}`;
      a.download = `${slug}_amplitude.png`;
    } else {
      const map = mapRef.current;
      if (map) { a.href = map.getCanvas().toDataURL(); a.download = `${slug}_map.png`; }
    }
    if (a.href) a.click();
    setShowExportMenu(false);
  }, [analysisResult, outputTab, projectName]);

  useEffect(() => {
    if (!showProjects) return;
    supabase.from('analysis_jobs').select('*').eq('status', 'complete')
      .order('created_at', { ascending: false }).limit(10)
      .then(({ data }) => setRecentJobs(data ?? []));
  }, [showProjects]);

  const loadJob = useCallback((job: any) => {
    if (!job.result) return;
    setAnalysisResult(job.result); setJobStatus('complete');
    setFiles([]); setSelectedFileIdx(0); setShowProjects(false);
    setRightTab('properties'); setOutputTab('condition');
    if (job.result.otsu_threshold) setDetectionThreshold(job.result.otsu_threshold);
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────────────
  const isAnalyzing  = jobStatus === 'pending' || jobStatus === 'processing';
  const hasResult    = analysisResult !== null;
  const totalFiles   = analysisResult?.per_file_summary.length ?? 0;
  const fileAccept   = manufacturer && MANUFACTURER_EXTS[manufacturer]
    ? MANUFACTURER_EXTS[manufacturer]
    : '.csv,.dzt,.DZT,.dt1,.DT1,.rd3,.rd7,.segy,.sgy,.dzg,.hd,.rad,.dt,.gec,.iprb,.iprh';

  const condBadge  = () => hasResult && analysisResult!.model_confidence_pct !== undefined
    ? { text: `${analysisResult!.model_confidence_pct.toFixed(0)}%`, color: badgeColor(analysisResult!.model_confidence_pct >= 80, analysisResult!.model_confidence_pct >= 60) }
    : null;
  const depthBadge = () => hasResult && analysisResult!.depth_accuracy_in !== undefined
    ? { text: `±${analysisResult!.depth_accuracy_in}"`, color: badgeColor(analysisResult!.depth_accuracy_in <= 0.25, analysisResult!.depth_accuracy_in <= 0.5) }
    : null;
  const ampBadge   = () => hasResult && analysisResult!.signal_quality
    ? { text: analysisResult!.signal_quality!, color: badgeColor(analysisResult!.signal_quality === 'Good', analysisResult!.signal_quality === 'Fair') }
    : null;

  const OUTPUT_TABS: { id: OutputTab; label: string; badge: () => {text:string;color:string}|null }[] = [
    { id: 'condition',   label: 'Condition Map', badge: condBadge  },
    { id: 'rebar_depth', label: 'Rebar Depth',   badge: depthBadge },
    { id: 'amplitude',   label: 'Amplitude',     badge: ampBadge   },
    { id: 'gps',         label: 'GPS Map',        badge: () => null },
  ];

  // ── RENDER ────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      background: BG, color: TEXT, overflow: 'hidden',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', userSelect: 'none',
    }}>
      <input ref={fileInputRef} type="file" multiple accept={fileAccept}
        onChange={onFileInput} style={{ display: 'none' }} />

      {!setupDone && (
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
          notes={notes} setNotes={setNotes}
          onComplete={completeSetup}
        />
      )}

      {/* ── TOP TOOLBAR ─────────────────────────────────────────────────────────── */}
      <div style={{ height: 48, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px', gap: 8, background: PANEL, borderBottom: `1px solid ${BORDER}`, position: 'relative', zIndex: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
          <button onClick={() => navigate('/dashboard')} title="Back"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: '6px 8px', display: 'flex', alignItems: 'center', borderRadius: 4, flexShrink: 0 }}
            onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
            onMouseLeave={e => (e.currentTarget.style.background = 'none')}
          ><ArrowLeft size={16} /></button>
          <div style={{ width: 1, height: 20, background: BORDER, flexShrink: 0 }} />
          <VerusLogo size={22} wordmarkColor="#0A0A0A" />
          <div style={{ width: 1, height: 20, background: BORDER, flexShrink: 0 }} />
          {editingProject
            ? <input autoFocus value={projectName} onChange={e => setProjectName(e.target.value)}
                onBlur={() => setEditingProject(false)} onKeyDown={e => e.key === 'Enter' && setEditingProject(false)}
                style={{ background: RAISED, border: `1px solid ${BORDER2}`, color: TEXT, fontSize: 13, fontWeight: 600, padding: '3px 8px', outline: 'none', width: 160, fontFamily: 'Inter, sans-serif' }} />
            : <span onClick={() => setEditingProject(true)} title="Rename"
                style={{ fontSize: 13, fontWeight: 600, color: TEXT, cursor: 'text', padding: '3px 6px', borderRadius: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 160 }}
                onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
                onMouseLeave={e => (e.currentTarget.style.background = 'none')}
              >{projectName}</span>
          }
          <span style={{ color: TEXT2, fontSize: 12, flexShrink: 0 }}>/</span>
          {editingStructure
            ? <input autoFocus value={structureName} onChange={e => setStructureName(e.target.value)}
                onBlur={() => setEditingStructure(false)} onKeyDown={e => e.key === 'Enter' && setEditingStructure(false)}
                style={{ background: RAISED, border: `1px solid ${BORDER2}`, color: TEXT2, fontSize: 12, padding: '3px 8px', outline: 'none', width: 130, fontFamily: 'Inter, sans-serif' }} />
            : <span onClick={() => setEditingStructure(true)} title="Rename"
                style={{ fontSize: 12, color: TEXT2, cursor: 'text', padding: '3px 6px', borderRadius: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 130 }}
                onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
                onMouseLeave={e => (e.currentTarget.style.background = 'none')}
              >{structureName}</span>
          }
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          <div style={{ display: 'flex', background: RAISED, border: `1px solid ${BORDER}`, borderRadius: 6, overflow: 'hidden' }}>
            {(['cscan', '3d'] as const).map(v => (
              <button key={v} onClick={() => setActiveView(v)}
                style={{ padding: '5px 16px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', background: activeView === v ? 'rgba(232,96,28,0.18)' : 'none', color: activeView === v ? ACCENT : TEXT2, border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif', transition: 'background 0.15s, color 0.15s' }}>
                {v === 'cscan' ? 'Maps' : '3D'}
              </button>
            ))}
          </div>
          {mouseCoords && activeView === 'cscan' && outputTab === 'gps' && (
            <span style={{ fontSize: 11, color: TEXT2, fontVariantNumeric: 'tabular-nums', minWidth: 130 }}>
              X {mouseCoords.x.toLocaleString()} ft &nbsp;|&nbsp; Y {mouseCoords.y.toLocaleString()} ft
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, justifyContent: 'flex-end', position: 'relative' }}>
          {setupDone && (
            <button onClick={newProject}
              style={{ padding: '6px 12px', background: 'none', border: `1px solid ${BORDER}`, color: TEXT2, fontSize: 11, fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter, sans-serif', borderRadius: 4, display: 'flex', alignItems: 'center', gap: 5 }}
              onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            ><Settings size={12} /> New Project</button>
          )}
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowExportMenu(v => !v)}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', background: RAISED, border: `1px solid ${BORDER2}`, color: TEXT, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif', borderRadius: 4 }}>
              <Download size={13} /> Export <ChevronDown size={11} />
            </button>
            {showExportMenu && (
              <div style={{ position: 'absolute', top: '100%', right: 0, marginTop: 4, background: RAISED, border: `1px solid ${BORDER2}`, zIndex: 100, minWidth: 180, boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}>
                {[
                  { label: 'Export Current Map', action: exportPNG },
                  { label: 'Export PDF Report',  action: () => setShowExportMenu(false) },
                  { label: 'Export CSV Data',    action: () => setShowExportMenu(false) },
                ].map(({ label, action }) => (
                  <button key={label} onClick={action}
                    style={{ display: 'block', width: '100%', textAlign: 'left', padding: '10px 16px', background: 'none', border: 'none', color: TEXT, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.05)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                  >{label}</button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── MAIN CONTENT ──────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <PanelGroup direction="horizontal" style={{ height: '100%' }}>

          {/* ── LEFT SIDEBAR ────────────────────────────────────────────────────── */}
          <Panel defaultSize={18} minSize={12} maxSize={28}
            style={{ background: PANEL, borderRight: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '12px 14px 10px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2 }}>Layers</span>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
              {LAYER_DEFS.map(({ id, label }) => (
                <div key={id} onClick={() => { setSelectedLayer(id); setRightTab('properties'); }}
                  style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 14px', cursor: 'pointer', background: selectedLayer === id ? 'rgba(232,96,28,0.1)' : 'none', borderLeft: `2px solid ${selectedLayer === id ? ACCENT : 'transparent'}`, transition: 'background 0.12s' }}
                  onMouseEnter={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; }}
                  onMouseLeave={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'none'; }}
                >
                  <span style={{ fontSize: 12, color: selectedLayer === id ? TEXT : TEXT2, flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {label}{id === 'gpr' && files.length > 0 && <span style={{ marginLeft: 6, fontSize: 10, color: TEXT2 }}>({files.length})</span>}
                  </span>
                  <button onClick={ev => { ev.stopPropagation(); setLayerVis(v => ({ ...v, [id]: !v[id as LayerId] })); }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: layerVis[id as LayerId] ? TEXT2 : '#C8C3BD', display: 'flex' }}>
                    {layerVis[id as LayerId] ? <Eye size={13} /> : <EyeOff size={13} />}
                  </button>
                </div>
              ))}
              {files.length > 0 && (
                <div style={{ borderTop: `1px solid ${BORDER}`, marginTop: 4 }}>
                  {files.map((f, i) => (
                    <div key={f.name} onClick={() => { setSelectedFileIdx(i); setSelectedLayer('gpr'); }}
                      style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 8px 5px 28px', cursor: 'pointer', background: selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none' }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.04)')}
                      onMouseLeave={e => (e.currentTarget.style.background = selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none')}
                    >
                      {isAnalyzing ? <Loader2 size={10} style={{ color: ACCENT, animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                        : jobStatus === 'complete' ? <Check size={10} style={{ color: '#22c55e', flexShrink: 0 }} /> : null}
                      <span style={{ fontSize: 11, color: TEXT2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>{f.name}</span>
                      {!isAnalyzing && (
                        <button onClick={ev => { ev.stopPropagation(); setFiles(prev => { const next = prev.filter((_,j) => j !== i); if (!next.length) { setJobStatus('idle'); setAnalysisResult(null); setErrorMsg(null); setSelectedFileIdx(0); } else if (selectedFileIdx >= next.length) setSelectedFileIdx(next.length-1); return next; }); }}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: TEXT2, display: 'flex', flexShrink: 0, opacity: 0.5 }}
                          onMouseEnter={e => { e.currentTarget.style.opacity='1'; e.currentTarget.style.color='#ef4444'; }}
                          onMouseLeave={e => { e.currentTarget.style.opacity='0.5'; e.currentTarget.style.color=TEXT2; }}>
                          <X size={11} />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}`, padding: 10, position: 'relative' }}>
              <button onClick={() => setShowAddMenu(v => !v)}
                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '8px', background: 'rgba(0,0,0,0.03)', border: `1px dashed ${BORDER2}`, color: TEXT2, cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'Inter, sans-serif', borderRadius: 3 }}
                onMouseEnter={e => { e.currentTarget.style.background='rgba(0,0,0,0.07)'; e.currentTarget.style.color=TEXT; }}
                onMouseLeave={e => { e.currentTarget.style.background='rgba(0,0,0,0.03)'; e.currentTarget.style.color=TEXT2; }}>
                <Plus size={12} /> Add Layer
              </button>
              {showAddMenu && (
                <div style={{ position: 'absolute', bottom: '100%', left: 10, right: 10, background: RAISED, border: `1px solid ${BORDER2}`, boxShadow: '0 -8px 24px rgba(0,0,0,0.12)', zIndex: 50 }}>
                  {['GPR Profiles','Point Annotations','Sketch','Notes'].map(opt => (
                    <button key={opt} onClick={() => { if (opt === 'GPR Profiles') fileInputRef.current?.click(); else setShowAddMenu(false); }}
                      style={{ display: 'block', width: '100%', textAlign: 'left', padding: '9px 14px', background: 'none', border: 'none', color: opt === 'GPR Profiles' ? TEXT : TEXT2, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.05)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
                      {opt}{opt !== 'GPR Profiles' && <span style={{ marginLeft: 8, fontSize: 9, color: TEXT2, opacity: 0.6 }}>soon</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}`, padding: 10 }}>
              <button onClick={() => setShowProjects(v => !v)}
                style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: showProjects ? 'rgba(232,96,28,0.1)' : 'none', border: 'none', color: showProjects ? ACCENT : TEXT2, cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'Inter, sans-serif', borderRadius: 3 }}
                onMouseEnter={e => { if (!showProjects) e.currentTarget.style.background='rgba(0,0,0,0.05)'; }}
                onMouseLeave={e => { if (!showProjects) e.currentTarget.style.background='none'; }}>
                <FolderOpen size={13} /> My Projects
              </button>
            </div>
          </Panel>

          <PanelResizeHandle style={{ width: 3, background: BORDER, cursor: 'col-resize' }} />

          {/* ── CENTER ──────────────────────────────────────────────────────────── */}
          <Panel style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
            <PanelGroup direction="vertical" style={{ flex: 1 }}>
              <Panel defaultSize={75} minSize={35} style={{ display: 'flex', overflow: 'hidden' }}>
                <PanelGroup direction="horizontal" style={{ flex: 1 }}>

                  {/* ── MAIN VIEWPORT ─────────────────────────────────────────── */}
                  <Panel style={{ position: 'relative', overflow: 'hidden', background: '#F5F3EF' }}>
                    <div ref={mapContainerRef} style={{ width: '100%', height: '100%', position: 'absolute', inset: 0, opacity: (activeView === 'cscan' && outputTab === 'gps') ? 1 : 0, pointerEvents: (activeView === 'cscan' && outputTab === 'gps') ? 'auto' : 'none', zIndex: (activeView === 'cscan' && outputTab === 'gps') ? 1 : 0, transition: 'opacity 0.2s' }} />
                    {!MAPBOX_TOKEN && activeView === 'cscan' && outputTab === 'gps' && (
                      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: PANEL, zIndex: 2 }}>
                        <p style={{ fontSize: 13, color: TEXT2 }}>Set <code style={{ background: RAISED, padding: '2px 6px', fontSize: 11 }}>VITE_MAPBOX_TOKEN</code> in .env.local.</p>
                      </div>
                    )}
                    {activeView === 'cscan' && outputTab === 'gps' && !hasResult && !isAnalyzing && files.length === 0 && MAPBOX_TOKEN && (
                      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', background: 'rgba(255,255,255,0.92)', border: `1.5px solid ${BORDER}`, padding: '28px 36px', textAlign: 'center', pointerEvents: 'none', backdropFilter: 'blur(4px)', zIndex: 3 }}>
                        <Radio size={28} style={{ color: '#B0A9A4', marginBottom: 12 }} />
                        <p style={{ fontSize: 14, fontWeight: 600, color: TEXT, margin: '0 0 6px' }}>Upload GPR profiles to begin analysis</p>
                        <p style={{ fontSize: 12, color: TEXT2, margin: 0 }}>Layers panel → Add Layer → GPR Profiles</p>
                      </div>
                    )}
                    {activeView === '3d' && (
                      <div style={{ position: 'absolute', inset: 0, zIndex: 5 }}>
                        {hasResult ? <ThreeDView perFileSummary={analysisResult!.per_file_summary} />
                          : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><p style={{ color: TEXT2, fontSize: 13 }}>Run an analysis to see the 3D view.</p></div>}
                      </div>
                    )}

                    {activeView === 'cscan' && (
                      <>
                        {/* Output tab bar */}
                        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10, height: 38, display: 'flex', alignItems: 'stretch', background: PANEL, borderBottom: `1px solid ${BORDER}` }}>
                          {OUTPUT_TABS.map(tab => {
                            const badge = tab.badge();
                            const disabled = tab.id !== 'gps' && !hasResult;
                            return (
                              <button key={tab.id} onClick={() => !disabled && setOutputTab(tab.id)}
                                style={{ padding: '0 14px', fontSize: 10, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', background: 'none', border: 'none', cursor: disabled ? 'default' : 'pointer', color: outputTab === tab.id ? ACCENT : (disabled ? '#C8C3BD' : TEXT2), borderBottom: `2px solid ${outputTab === tab.id ? ACCENT : 'transparent'}`, display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'Inter, sans-serif', transition: 'color 0.12s', whiteSpace: 'nowrap' }}>
                                {tab.label}
                                {badge && <span style={{ fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 8, background: badge.color + '20', color: badge.color, letterSpacing: '0.04em' }}>{badge.text}</span>}
                              </button>
                            );
                          })}
                        </div>

                        {outputTab !== 'gps' && (
                          <div style={{ position: 'absolute', inset: 0, top: 38, overflow: 'auto', background: PANEL, zIndex: 3 }}>
                            <OutputMaps
                              outputTab={outputTab} hasResult={hasResult} analysisResult={analysisResult}
                              useCondCanvas={useCondCanvas} condCanvasRef={condCanvasRef}
                              useRebarCanvas={useRebarCanvas} rebarCanvasRef={rebarCanvasRef}
                              useAmpCanvas={useAmpCanvas} ampCanvasRef={ampCanvasRef}
                              onExport={exportPNG}
                              condBadge={condBadge} depthBadge={depthBadge} ampBadge={ampBadge}
                            />
                          </div>
                        )}
                      </>
                    )}
                  </Panel>

                  <PanelResizeHandle style={{ width: 3, background: BORDER, cursor: 'col-resize' }} />

                  {/* ── RIGHT PANEL ───────────────────────────────────────────── */}
                  <Panel ref={rightPanelRef} defaultSize={22} minSize={0} collapsible collapsedSize={0}
                    style={{ background: PANEL, borderLeft: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <div style={{ display: 'flex', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
                      {(['properties','analysis'] as const).map(tab => (
                        <button key={tab} onClick={() => setRightTab(tab)}
                          style={{ flex: 1, padding: '10px 0', fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', background: 'none', border: 'none', cursor: 'pointer', color: rightTab === tab ? TEXT : TEXT2, borderBottom: `2px solid ${rightTab === tab ? ACCENT : 'transparent'}`, fontFamily: 'Inter, sans-serif', transition: 'color 0.12s' }}>
                          {tab}
                        </button>
                      ))}
                      <button onClick={() => rightPanelRef.current?.collapse()} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 10px', color: TEXT2 }}><ChevronRight size={13} /></button>
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
                      {rightTab === 'properties' && (
                        <>
                          {selectedLayer === 'gpr' && (
                            <div>
                              <div style={{ padding: '4px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2 }}>GPR Profiles</div>
                              {hasResult ? analysisResult!.per_file_summary.map((f, i) => (
                                <div key={f.filename} onClick={() => { setSelectedFileIdx(i); bottomPanelRef.current?.expand(); }}
                                  style={{ padding: '8px 14px', cursor: 'pointer', background: selectedFileIdx === i ? 'rgba(232,96,28,0.08)' : 'none', borderLeft: `2px solid ${selectedFileIdx === i ? ACCENT : 'transparent'}` }}
                                  onMouseEnter={e => { if (selectedFileIdx !== i) e.currentTarget.style.background='rgba(0,0,0,0.04)'; }}
                                  onMouseLeave={e => { if (selectedFileIdx !== i) e.currentTarget.style.background='none'; }}>
                                  <div style={{ fontSize: 11, color: TEXT, marginBottom: 4, fontFamily: 'monospace', wordBreak: 'break-all' }}>{f.filename}</div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                                    <span style={{ fontSize: 10, color: TEXT2 }}>{f.signals.toLocaleString()} signals</span>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                      <div style={{ width: 40, height: 4, background: '#E2DED9', borderRadius: 2, overflow: 'hidden' }}>
                                        <div style={{ height: '100%', width: `${f.delam_pct}%`, background: delamColor(f.delam_pct), borderRadius: 2 }} />
                                      </div>
                                      <span style={{ fontSize: 10, color: delamColor(f.delam_pct), fontWeight: 700, minWidth: 32, textAlign: 'right' }}>{f.delam_pct.toFixed(1)}%</span>
                                    </div>
                                  </div>
                                </div>
                              )) : <div style={{ padding: '24px 14px', textAlign: 'center' }}><p style={{ fontSize: 12, color: TEXT2 }}>No files analyzed yet.</p></div>}
                            </div>
                          )}
                          {selectedLayer === 'condition' && (
                            <div style={{ padding: '4px 14px' }}>
                              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 12 }}>Condition Grid</div>
                              <label style={{ display: 'block', fontSize: 10, color: TEXT2, marginBottom: 6 }}>Color scheme</label>
                              <select style={{ width: '100%', padding: '7px 10px', background: RAISED, border: `1px solid ${BORDER}`, color: TEXT, fontSize: 12, marginBottom: 16, fontFamily: 'Inter, sans-serif' }}>
                                <option>Green → Yellow → Red</option><option>Viridis</option><option>Grayscale</option>
                              </select>
                              <label style={{ display: 'block', fontSize: 10, color: TEXT2, marginBottom: 6 }}>Opacity: {conditionOpacity}%</label>
                              <input type="range" min={0} max={100} value={conditionOpacity} onChange={e => setConditionOpacity(+e.target.value)} style={{ width: '100%', accentColor: ACCENT, marginBottom: 16 }} />
                            </div>
                          )}
                          {selectedLayer !== 'gpr' && selectedLayer !== 'condition' && (
                            <div style={{ padding: '24px 14px', textAlign: 'center' }}><p style={{ fontSize: 12, color: TEXT2 }}>No configurable properties for this layer.</p></div>
                          )}
                          {setupDone && (
                            <div style={{ margin: '16px 14px 0', padding: '12px', background: RAISED, border: `1px solid ${BORDER}` }}>
                              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 8 }}>Setup</div>
                              {[
                                { label: 'Equipment', value: MANUFACTURERS.find(m => m.key === manufacturer)?.name || '—' },
                                { label: 'Frequency', value: `${frequencyMhz} MHz` },
                                { label: 'Bridge ID', value: bridgeId || '—' },
                                { label: 'Insp. Date', value: inspDate },
                              ].map(({ label, value }) => (
                                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                                  <span style={{ color: TEXT2 }}>{label}</span>
                                  <span style={{ color: TEXT, fontWeight: 600, maxWidth: 120, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{value}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )}

                      {rightTab === 'analysis' && (
                        <div style={{ padding: '0 14px' }}>
                          <div style={{ marginBottom: 20 }}>
                            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>AI Model</div>
                            <div style={{ background: RAISED, padding: '10px 12px', fontSize: 11 }}>
                              {[
                                { label: 'Version',   value: 'model_v13.pth' },
                                { label: 'Standard',  value: 'ASTM D6087' },
                                { label: 'Threshold', value: detectionThreshold.toFixed(2) },
                                { label: 'Frequency', value: `${frequencyMhz} MHz` },
                              ].map(({ label, value }) => (
                                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                  <span style={{ color: TEXT2 }}>{label}</span>
                                  <span style={{ color: TEXT, fontWeight: 600 }}>{value}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          {files.length === 0 && (
                            <button onClick={() => fileInputRef.current?.click()}
                              style={{ width: '100%', padding: '10px', marginBottom: 16, background: 'rgba(232,96,28,0.12)', border: `1px solid rgba(232,96,28,0.3)`, color: ACCENT, fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                              Upload GPR Files
                            </button>
                          )}
                          {(isAnalyzing || jobStatus === 'complete' || jobStatus === 'failed') && (
                            <div style={{ marginBottom: 16 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                {isAnalyzing && <Loader2 size={13} style={{ color: ACCENT, animation: 'spin 1s linear infinite' }} />}
                                {jobStatus === 'complete' && <Check size={13} style={{ color: '#22c55e' }} />}
                                {jobStatus === 'failed' && <AlertCircle size={13} style={{ color: '#ef4444' }} />}
                                <span style={{ fontSize: 12, color: isAnalyzing ? TEXT : jobStatus === 'complete' ? '#22c55e' : '#ef4444' }}>
                                  {isAnalyzing ? statusMsg : jobStatus === 'complete' ? 'Analysis complete' : 'Analysis failed'}
                                </span>
                              </div>
                              {isAnalyzing && <div style={{ height: 3, background: '#E2DED9', overflow: 'hidden', borderRadius: 2 }}><div style={{ height: '100%', width: '40%', background: ACCENT, borderRadius: 2, animation: 'verus-bar 1.8s ease-in-out infinite' }} /></div>}
                              {errorMsg && <p style={{ fontSize: 11, color: '#ef4444', marginTop: 6, lineHeight: 1.5 }}>{errorMsg}</p>}
                            </div>
                          )}
                          {files.length > 0 && jobStatus !== 'pending' && jobStatus !== 'processing' && (
                            <button onClick={() => { setJobStatus('idle'); startAnalysis(); }}
                              style={{ width: '100%', padding: '10px', marginBottom: 16, background: ACCENT, border: 'none', color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                              Re-run Analysis
                            </button>
                          )}
                          {hasResult && (
                            <div style={{ background: RAISED, padding: '12px' }}>
                              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>Summary</div>
                              {[
                                { label: 'Signals',       value: analysisResult!.signals_analyzed.toLocaleString() },
                                { label: 'Delamination',  value: `${analysisResult!.delamination_pct.toFixed(1)}%`, color: delamColor(analysisResult!.delamination_pct) },
                                { label: 'Sound',         value: `${analysisResult!.sound_pct.toFixed(1)}%` },
                                { label: 'Analysis time', value: `${analysisResult!.analysis_time_sec.toFixed(1)}s` },
                              ].map(({ label, value, color }) => (
                                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11 }}>
                                  <span style={{ color: TEXT2 }}>{label}</span>
                                  <span style={{ color: color || TEXT, fontWeight: 600 }}>{value}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <AdjustPanel
                      hasResult={hasResult} outputTab={outputTab} analysisResult={analysisResult}
                      detectionThreshold={detectionThreshold} setDetectionThreshold={setDetectionThreshold} setUseCondCanvas={setUseCondCanvas}
                      dielectricEr={dielectricEr} setDielectricEr={setDielectricEr} setUseRebarCanvas={setUseRebarCanvas}
                      ampClampMin={ampClampMin} setAmpClampMin={setAmpClampMin}
                      ampClampMax={ampClampMax} setAmpClampMax={setAmpClampMax} setUseAmpCanvas={setUseAmpCanvas}
                      projectId={projectId}
                    />

                    <div style={{ borderTop: `1px solid ${BORDER}`, padding: '6px 14px', flexShrink: 0 }}>
                      <button onClick={() => rightPanelRef.current?.collapse()}
                        style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                        <ChevronRight size={11} /> Collapse
                      </button>
                    </div>
                  </Panel>
                </PanelGroup>
              </Panel>

              <PanelResizeHandle style={{ height: 3, background: BORDER, cursor: 'row-resize' }} />

              {/* ── BOTTOM B-SCAN PANEL ───────────────────────────────────────── */}
              <Panel ref={bottomPanelRef} defaultSize={25} minSize={0} collapsible collapsedSize={0}
                style={{ background: PANEL, borderTop: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <div style={{ height: 36, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8, padding: '0 12px', borderBottom: `1px solid ${BORDER}`, background: RAISED }}>
                  <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2, flexShrink: 0 }}>B-Scan Profile</span>
                  {hasResult && totalFiles > 0 && (
                    <>
                      <div style={{ width: 1, height: 16, background: BORDER, flexShrink: 0 }} />
                      <button onClick={() => setSelectedFileIdx(i => Math.max(0, i-1))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 2 }}><ChevronLeft size={13} /></button>
                      <span style={{ fontSize: 11, color: TEXT, minWidth: 80, textAlign: 'center' }}>Swath {selectedFileIdx+1} / {totalFiles}</span>
                      <button onClick={() => setSelectedFileIdx(i => Math.min(totalFiles-1, i+1))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 2 }}><ChevronRight size={13} /></button>
                      <div style={{ width: 1, height: 16, background: BORDER }} />
                      {['In-Line','Cross'].map(t => (
                        <button key={t} style={{ padding: '2px 10px', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', background: t === 'In-Line' ? 'rgba(232,96,28,0.15)' : 'none', border: `1px solid ${t==='In-Line' ? 'rgba(232,96,28,0.3)' : BORDER}`, color: t === 'In-Line' ? ACCENT : TEXT2, cursor: 'pointer', fontFamily: 'Inter, sans-serif', borderRadius: 3 }}>{t}</button>
                      ))}
                    </>
                  )}
                  <div style={{ flex: 1 }} />
                  <button onClick={() => setBottomExpanded(v => !v)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}>
                    {bottomExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  </button>
                  <button onClick={() => bottomPanelRef.current?.collapse()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}><ChevronDown size={13} /></button>
                </div>
                <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#F5F3EF', position: 'relative' }}>
                  {hasResult ? (
                    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'stretch' }}>
                      <div style={{ width: 28, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, color: TEXT2, letterSpacing: '0.06em', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Travel time [ns]</div>
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                        <canvas ref={bscanCanvasRef} style={{ width: '100%', height: '100%', imageRendering: 'pixelated', display: 'block' }} />
                        <div style={{ height: 18, flexShrink: 0, textAlign: 'center', fontSize: 9, color: TEXT2, letterSpacing: '0.06em', paddingTop: 4 }}>Trace number</div>
                      </div>
                    </div>
                  ) : (
                    <p style={{ fontSize: 12, color: TEXT2, textAlign: 'center', padding: 16 }}>Select a GPR profile to view B-scan</p>
                  )}
                </div>
              </Panel>
            </PanelGroup>

            <button onClick={() => rightPanelRef.current?.expand()}
              style={{ position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)', background: RAISED, border: `1px solid ${BORDER2}`, borderRight: 'none', color: TEXT2, cursor: 'pointer', padding: '10px 4px', zIndex: 30, display: 'flex' }}
              title="Open properties panel"><ChevronLeft size={13} /></button>
          </Panel>
        </PanelGroup>
      </div>

      {/* ── MY PROJECTS DRAWER ──────────────────────────────────────────────── */}
      {showProjects && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'flex' }} onClick={() => setShowProjects(false)}>
          <div style={{ flex: 1 }} />
          <div style={{ width: 360, height: '100%', background: PANEL, borderLeft: `1px solid ${BORDER2}`, display: 'flex', flexDirection: 'column', boxShadow: '-8px 0 32px rgba(0,0,0,0.15)' }} onClick={e => e.stopPropagation()}>
            <div style={{ padding: '14px 20px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: TEXT }}>My Projects</span>
              <button onClick={() => setShowProjects(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2 }}><X size={16} /></button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {recentJobs.length === 0
                ? <div style={{ padding: 40, textAlign: 'center' }}><p style={{ fontSize: 13, color: TEXT2 }}>No completed analyses yet.</p></div>
                : recentJobs.map(job => (
                  <div key={job.id} onClick={() => loadJob(job)}
                    style={{ padding: '14px 20px', borderBottom: `1px solid ${BORDER}`, cursor: 'pointer' }}
                    onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: TEXT, marginBottom: 4 }}>
                      Analysis — {new Date(job.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </div>
                    <div style={{ display: 'flex', gap: 16, fontSize: 11, color: TEXT2 }}>
                      <span>{job.signals_analyzed?.toLocaleString() ?? '—'} signals</span>
                      <span style={{ color: delamColor(job.delamination_pct ?? 0) }}>{job.delamination_pct?.toFixed(1) ?? '—'}% delam</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes verus-bar { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
        .mapboxgl-ctrl-bottom-right { z-index: 1 !important; }
        .mapboxgl-ctrl-bottom-left  { z-index: 1 !important; }
      `}</style>
    </div>
  );
}
