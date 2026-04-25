/**
 * GPRWorkspace.tsx
 * Full-screen dark-themed GPR analysis workspace at /inspect/gpr.
 * Layout: Toolbar | LeftSidebar | (MainViewport / RightPanel) | BottomPanel
 */

import {
  useState, useEffect, useRef, useCallback,
} from 'react';
import { useNavigate } from 'react-router';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { ImperativePanelHandle } from 'react-resizable-panels';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  ArrowLeft, Eye, EyeOff, Plus, ChevronDown, ChevronLeft, ChevronRight,
  Layers, Download, X, FolderOpen, Loader2, Check, AlertCircle,
  ChevronUp, Maximize2, Minimize2, Radio,
} from 'lucide-react';
import ThreeDView from '../../components/ThreeDView';
import { useAuth } from '../../../context/AuthContext';
import { supabase } from '../../../lib/supabase';
import VerusLogo from '../../components/VerusLogo';

// ── Types ─────────────────────────────────────────────────────────────────────

interface BscanData { data: string; n_traces: number; n_samples: number }
interface GpsData {
  lat_start: number; lon_start: number;
  lat_end: number;   lon_end: number;
  coordinates: [number, number][];
}
interface FileResult {
  filename:  string;
  signals:   number;
  delam_pct: number;
  gps?:      GpsData | null;
  bscan?:    BscanData;
}
interface AnalysisResult {
  signals_analyzed:  number;
  delamination_pct:  number;
  sound_pct:         number;
  analysis_time_sec: number;
  cscan_image:       string;
  per_file_summary:  FileResult[];
}
interface UploadedFile { file: File; name: string }

// ── Constants ─────────────────────────────────────────────────────────────────

const SERVER = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : 'https://verus-server.onrender.com';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? '';
const DEFAULT_CENTER: [number, number] = [-73.9519, 40.8517];
const GPR_EXTS = new Set([
  '.csv', '.dzt', '.dt1', '.rd3', '.rd7', '.segy', '.sgy', '.dzg', '.hd', '.rad',
]);

const BG      = '#080d17';
const PANEL   = '#0c1525';
const RAISED  = '#111e34';
const BORDER  = 'rgba(255,255,255,0.07)';
const BORDER2 = 'rgba(255,255,255,0.14)';
const TEXT    = '#dde3f0';
const TEXT2   = '#4d6480';
const ACCENT  = '#E8601C';

const STATUS_MSGS = [
  'Uploading files…', 'Waking up AI model…',
  'Running inference…', 'Generating C-scan…', 'Almost done…',
];

const LAYER_DEFS = [
  { id: 'gpr',         label: 'GPR Profiles',      Icon: Radio },
  { id: 'condition',   label: 'Condition Grid',     Icon: Layers },
  { id: 'amplitude',   label: 'Amplitude Grid',     Icon: Layers },
  { id: 'satellite',   label: 'Satellite Image',    Icon: Layers },
  { id: 'annotations', label: 'Point Annotations',  Icon: Layers },
] as const;

type LayerId = typeof LAYER_DEFS[number]['id'];

// ── Helpers ───────────────────────────────────────────────────────────────────

function delamColor(pct: number) {
  const t = Math.min(1, Math.max(0, pct / 100));
  if (t <= 0.5) {
    const s = t * 2;
    const r = Math.round(0x22 + (0xf5 - 0x22) * s);
    const g = Math.round(0xc5 + (0x9e - 0xc5) * s);
    const b = Math.round(0x5e + (0x0b - 0x5e) * s);
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  }
  const s = (t - 0.5) * 2;
  const r = Math.round(0xf5 + (0xef - 0xf5) * s);
  const g = Math.round(0x9e + (0x44 - 0x9e) * s);
  const b = Math.round(0x0b + (0x44 - 0x0b) * s);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function GPRWorkspace() {
  const navigate  = useNavigate();
  const { session } = useAuth();

  // ── Project info ────────────────────────────────────────────────────────────
  const [projectName,    setProjectName]    = useState('New Project');
  const [structureName,  setStructureName]  = useState('Bridge Deck');
  const [editingProject,   setEditingProject]   = useState(false);
  const [editingStructure, setEditingStructure] = useState(false);

  // ── View ────────────────────────────────────────────────────────────────────
  const [activeView, setActiveView] = useState<'top' | '3d'>('top');
  const [rightTab,   setRightTab]   = useState<'properties' | 'analysis'>('analysis');
  const [bottomExpanded, setBottomExpanded] = useState(false);
  const [mouseCoords,    setMouseCoords]    = useState<{ x: number; y: number } | null>(null);

  // ── Layers ──────────────────────────────────────────────────────────────────
  const [selectedLayer, setSelectedLayer] = useState<LayerId>('gpr');
  const [layerVis, setLayerVis] = useState<Record<LayerId, boolean>>({
    gpr: true, condition: true, amplitude: false, satellite: true, annotations: true,
  });
  const [conditionOpacity, setConditionOpacity] = useState(80);
  const [showAddMenu,    setShowAddMenu]    = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showProjects,   setShowProjects]   = useState(false);

  // ── Analysis ────────────────────────────────────────────────────────────────
  const [files,          setFiles]          = useState<UploadedFile[]>([]);
  const [jobId,          setJobId]          = useState<string | null>(null);
  const [jobStatus,      setJobStatus]      = useState<'idle'|'pending'|'processing'|'complete'|'failed'>('idle');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMsg,       setErrorMsg]       = useState<string | null>(null);
  const [statusMsg,      setStatusMsg]      = useState('');
  const [selectedFileIdx, setSelectedFileIdx] = useState(0);
  const [recentJobs,     setRecentJobs]     = useState<any[]>([]);

  // ── Refs ─────────────────────────────────────────────────────────────────────
  const fileInputRef    = useRef<HTMLInputElement>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef          = useRef<mapboxgl.Map | null>(null);
  const bscanCanvasRef  = useRef<HTMLCanvasElement>(null);
  const pollRef         = useRef<ReturnType<typeof setInterval> | null>(null);
  const statusCycleRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const rightPanelRef   = useRef<ImperativePanelHandle>(null);
  const bottomPanelRef  = useRef<ImperativePanelHandle>(null);

  // ── Mapbox init ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current || !MAPBOX_TOKEN) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;

    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: DEFAULT_CENTER,
      zoom: 5,
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

  // ── Add / refresh condition overlay when result arrives ─────────────────────
  useEffect(() => {
    const map = analysisResult && mapRef.current ? mapRef.current : null;
    if (!map || !analysisResult) return;

    const add = () => {
      ['condition-fill', 'condition-hover'].forEach(id => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      if (map.getSource('condition')) map.removeSource('condition');

      const gpsFiles = analysisResult.per_file_summary.filter(f => f.gps);
      if (!gpsFiles.length) return;

      const features: GeoJSON.Feature<GeoJSON.LineString>[] = gpsFiles.map(f => ({
        type: 'Feature',
        properties: { filename: f.filename, delam_pct: f.delam_pct },
        geometry: {
          type: 'LineString',
          coordinates: f.gps!.coordinates.map(([lat, lon]) => [lon, lat]),
        },
      }));

      map.addSource('condition', { type: 'geojson', data: { type: 'FeatureCollection', features } });
      map.addLayer({
        id: 'condition-fill', type: 'line', source: 'condition',
        layout: { visibility: layerVis.condition ? 'visible' : 'none' },
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'delam_pct'],
            0, '#22c55e', 50, '#f59e0b', 100, '#ef4444'],
          'line-width': 5,
          'line-opacity': conditionOpacity / 100,
        },
      });

      const first = gpsFiles[0].gps!;
      map.flyTo({ center: [first.lon_start, first.lat_start], zoom: 17, duration: 1500 });
    };

    if (map.isStyleLoaded()) add(); else map.once('load', add);
  }, [analysisResult]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Sync layer visibility ────────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getLayer('condition-fill')) {
      map.setLayoutProperty('condition-fill', 'visibility', layerVis.condition ? 'visible' : 'none');
    }
  }, [layerVis.condition]);

  // ── Sync opacity ─────────────────────────────────────────────────────────────
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
        canvas.width  = 400;
        canvas.height = 120;
        ctx.fillStyle = PANEL;
        ctx.fillRect(0, 0, 400, 120);
        ctx.fillStyle = TEXT2;
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('B-scan data not available for this file', 200, 65);
      }
      return;
    }

    const { data, n_traces, n_samples } = fileResult.bscan;

    // Decode base64 → float32
    const binary = atob(data);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const floats = new Float32Array(bytes.buffer);

    // Normalize to 0-255
    let minVal = Infinity, maxVal = -Infinity;
    for (const v of floats) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
    const range = maxVal - minVal || 1;

    canvas.width  = n_traces;
    canvas.height = n_samples;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Grayscale radargram: column = trace, row = sample
    const imageData = ctx.createImageData(n_traces, n_samples);
    for (let t = 0; t < n_traces; t++) {
      for (let s = 0; s < n_samples; s++) {
        const v   = Math.round(((floats[t * n_samples + s] - minVal) / range) * 255);
        const idx = (s * n_traces + t) * 4;
        imageData.data[idx]     = v;
        imageData.data[idx + 1] = v;
        imageData.data[idx + 2] = v;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // Rebar pick line (max |amplitude| per trace) — magenta
    ctx.strokeStyle = '#ff3cff';
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
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

  // ── File upload handler ───────────────────────────────────────────────────────
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

  // ── Auto-trigger analysis when new files added ────────────────────────────────
  useEffect(() => {
    if (files.length > 0 && jobStatus === 'idle') {
      startAnalysis();
    }
  }, [files]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Analysis flow ─────────────────────────────────────────────────────────────
  const startAnalysis = useCallback(async () => {
    if (!files.length || jobStatus === 'pending' || jobStatus === 'processing') return;

    setJobStatus('pending');
    setErrorMsg(null);
    setStatusMsg(STATUS_MSGS[0]);

    let msgIdx = 0;
    statusCycleRef.current = setInterval(() => {
      msgIdx = (msgIdx + 1) % STATUS_MSGS.length;
      setStatusMsg(STATUS_MSGS[msgIdx]);
    }, 4000);

    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f.file));

      const headers: Record<string, string> = {};
      if (session?.access_token) headers['Authorization'] = `Bearer ${session.access_token}`;

      const res = await fetch(`${SERVER}/analyze`, {
        method: 'POST', headers, body: formData,
        signal: AbortSignal.timeout(150000), // 2.5 min — covers Render cold start + upload
      });

      clearInterval(statusCycleRef.current!);

      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.detail || j.error || msg; } catch {}
        setErrorMsg(msg);
        setJobStatus('failed');
        return;
      }

      const { job_id } = await res.json();
      setJobId(job_id);
      setJobStatus('processing');
      setStatusMsg('Processing GPR data…');

      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const jr = await fetch(`${SERVER}/job/${job_id}`, { headers });
          if (!jr.ok) return;
          const job = await jr.json();

          if (job.status === 'complete' && job.result) {
            clearInterval(pollRef.current!);
            setAnalysisResult(job.result);
            setJobStatus('complete');
            setRightTab('properties');
            setSelectedFileIdx(0);
            bottomPanelRef.current?.expand();
          } else if (job.status === 'failed') {
            clearInterval(pollRef.current!);
            setErrorMsg(job.error || 'Analysis failed');
            setJobStatus('failed');
          }
        } catch { /* poll silently */ }
      }, 3000);

    } catch (err) {
      clearInterval(statusCycleRef.current!);
      setErrorMsg(err instanceof Error ? err.message : 'Analysis failed');
      setJobStatus('failed');
    }
  }, [files, session, jobStatus]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (statusCycleRef.current) clearInterval(statusCycleRef.current);
  }, []);

  // ── Export PNG ────────────────────────────────────────────────────────────────
  const exportPNG = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const a   = document.createElement('a');
    a.download = `${projectName.replace(/\s+/g, '_')}_cscan.png`;
    a.href     = map.getCanvas().toDataURL();
    a.click();
    setShowExportMenu(false);
  }, [projectName]);

  // ── Load recent jobs for Projects drawer ────────────────────────────────────
  useEffect(() => {
    if (!showProjects) return;
    supabase.from('analysis_jobs').select('*').eq('status', 'complete')
      .order('created_at', { ascending: false }).limit(10)
      .then(({ data }) => setRecentJobs(data ?? []));
  }, [showProjects]);

  // ── Load a saved job back into workspace ────────────────────────────────────
  const loadJob = useCallback((job: any) => {
    if (!job.result) return;
    setAnalysisResult(job.result);
    setJobStatus('complete');
    setFiles([]);
    setSelectedFileIdx(0);
    setShowProjects(false);
    setRightTab('properties');
  }, []);

  // ── Derived ──────────────────────────────────────────────────────────────────
  const isAnalyzing   = jobStatus === 'pending' || jobStatus === 'processing';
  const hasResult     = analysisResult !== null;
  const selectedFile  = hasResult ? (analysisResult.per_file_summary[selectedFileIdx] ?? null) : null;
  const totalFiles    = analysisResult?.per_file_summary.length ?? 0;

  // ─────────────────────────────────────────────────────────────────────────────
  // ── RENDER ──────────────────────────────────────────────────────────────────
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      background: BG, color: TEXT, overflow: 'hidden',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      userSelect: 'none',
    }}>
      {/* Hidden file input */}
      <input
        ref={fileInputRef} type="file" multiple
        accept=".csv,.dzt,.DZT,.dt1,.DT1,.rd3,.rd7,.segy,.sgy,.dzg,.hd,.rad"
        onChange={onFileInput}
        style={{ display: 'none' }}
      />

      {/* ── TOP TOOLBAR ──────────────────────────────────────────────────────── */}
      <div style={{
        height: 48, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 12px', gap: 8,
        background: PANEL, borderBottom: `1px solid ${BORDER}`,
        position: 'relative', zIndex: 40,
      }}>
        {/* Left: back + project name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
          <button
            onClick={() => navigate('/dashboard')}
            title="Back to Dashboard"
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: TEXT2, padding: '6px 8px', display: 'flex', alignItems: 'center',
              borderRadius: 4, flexShrink: 0,
            }}
            onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
            onMouseLeave={e => (e.currentTarget.style.background = 'none')}
          >
            <ArrowLeft size={16} />
          </button>

          <div style={{ width: 1, height: 20, background: BORDER, flexShrink: 0 }} />

          <VerusLogo size={22} wordmarkColor="rgba(221,227,240,0.6)" />

          <div style={{ width: 1, height: 20, background: BORDER, flexShrink: 0 }} />

          {/* Project name */}
          {editingProject ? (
            <input
              autoFocus
              value={projectName}
              onChange={e => setProjectName(e.target.value)}
              onBlur={() => setEditingProject(false)}
              onKeyDown={e => e.key === 'Enter' && setEditingProject(false)}
              style={{
                background: RAISED, border: `1px solid ${BORDER2}`,
                color: TEXT, fontSize: 13, fontWeight: 600,
                padding: '3px 8px', outline: 'none', width: 160,
                fontFamily: 'Inter, sans-serif',
              }}
            />
          ) : (
            <span
              onClick={() => setEditingProject(true)}
              title="Click to rename"
              style={{
                fontSize: 13, fontWeight: 600, color: TEXT, cursor: 'text',
                padding: '3px 6px', borderRadius: 3,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 160,
              }}
              onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >
              {projectName}
            </span>
          )}

          <span style={{ color: TEXT2, fontSize: 12, flexShrink: 0 }}>/</span>

          {/* Structure name */}
          {editingStructure ? (
            <input
              autoFocus
              value={structureName}
              onChange={e => setStructureName(e.target.value)}
              onBlur={() => setEditingStructure(false)}
              onKeyDown={e => e.key === 'Enter' && setEditingStructure(false)}
              style={{
                background: RAISED, border: `1px solid ${BORDER2}`,
                color: TEXT2, fontSize: 12,
                padding: '3px 8px', outline: 'none', width: 130,
                fontFamily: 'Inter, sans-serif',
              }}
            />
          ) : (
            <span
              onClick={() => setEditingStructure(true)}
              title="Click to rename"
              style={{
                fontSize: 12, color: TEXT2, cursor: 'text',
                padding: '3px 6px', borderRadius: 3,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 130,
              }}
              onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >
              {structureName}
            </span>
          )}
        </div>

        {/* Center: view toggle + coord readout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          <div style={{
            display: 'flex', background: RAISED, border: `1px solid ${BORDER}`, borderRadius: 6,
            overflow: 'hidden',
          }}>
            {(['top', '3d'] as const).map(v => (
              <button key={v}
                onClick={() => setActiveView(v)}
                style={{
                  padding: '5px 16px', fontSize: 11, fontWeight: 700,
                  letterSpacing: '0.06em', textTransform: 'uppercase',
                  background: activeView === v ? 'rgba(232,96,28,0.18)' : 'none',
                  color: activeView === v ? ACCENT : TEXT2,
                  border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  transition: 'background 0.15s, color 0.15s',
                }}
              >
                {v === 'top' ? 'Top' : '3D'}
              </button>
            ))}
          </div>

          {mouseCoords && (
            <span style={{ fontSize: 11, color: TEXT2, fontVariantNumeric: 'tabular-nums', minWidth: 130 }}>
              X {mouseCoords.x.toLocaleString()} ft &nbsp;|&nbsp; Y {mouseCoords.y.toLocaleString()} ft
            </span>
          )}
        </div>

        {/* Right: export */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, justifyContent: 'flex-end', position: 'relative' }}>
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowExportMenu(v => !v)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 14px', background: RAISED, border: `1px solid ${BORDER2}`,
                color: TEXT, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em',
                textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                borderRadius: 4,
              }}
            >
              <Download size={13} /> Export <ChevronDown size={11} />
            </button>
            {showExportMenu && (
              <div style={{
                position: 'absolute', top: '100%', right: 0, marginTop: 4,
                background: RAISED, border: `1px solid ${BORDER2}`,
                zIndex: 100, minWidth: 180, boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
              }}>
                {[
                  { label: 'Export PNG',        action: exportPNG },
                  { label: 'Export PDF Report', action: () => setShowExportMenu(false) },
                  { label: 'Export CSV Data',   action: () => setShowExportMenu(false) },
                ].map(({ label, action }) => (
                  <button key={label} onClick={action} style={{
                    display: 'block', width: '100%', textAlign: 'left',
                    padding: '10px 16px', background: 'none', border: 'none',
                    color: TEXT, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.06)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── MAIN CONTENT ─────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <PanelGroup direction="horizontal" style={{ height: '100%' }}>

          {/* ── LEFT SIDEBAR ─────────────────────────────────────────────────── */}
          <Panel defaultSize={18} minSize={12} maxSize={28}
            style={{ background: PANEL, borderRight: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
          >
            {/* Layers header */}
            <div style={{
              padding: '12px 14px 10px',
              borderBottom: `1px solid ${BORDER}`,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2 }}>
                Layers
              </span>
            </div>

            {/* Layer list */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
              {LAYER_DEFS.map(({ id, label }) => (
                <div key={id}
                  onClick={() => { setSelectedLayer(id); setRightTab('properties'); }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '7px 14px', cursor: 'pointer',
                    background: selectedLayer === id ? 'rgba(232,96,28,0.1)' : 'none',
                    borderLeft: `2px solid ${selectedLayer === id ? ACCENT : 'transparent'}`,
                    transition: 'background 0.12s',
                  }}
                  onMouseEnter={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
                  onMouseLeave={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'none'; }}
                >
                  <span style={{ fontSize: 12, color: selectedLayer === id ? TEXT : TEXT2, flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {label}
                    {id === 'gpr' && files.length > 0 && (
                      <span style={{ marginLeft: 6, fontSize: 10, color: TEXT2 }}>
                        ({files.length})
                      </span>
                    )}
                  </span>
                  <button
                    onClick={ev => { ev.stopPropagation(); setLayerVis(v => ({ ...v, [id]: !v[id as LayerId] })); }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: layerVis[id as LayerId] ? TEXT2 : 'rgba(255,255,255,0.18)', display: 'flex' }}
                    title={layerVis[id as LayerId] ? 'Hide layer' : 'Show layer'}
                  >
                    {layerVis[id as LayerId] ? <Eye size={13} /> : <EyeOff size={13} />}
                  </button>
                </div>
              ))}

              {/* GPR file sub-list */}
              {files.length > 0 && (
                <div style={{ borderTop: `1px solid ${BORDER}`, marginTop: 4 }}>
                  {files.map((f, i) => (
                    <div key={f.name}
                      onClick={() => { setSelectedFileIdx(i); setSelectedLayer('gpr'); }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        padding: '5px 14px 5px 28px', cursor: 'pointer',
                        background: selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
                      onMouseLeave={e => (e.currentTarget.style.background = selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none')}
                    >
                      {isAnalyzing && jobStatus !== 'complete' ? (
                        <Loader2 size={10} style={{ color: ACCENT, animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                      ) : jobStatus === 'complete' ? (
                        <Check size={10} style={{ color: '#22c55e', flexShrink: 0 }} />
                      ) : null}
                      <span style={{ fontSize: 11, color: TEXT2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {f.name}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Add Layer button */}
            <div style={{ borderTop: `1px solid ${BORDER}`, padding: 10, position: 'relative' }}>
              <button
                onClick={() => setShowAddMenu(v => !v)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  padding: '8px', background: 'rgba(255,255,255,0.04)',
                  border: `1px dashed ${BORDER2}`, color: TEXT2, cursor: 'pointer',
                  fontSize: 11, fontWeight: 600, fontFamily: 'Inter, sans-serif',
                  borderRadius: 3, transition: 'background 0.12s, color 0.12s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = TEXT; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = TEXT2; }}
              >
                <Plus size={12} /> Add Layer
              </button>

              {showAddMenu && (
                <div style={{
                  position: 'absolute', bottom: '100%', left: 10, right: 10,
                  background: RAISED, border: `1px solid ${BORDER2}`,
                  boxShadow: '0 -8px 24px rgba(0,0,0,0.5)', zIndex: 50,
                }}>
                  {['GPR Profiles', 'Point Annotations', 'Sketch', 'Notes'].map(opt => (
                    <button key={opt}
                      onClick={() => {
                        if (opt === 'GPR Profiles') fileInputRef.current?.click();
                        else setShowAddMenu(false);
                      }}
                      style={{
                        display: 'block', width: '100%', textAlign: 'left',
                        padding: '9px 14px', background: 'none', border: 'none',
                        color: opt === 'GPR Profiles' ? TEXT : TEXT2,
                        fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.06)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                    >
                      {opt}
                      {opt !== 'GPR Profiles' && (
                        <span style={{ marginLeft: 8, fontSize: 9, color: TEXT2, opacity: 0.6 }}>soon</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* My Projects button */}
            <div style={{ borderTop: `1px solid ${BORDER}`, padding: 10 }}>
              <button
                onClick={() => setShowProjects(v => !v)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                  padding: '8px 10px', background: showProjects ? 'rgba(232,96,28,0.1)' : 'none',
                  border: 'none', color: showProjects ? ACCENT : TEXT2,
                  cursor: 'pointer', fontSize: 11, fontWeight: 600,
                  fontFamily: 'Inter, sans-serif', borderRadius: 3,
                }}
                onMouseEnter={e => { if (!showProjects) e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; }}
                onMouseLeave={e => { if (!showProjects) e.currentTarget.style.background = 'none'; }}
              >
                <FolderOpen size={13} /> My Projects
              </button>
            </div>
          </Panel>

          <PanelResizeHandle style={{ width: 3, background: BORDER, cursor: 'col-resize', transition: 'background 0.15s' }}
            onDragging={d => { const el = document.activeElement as HTMLElement; if (el) el.style.background = d ? BORDER2 : BORDER; }}
          />

          {/* ── CENTER (viewport + right panel + bottom panel) ─────────────── */}
          <Panel style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
            <PanelGroup direction="vertical" style={{ flex: 1 }}>

              {/* Top row: main viewport + right panel */}
              <Panel defaultSize={75} minSize={35} style={{ display: 'flex', overflow: 'hidden' }}>
                <PanelGroup direction="horizontal" style={{ flex: 1 }}>

                  {/* ── MAIN VIEWPORT ─────────────────────────────────────── */}
                  <Panel style={{ position: 'relative', overflow: 'hidden', background: '#050b14' }}>

                    {/* Mapbox container — always in DOM, hidden when 3D active */}
                    <div
                      ref={mapContainerRef}
                      style={{ width: '100%', height: '100%', position: 'absolute', inset: 0,
                        opacity: activeView === 'top' ? 1 : 0,
                        pointerEvents: activeView === 'top' ? 'auto' : 'none',
                        transition: 'opacity 0.3s',
                      }}
                    />

                    {/* No-token fallback */}
                    {!MAPBOX_TOKEN && activeView === 'top' && (
                      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: PANEL }}>
                        <div style={{ textAlign: 'center', padding: 32 }}>
                          <p style={{ fontSize: 13, color: TEXT2 }}>
                            Set <code style={{ background: RAISED, padding: '2px 6px', fontSize: 11 }}>VITE_MAPBOX_TOKEN</code> in .env.local to enable the map.
                          </p>
                        </div>
                      </div>
                    )}

                    {/* No data overlay */}
                    {!hasResult && !isAnalyzing && files.length === 0 && activeView === 'top' && MAPBOX_TOKEN && (
                      <div style={{
                        position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
                        background: 'rgba(12,21,37,0.88)', border: `1px solid ${BORDER2}`,
                        padding: '28px 36px', textAlign: 'center', pointerEvents: 'none',
                        backdropFilter: 'blur(4px)',
                      }}>
                        <Radio size={28} style={{ color: TEXT2, marginBottom: 12 }} />
                        <p style={{ fontSize: 14, fontWeight: 600, color: TEXT, margin: '0 0 6px' }}>
                          Upload GPR profiles to begin analysis
                        </p>
                        <p style={{ fontSize: 12, color: TEXT2, margin: 0 }}>
                          Use the Layers panel → Add Layer → GPR Profiles
                        </p>
                      </div>
                    )}

                    {/* 3D view */}
                    {activeView === '3d' && (
                      <div style={{ position: 'absolute', inset: 0, transition: 'opacity 0.3s' }}>
                        {hasResult ? (
                          <ThreeDView perFileSummary={analysisResult!.per_file_summary} />
                        ) : (
                          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <p style={{ color: TEXT2, fontSize: 13 }}>Run an analysis to see the 3D view.</p>
                          </div>
                        )}
                      </div>
                    )}
                  </Panel>

                  <PanelResizeHandle style={{ width: 3, background: BORDER, cursor: 'col-resize' }} />

                  {/* ── RIGHT PANEL ─────────────────────────────────────────── */}
                  <Panel
                    ref={rightPanelRef}
                    defaultSize={22} minSize={0} collapsible collapsedSize={0}
                    style={{ background: PANEL, borderLeft: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
                  >
                    {/* Tabs */}
                    <div style={{ display: 'flex', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
                      {(['properties', 'analysis'] as const).map(tab => (
                        <button key={tab}
                          onClick={() => setRightTab(tab)}
                          style={{
                            flex: 1, padding: '10px 0', fontSize: 10, fontWeight: 700,
                            letterSpacing: '0.08em', textTransform: 'uppercase',
                            background: 'none', border: 'none', cursor: 'pointer',
                            color: rightTab === tab ? TEXT : TEXT2,
                            borderBottom: `2px solid ${rightTab === tab ? ACCENT : 'transparent'}`,
                            fontFamily: 'Inter, sans-serif', transition: 'color 0.12s',
                          }}
                        >
                          {tab}
                        </button>
                      ))}
                      <button
                        onClick={() => rightPanelRef.current?.collapse()}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 10px', color: TEXT2 }}
                        title="Collapse panel"
                      >
                        <ChevronRight size={13} />
                      </button>
                    </div>

                    {/* Panel body */}
                    <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>

                      {/* ── PROPERTIES tab ─────────────────────────────────── */}
                      {rightTab === 'properties' && (
                        <>
                          {/* File list when GPR layer selected */}
                          {selectedLayer === 'gpr' && (
                            <div>
                              <div style={{ padding: '4px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2 }}>
                                GPR Profiles
                              </div>
                              {hasResult ? analysisResult!.per_file_summary.map((f, i) => (
                                <div key={f.filename}
                                  onClick={() => { setSelectedFileIdx(i); bottomPanelRef.current?.expand(); }}
                                  style={{
                                    padding: '8px 14px', cursor: 'pointer',
                                    background: selectedFileIdx === i ? 'rgba(232,96,28,0.08)' : 'none',
                                    borderLeft: `2px solid ${selectedFileIdx === i ? ACCENT : 'transparent'}`,
                                  }}
                                  onMouseEnter={e => { if (selectedFileIdx !== i) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
                                  onMouseLeave={e => { if (selectedFileIdx !== i) e.currentTarget.style.background = 'none'; }}
                                >
                                  <div style={{ fontSize: 11, color: TEXT, marginBottom: 4, fontFamily: 'monospace', wordBreak: 'break-all' }}>
                                    {f.filename}
                                  </div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                                    <span style={{ fontSize: 10, color: TEXT2 }}>{f.signals.toLocaleString()} signals</span>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                      <div style={{ width: 40, height: 4, background: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                                        <div style={{ height: '100%', width: `${f.delam_pct}%`, background: delamColor(f.delam_pct), borderRadius: 2 }} />
                                      </div>
                                      <span style={{ fontSize: 10, color: delamColor(f.delam_pct), fontWeight: 700, minWidth: 32, textAlign: 'right' }}>
                                        {f.delam_pct.toFixed(1)}%
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              )) : (
                                <div style={{ padding: '24px 14px', textAlign: 'center' }}>
                                  <p style={{ fontSize: 12, color: TEXT2 }}>No files analyzed yet.</p>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Condition Grid properties */}
                          {selectedLayer === 'condition' && (
                            <div style={{ padding: '4px 14px' }}>
                              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 12 }}>
                                Condition Grid
                              </div>

                              <label style={{ display: 'block', fontSize: 10, color: TEXT2, marginBottom: 6 }}>
                                Color scheme
                              </label>
                              <select style={{
                                width: '100%', padding: '7px 10px', background: RAISED,
                                border: `1px solid ${BORDER}`, color: TEXT, fontSize: 12,
                                marginBottom: 16, fontFamily: 'Inter, sans-serif',
                              }}>
                                <option>Green → Yellow → Red</option>
                                <option>Viridis</option>
                                <option>Grayscale</option>
                              </select>

                              <label style={{ display: 'block', fontSize: 10, color: TEXT2, marginBottom: 6 }}>
                                Opacity: {conditionOpacity}%
                              </label>
                              <input
                                type="range" min={0} max={100} value={conditionOpacity}
                                onChange={e => setConditionOpacity(+e.target.value)}
                                style={{ width: '100%', accentColor: ACCENT, marginBottom: 16 }}
                              />

                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: TEXT2 }}>
                                <span>Threshold</span>
                                <span style={{ color: TEXT, fontWeight: 600 }}>0.65</span>
                              </div>
                            </div>
                          )}

                          {/* Other layers */}
                          {selectedLayer !== 'gpr' && selectedLayer !== 'condition' && (
                            <div style={{ padding: '24px 14px', textAlign: 'center' }}>
                              <p style={{ fontSize: 12, color: TEXT2 }}>
                                No configurable properties for this layer.
                              </p>
                            </div>
                          )}
                        </>
                      )}

                      {/* ── ANALYSIS tab ────────────────────────────────────── */}
                      {rightTab === 'analysis' && (
                        <div style={{ padding: '0 14px' }}>
                          {/* Model info */}
                          <div style={{ marginBottom: 20 }}>
                            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>
                              AI Model
                            </div>
                            <div style={{ background: RAISED, padding: '10px 12px', fontSize: 11 }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                <span style={{ color: TEXT2 }}>Version</span>
                                <span style={{ color: TEXT, fontWeight: 600 }}>model_v13.pth</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                <span style={{ color: TEXT2 }}>Standard</span>
                                <span style={{ color: TEXT }}>ASTM D6087</span>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: TEXT2 }}>Threshold</span>
                                <span style={{ color: TEXT }}>0.65</span>
                              </div>
                            </div>
                          </div>

                          {/* Upload trigger */}
                          {files.length === 0 && (
                            <button
                              onClick={() => fileInputRef.current?.click()}
                              style={{
                                width: '100%', padding: '10px', marginBottom: 16,
                                background: 'rgba(232,96,28,0.12)', border: `1px solid rgba(232,96,28,0.3)`,
                                color: ACCENT, fontSize: 11, fontWeight: 700, letterSpacing: '0.07em',
                                textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                              }}
                            >
                              Upload GPR Files
                            </button>
                          )}

                          {/* Status indicator */}
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

                              {isAnalyzing && (
                                <div style={{ height: 3, background: 'rgba(255,255,255,0.08)', overflow: 'hidden', borderRadius: 2 }}>
                                  <div style={{
                                    height: '100%', width: '40%', background: ACCENT, borderRadius: 2,
                                    animation: 'verus-bar 1.8s ease-in-out infinite',
                                  }} />
                                </div>
                              )}

                              {errorMsg && (
                                <p style={{ fontSize: 11, color: '#ef4444', marginTop: 6, lineHeight: 1.5 }}>{errorMsg}</p>
                              )}
                            </div>
                          )}

                          {/* Re-run button */}
                          {files.length > 0 && jobStatus !== 'pending' && jobStatus !== 'processing' && (
                            <button
                              onClick={() => { setJobStatus('idle'); startAnalysis(); }}
                              style={{
                                width: '100%', padding: '10px', marginBottom: 16,
                                background: ACCENT, border: 'none', color: '#fff',
                                fontSize: 11, fontWeight: 700, letterSpacing: '0.07em',
                                textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                              }}
                            >
                              Re-run Analysis
                            </button>
                          )}

                          {/* Summary stats */}
                          {hasResult && (
                            <div style={{ background: RAISED, padding: '12px' }}>
                              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>
                                Summary
                              </div>
                              {[
                                { label: 'Signals',      value: analysisResult!.signals_analyzed.toLocaleString() },
                                { label: 'Delamination', value: `${analysisResult!.delamination_pct.toFixed(1)}%`, color: delamColor(analysisResult!.delamination_pct) },
                                { label: 'Sound',        value: `${analysisResult!.sound_pct.toFixed(1)}%` },
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

                    {/* Collapse handle at bottom */}
                    <div style={{ borderTop: `1px solid ${BORDER}`, padding: '6px 14px', flexShrink: 0 }}>
                      <button
                        onClick={() => rightPanelRef.current?.collapse()}
                        style={{
                          width: '100%', background: 'none', border: 'none', cursor: 'pointer',
                          color: TEXT2, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
                        }}
                      >
                        <ChevronRight size={11} /> Collapse
                      </button>
                    </div>
                  </Panel>
                </PanelGroup>
              </Panel>

              <PanelResizeHandle style={{ height: 3, background: BORDER, cursor: 'row-resize' }} />

              {/* ── BOTTOM B-SCAN PANEL ─────────────────────────────────────── */}
              <Panel
                ref={bottomPanelRef}
                defaultSize={25} minSize={0} collapsible collapsedSize={0}
                style={{ background: PANEL, borderTop: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
              >
                {/* B-scan header */}
                <div style={{
                  height: 36, flexShrink: 0,
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '0 12px', borderBottom: `1px solid ${BORDER}`,
                  background: RAISED,
                }}>
                  <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2, flexShrink: 0 }}>
                    B-Scan Profile
                  </span>

                  {hasResult && totalFiles > 0 && (
                    <>
                      <div style={{ width: 1, height: 16, background: BORDER, flexShrink: 0 }} />
                      <button onClick={() => setSelectedFileIdx(i => Math.max(0, i - 1))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 2 }}>
                        <ChevronLeft size={13} />
                      </button>
                      <span style={{ fontSize: 11, color: TEXT, minWidth: 80, textAlign: 'center' }}>
                        Swath {selectedFileIdx + 1} / {totalFiles}
                      </span>
                      <button onClick={() => setSelectedFileIdx(i => Math.min(totalFiles - 1, i + 1))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 2 }}>
                        <ChevronRight size={13} />
                      </button>

                      <div style={{ width: 1, height: 16, background: BORDER }} />
                      {['In-Line', 'Cross'].map(t => (
                        <button key={t} style={{
                          padding: '2px 10px', fontSize: 10, fontWeight: 700,
                          letterSpacing: '0.06em', textTransform: 'uppercase',
                          background: t === 'In-Line' ? 'rgba(232,96,28,0.15)' : 'none',
                          border: `1px solid ${t === 'In-Line' ? 'rgba(232,96,28,0.3)' : BORDER}`,
                          color: t === 'In-Line' ? ACCENT : TEXT2, cursor: 'pointer',
                          fontFamily: 'Inter, sans-serif', borderRadius: 3,
                        }}>
                          {t}
                        </button>
                      ))}
                    </>
                  )}

                  <div style={{ flex: 1 }} />

                  <button
                    onClick={() => setBottomExpanded(v => !v)}
                    title={bottomExpanded ? 'Shrink panel' : 'Expand panel'}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}
                  >
                    {bottomExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
                  </button>
                  <button
                    onClick={() => bottomPanelRef.current?.collapse()}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}
                    title="Collapse B-scan"
                  >
                    <ChevronDown size={13} />
                  </button>
                </div>

                {/* Canvas area */}
                <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#050b14', position: 'relative' }}>
                  {hasResult ? (
                    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'stretch' }}>
                      {/* Y-axis label */}
                      <div style={{
                        width: 28, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 9, color: TEXT2, letterSpacing: '0.06em',
                        writingMode: 'vertical-rl', transform: 'rotate(180deg)',
                      }}>
                        Travel time [ns]
                      </div>
                      {/* Canvas wrapper */}
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                        <canvas
                          ref={bscanCanvasRef}
                          style={{ width: '100%', height: '100%', imageRendering: 'pixelated', display: 'block' }}
                        />
                        {/* X-axis label */}
                        <div style={{ height: 18, flexShrink: 0, textAlign: 'center', fontSize: 9, color: TEXT2, letterSpacing: '0.06em', paddingTop: 4 }}>
                          Trace number
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p style={{ fontSize: 12, color: TEXT2, textAlign: 'center', padding: 16 }}>
                      Select a GPR profile from the layers panel to view B-scan
                    </p>
                  )}
                </div>
              </Panel>
            </PanelGroup>

            {/* Expand right panel button when collapsed */}
            <button
              onClick={() => rightPanelRef.current?.expand()}
              style={{
                position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                background: RAISED, border: `1px solid ${BORDER2}`,
                borderRight: 'none', color: TEXT2, cursor: 'pointer',
                padding: '10px 4px', zIndex: 30, display: 'flex',
              }}
              title="Open properties panel"
            >
              <ChevronLeft size={13} />
            </button>
          </Panel>
        </PanelGroup>
      </div>

      {/* ── MY PROJECTS DRAWER ────────────────────────────────────────────────── */}
      {showProjects && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 200,
          display: 'flex',
        }}
          onClick={() => setShowProjects(false)}
        >
          <div style={{ flex: 1 }} />
          <div
            style={{
              width: 360, height: '100%', background: PANEL,
              borderLeft: `1px solid ${BORDER2}`, display: 'flex', flexDirection: 'column',
              boxShadow: '-8px 0 32px rgba(0,0,0,0.6)',
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              padding: '14px 20px', borderBottom: `1px solid ${BORDER}`,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: TEXT }}>My Projects</span>
              <button onClick={() => setShowProjects(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2 }}>
                <X size={16} />
              </button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto' }}>
              {recentJobs.length === 0 ? (
                <div style={{ padding: 40, textAlign: 'center' }}>
                  <p style={{ fontSize: 13, color: TEXT2 }}>No completed analyses yet.</p>
                </div>
              ) : recentJobs.map(job => (
                <div key={job.id}
                  onClick={() => loadJob(job)}
                  style={{
                    padding: '14px 20px', borderBottom: `1px solid ${BORDER}`,
                    cursor: 'pointer',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                >
                  <div style={{ fontSize: 12, fontWeight: 600, color: TEXT, marginBottom: 4 }}>
                    Analysis — {new Date(job.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                  </div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 11, color: TEXT2 }}>
                    <span>{job.signals_analyzed?.toLocaleString() ?? '—'} signals</span>
                    <span style={{ color: delamColor(job.delamination_pct ?? 0) }}>
                      {job.delamination_pct?.toFixed(1) ?? '—'}% delam
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── CSS animations ───────────────────────────────────────────────────── */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes verus-bar {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
        .mapboxgl-ctrl-bottom-right { z-index: 1 !important; }
        .mapboxgl-ctrl-bottom-left  { z-index: 1 !important; }
      `}</style>
    </div>
  );
}
