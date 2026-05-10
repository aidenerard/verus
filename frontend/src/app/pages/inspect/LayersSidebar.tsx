/**
 * LayersSidebar.tsx
 * The 220px left sidebar: layer list with visibility toggles, per-file sub-list
 * with remove buttons, "Add Layer" menu, and "My Projects" button.
 *
 * Does NOT: contain map logic, canvas rendering, or analysis job management.
 */

import { Eye, EyeOff, Plus, FolderOpen, Loader2, Check, X } from 'lucide-react';

import { PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2, ACCENT, LAYER_DEFS } from './constants';
import type { LayerId } from './constants';
import type { UploadedFile } from './types';

interface LayersSidebarProps {
  selectedLayer: LayerId;
  setSelectedLayer: (id: LayerId) => void;
  setRightIconOpen: (v: 'properties' | 'analysis' | 'adjust' | null) => void;
  layerVis: Record<LayerId, boolean>;
  setLayerVis: (fn: (v: Record<LayerId, boolean>) => Record<LayerId, boolean>) => void;
  files: UploadedFile[];
  selectedFileIdx: number;
  setSelectedFileIdx: (i: number) => void;
  isAnalyzing: boolean;
  jobStatus: 'idle' | 'pending' | 'processing' | 'complete' | 'failed';
  setFiles: (fn: (prev: UploadedFile[]) => UploadedFile[]) => void;
  setJobStatus: (v: 'idle' | 'pending' | 'processing' | 'complete' | 'failed') => void;
  setAnalysisResult: (v: null) => void;
  setErrorMsg: (v: null) => void;
  showAddMenu: boolean;
  setShowAddMenu: (fn: (v: boolean) => boolean) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  showProjects: boolean;
  setShowProjects: (fn: (v: boolean) => boolean) => void;
}

export default function LayersSidebar({
  selectedLayer, setSelectedLayer, setRightIconOpen,
  layerVis, setLayerVis, files, selectedFileIdx, setSelectedFileIdx,
  isAnalyzing, jobStatus, setFiles, setJobStatus, setAnalysisResult, setErrorMsg,
  showAddMenu, setShowAddMenu, fileInputRef, showProjects, setShowProjects,
}: LayersSidebarProps) {
  return (
    <div style={{ width: 220, flexShrink: 0, background: PANEL, borderRight: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '10px 14px 8px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2 }}>Layers</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
        {LAYER_DEFS.map(({ id, label }) => (
          <div key={id} onClick={() => { setSelectedLayer(id); setRightIconOpen('properties'); }}
            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 14px', height: 36, cursor: 'pointer', background: selectedLayer === id ? 'rgba(232,96,28,0.1)' : 'none', borderLeft: `2px solid ${selectedLayer === id ? ACCENT : 'transparent'}`, transition: 'background 0.12s' }}
            onMouseEnter={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; }}
            onMouseLeave={e => { if (selectedLayer !== id) e.currentTarget.style.background = 'none'; }}
          >
            <span style={{ fontSize: 12, color: selectedLayer === id ? TEXT : TEXT2, flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {label}{id === 'gpr' && files.length > 0 && <span style={{ marginLeft: 6, fontSize: 10, color: TEXT2 }}>({files.length})</span>}
            </span>
            <button onClick={ev => { ev.stopPropagation(); setLayerVis(v => ({ ...v, [id]: !v[id as LayerId] })); }}
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: layerVis[id as LayerId] ? TEXT2 : '#B0A9A4', display: 'flex' }}>
              {layerVis[id as LayerId] ? <Eye size={13} /> : <EyeOff size={13} />}
            </button>
          </div>
        ))}
        {files.length > 0 && (
          <div style={{ borderTop: `1px solid ${BORDER}`, marginTop: 4 }}>
            {files.map((f, i) => (
              <div key={f.name} onClick={() => { setSelectedFileIdx(i); setSelectedLayer('gpr'); }}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 8px 5px 28px', height: 30, cursor: 'pointer', background: selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.04)')}
                onMouseLeave={e => (e.currentTarget.style.background = selectedFileIdx === i && selectedLayer === 'gpr' ? 'rgba(232,96,28,0.08)' : 'none')}
              >
                {isAnalyzing ? <Loader2 size={10} style={{ color: ACCENT, animation: 'spin 1s linear infinite', flexShrink: 0 }} />
                  : jobStatus === 'complete' ? <Check size={10} style={{ color: '#22c55e', flexShrink: 0 }} /> : null}
                <span style={{ fontSize: 11, color: TEXT2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>{f.name}</span>
                {!isAnalyzing && (
                  <button onClick={ev => {
                    ev.stopPropagation();
                    setFiles(prev => {
                      const next = prev.filter((_, j) => j !== i);
                      if (!next.length) { setJobStatus('idle'); setAnalysisResult(null); setErrorMsg(null); setSelectedFileIdx(0); }
                      else if (selectedFileIdx >= next.length) setSelectedFileIdx(next.length - 1);
                      return next;
                    });
                  }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: TEXT2, display: 'flex', flexShrink: 0, opacity: 0.5 }}
                    onMouseEnter={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.color = '#ef4444'; }}
                    onMouseLeave={e => { e.currentTarget.style.opacity = '0.5'; e.currentTarget.style.color = TEXT2; }}>
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
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '7px', background: 'rgba(0,0,0,0.03)', border: `1px dashed ${BORDER2}`, color: TEXT2, cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'Inter, sans-serif', borderRadius: 3 }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,0,0,0.05)'; e.currentTarget.style.color = TEXT; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(0,0,0,0.03)'; e.currentTarget.style.color = TEXT2; }}>
          <Plus size={12} /> Add Layer
        </button>
        {showAddMenu && (
          <div style={{ position: 'absolute', bottom: '100%', left: 10, right: 10, background: RAISED, border: `1px solid ${BORDER2}`, boxShadow: '0 -8px 24px rgba(0,0,0,0.1)', zIndex: 50 }}>
            {['Scan Lines', 'Notes'].map(opt => (
              <button key={opt} onClick={() => { if (opt === 'Scan Lines') fileInputRef.current?.click(); else setShowAddMenu(() => false); }}
                style={{ display: 'block', width: '100%', textAlign: 'left', padding: '9px 14px', background: 'none', border: 'none', color: opt === 'Scan Lines' ? TEXT : TEXT2, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.04)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
                {opt}{opt !== 'Scan Lines' && <span style={{ marginLeft: 8, fontSize: 9, color: TEXT2, opacity: 0.6 }}>soon</span>}
              </button>
            ))}
          </div>
        )}
      </div>
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: 10 }}>
        <button onClick={() => setShowProjects(v => !v)}
          style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px', background: showProjects ? 'rgba(232,96,28,0.1)' : 'none', border: 'none', color: showProjects ? ACCENT : TEXT2, cursor: 'pointer', fontSize: 11, fontWeight: 600, fontFamily: 'Inter, sans-serif', borderRadius: 3 }}
          onMouseEnter={e => { if (!showProjects) e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; }}
          onMouseLeave={e => { if (!showProjects) e.currentTarget.style.background = 'none'; }}>
          <FolderOpen size={13} /> My Projects
        </button>
      </div>
    </div>
  );
}
