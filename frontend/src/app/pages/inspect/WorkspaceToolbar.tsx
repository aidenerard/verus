/**
 * WorkspaceToolbar.tsx
 * The 44px top toolbar: back button, project/structure name editing,
 * view toggle (Maps / 3D), coordinate display, settings menu, and export menu.
 *
 * Does NOT: contain analysis logic, layer management, or canvas rendering.
 */

import { ArrowLeft, Download, Settings } from 'lucide-react';
import type { NavigateFunction } from 'react-router';

import { BG, PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2, ACCENT } from './constants';
import type { OutputTab } from './types';

interface WorkspaceToolbarProps {
  navigate: NavigateFunction;
  editingProject: boolean;
  setEditingProject: (v: boolean) => void;
  projectName: string;
  setProjectName: (v: string) => void;
  editingStructure: boolean;
  setEditingStructure: (v: boolean) => void;
  structureName: string;
  setStructureName: (v: string) => void;
  activeView: 'cscan' | '3d';
  setActiveView: (v: 'cscan' | '3d') => void;
  mouseCoords: { x: number; y: number } | null;
  outputTab: OutputTab;
  setupDone: boolean;
  showSettingsMenu: boolean;
  setShowSettingsMenu: (fn: (v: boolean) => boolean) => void;
  newProject: () => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  setRightIconOpen: (v: 'properties' | 'analysis' | 'adjust' | null) => void;
  showExportMenu: boolean;
  setShowExportMenu: (fn: (v: boolean) => boolean) => void;
  exportPNG: () => void;
  setSetupDone: (v: boolean) => void;
  setSetupStep: (v: 1 | 2 | 3) => void;
}

export default function WorkspaceToolbar({
  navigate, editingProject, setEditingProject, projectName, setProjectName,
  editingStructure, setEditingStructure, structureName, setStructureName,
  activeView, setActiveView, mouseCoords, outputTab, setupDone,
  showSettingsMenu, setShowSettingsMenu, newProject, fileInputRef,
  setRightIconOpen, showExportMenu, setShowExportMenu, exportPNG,
  setSetupDone, setSetupStep,
}: WorkspaceToolbarProps) {
  return (
    <div style={{ height: 44, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px', gap: 8, background: BG, borderBottom: `1px solid ${BORDER}`, position: 'relative', zIndex: 40 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
        <button onClick={() => navigate('/dashboard')} title="Back"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: '5px 7px', display: 'flex', alignItems: 'center', borderRadius: 4, flexShrink: 0 }}
          onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
          onMouseLeave={e => (e.currentTarget.style.background = 'none')}
        ><ArrowLeft size={15} /></button>
        <div style={{ width: 1, height: 18, background: BORDER, flexShrink: 0 }} />
        {editingProject
          ? <input autoFocus value={projectName} onChange={e => setProjectName(e.target.value)}
              onBlur={() => setEditingProject(false)} onKeyDown={e => e.key === 'Enter' && setEditingProject(false)}
              style={{ background: RAISED, border: `1px solid ${BORDER2}`, color: TEXT, fontSize: 13, fontWeight: 600, padding: '3px 8px', outline: 'none', width: 160, fontFamily: 'Inter, sans-serif' }} />
          : <span onClick={() => setEditingProject(true)} title="Rename"
              style={{ fontSize: 13, fontWeight: 600, color: TEXT, cursor: 'text', padding: '3px 6px', borderRadius: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 180 }}
              onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >{projectName}</span>
        }
        <span style={{ color: TEXT2, fontSize: 11, flexShrink: 0 }}>/</span>
        {editingStructure
          ? <input autoFocus value={structureName} onChange={e => setStructureName(e.target.value)}
              onBlur={() => setEditingStructure(false)} onKeyDown={e => e.key === 'Enter' && setEditingStructure(false)}
              style={{ background: RAISED, border: `1px solid ${BORDER2}`, color: TEXT2, fontSize: 12, padding: '3px 8px', outline: 'none', width: 130, fontFamily: 'Inter, sans-serif' }} />
          : <span onClick={() => setEditingStructure(true)} title="Rename"
              style={{ fontSize: 12, color: TEXT2, cursor: 'text', padding: '3px 6px', borderRadius: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 150 }}
              onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >{structureName}</span>
        }
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
        <div style={{ display: 'flex', background: RAISED, border: `1px solid ${BORDER}`, borderRadius: 6, overflow: 'hidden' }}>
          {(['cscan', '3d'] as const).map(v => (
            <button key={v} onClick={() => setActiveView(v)}
              style={{ padding: '5px 16px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', background: activeView === v ? 'rgba(232,96,28,0.12)' : 'none', color: activeView === v ? ACCENT : TEXT2, border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif', transition: 'background 0.15s, color 0.15s' }}>
              {v === 'cscan' ? 'Maps' : '3D'}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, justifyContent: 'flex-end', position: 'relative' }}>
        {mouseCoords && activeView === 'cscan' && outputTab === 'gps' && (
          <span style={{ fontSize: 11, color: TEXT2, fontVariantNumeric: 'tabular-nums', minWidth: 130 }}>
            X {mouseCoords.x.toLocaleString()} ft &nbsp;|&nbsp; Y {mouseCoords.y.toLocaleString()} ft
          </span>
        )}
        {setupDone && (
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowSettingsMenu(v => !v)} title="Settings"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 32, height: 32, background: showSettingsMenu ? RAISED : 'none', border: `1px solid ${showSettingsMenu ? BORDER2 : 'transparent'}`, color: TEXT2, cursor: 'pointer', borderRadius: 6 }}
              onMouseEnter={e => { e.currentTarget.style.background = RAISED; e.currentTarget.style.borderColor = BORDER2; }}
              onMouseLeave={e => { if (!showSettingsMenu) { e.currentTarget.style.background = 'none'; e.currentTarget.style.borderColor = 'transparent'; } }}
            ><Settings size={15} /></button>
            {showSettingsMenu && (
              <div style={{ position: 'absolute', top: '100%', right: 0, marginTop: 4, background: RAISED, border: `1px solid ${BORDER2}`, zIndex: 100, minWidth: 180, boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}>
                {[
                  { label: 'Edit Equipment',  action: () => { setSetupDone(false); setSetupStep(1); setShowSettingsMenu(() => false); } },
                  { label: 'Re-run Analysis', action: () => { fileInputRef.current?.click(); setRightIconOpen('analysis'); setShowSettingsMenu(() => false); } },
                  { label: 'New Project',     action: () => { newProject(); setShowSettingsMenu(() => false); } },
                ].map(({ label, action }) => (
                  <button key={label} onClick={action}
                    style={{ display: 'block', width: '100%', textAlign: 'left', padding: '10px 16px', background: 'none', border: 'none', color: TEXT, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.04)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                  >{label}</button>
                ))}
              </div>
            )}
          </div>
        )}
        <div style={{ position: 'relative' }}>
          <button onClick={() => setShowExportMenu(v => !v)}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', background: ACCENT, border: 'none', color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif', borderRadius: 4 }}>
            <Download size={12} /> Export
          </button>
          {showExportMenu && (
            <div style={{ position: 'absolute', top: '100%', right: 0, marginTop: 4, background: RAISED, border: `1px solid ${BORDER2}`, zIndex: 100, minWidth: 180, boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}>
              {[
                { label: 'Export Current Map', action: exportPNG },
                { label: 'Export PDF Report',  action: () => setShowExportMenu(() => false) },
                { label: 'Export CSV Data',    action: () => setShowExportMenu(() => false) },
              ].map(({ label, action }) => (
                <button key={label} onClick={action}
                  style={{ display: 'block', width: '100%', textAlign: 'left', padding: '10px 16px', background: 'none', border: 'none', color: TEXT, fontSize: 12, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.04)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                >{label}</button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
