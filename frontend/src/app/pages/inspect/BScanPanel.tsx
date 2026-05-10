/**
 * BScanPanel.tsx
 * The collapsible bottom B-scan panel: header with swath navigation and
 * expand/collapse controls, plus the canvas display area.
 *
 * Does NOT: perform canvas rendering (that is handled by useCanvasRenderers).
 */

import { ChevronDown, ChevronLeft, ChevronRight, Maximize2, Minimize2 } from 'lucide-react';
import { Panel } from 'react-resizable-panels';
import type { ImperativePanelHandle } from 'react-resizable-panels';

import { PANEL, RAISED, BORDER, TEXT, TEXT2, ACCENT } from './constants';

interface BScanPanelProps {
  bottomPanelRef: React.RefObject<ImperativePanelHandle>;
  hasResult: boolean;
  totalFiles: number;
  selectedFileIdx: number;
  setSelectedFileIdx: (fn: (i: number) => number) => void;
  bscanCanvasRef: React.RefObject<HTMLCanvasElement>;
  bottomExpanded: boolean;
  setBottomExpanded: (fn: (v: boolean) => boolean) => void;
}

export default function BScanPanel({
  bottomPanelRef, hasResult, totalFiles, selectedFileIdx, setSelectedFileIdx,
  bscanCanvasRef, bottomExpanded, setBottomExpanded,
}: BScanPanelProps) {
  return (
    <Panel ref={bottomPanelRef} defaultSize={25} minSize={0} collapsible collapsedSize={0}
      style={{ background: PANEL, borderTop: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ height: 36, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10, padding: '0 14px', borderBottom: `1px solid ${BORDER}`, background: RAISED }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2, flexShrink: 0 }}>B-Scan</span>
        {hasResult && totalFiles > 0 && (
          <>
            <div style={{ width: 1, height: 16, background: BORDER, flexShrink: 0 }} />
            <button onClick={() => setSelectedFileIdx(i => Math.max(0, i - 1))}
              style={{ background: 'none', border: `1px solid ${BORDER}`, cursor: 'pointer', color: TEXT2, padding: '2px 8px', borderRadius: 20, display: 'flex', alignItems: 'center' }}
              onMouseEnter={e => (e.currentTarget.style.color = TEXT)} onMouseLeave={e => (e.currentTarget.style.color = TEXT2)}
            ><ChevronLeft size={14} /></button>
            <span style={{ fontSize: 11, color: TEXT, minWidth: 80, textAlign: 'center', fontWeight: 600 }}>Swath {selectedFileIdx + 1} / {totalFiles}</span>
            <button onClick={() => setSelectedFileIdx(i => Math.min(totalFiles - 1, i + 1))}
              style={{ background: 'none', border: `1px solid ${BORDER}`, cursor: 'pointer', color: TEXT2, padding: '2px 8px', borderRadius: 20, display: 'flex', alignItems: 'center' }}
              onMouseEnter={e => (e.currentTarget.style.color = TEXT)} onMouseLeave={e => (e.currentTarget.style.color = TEXT2)}
            ><ChevronRight size={14} /></button>
            <div style={{ width: 1, height: 16, background: BORDER }} />
            {(['In-Line', 'Cross'] as const).map(t => (
              <button key={t}
                style={{ padding: '3px 12px', fontSize: 10, fontWeight: 700, letterSpacing: '0.05em', background: t === 'In-Line' ? ACCENT : 'none', border: `1px solid ${t === 'In-Line' ? ACCENT : BORDER}`, color: t === 'In-Line' ? '#fff' : TEXT2, cursor: 'pointer', fontFamily: 'Inter, sans-serif', borderRadius: 20 }}>{t}</button>
            ))}
          </>
        )}
        <div style={{ flex: 1 }} />
        <button onClick={() => setBottomExpanded(v => !v)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}>
          {bottomExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
        </button>
        <button onClick={() => bottomPanelRef.current?.collapse()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}><ChevronDown size={13} /></button>
      </div>
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', background: RAISED, position: 'relative' }}>
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
  );
}
