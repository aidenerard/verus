/**
 * CentralColumn.tsx — vertical split panel: main viewport + icon strip + right panel + B-scan.
 * Contains the PanelGroup structure, tab bar, OutputMaps, and all slide-out panels.
 * Does NOT own any state — receives all values as props from GPRWorkspace.
 */

import type { RefObject } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { ImperativePanelHandle } from 'react-resizable-panels';
import { Layers, BarChart2, SlidersHorizontal } from 'lucide-react';
import ThreeDView from '../../components/ThreeDView';
import { PANEL, RAISED, BORDER, TEXT, TEXT2, ACCENT, MAPBOX_TOKEN } from './constants';
import type { LayerId } from './constants';
import type { AnalysisResult, OutputTab, UploadedFile } from './types';
import OutputMaps from './OutputMaps';
import RightSlidePanel from './RightSlidePanel';
import BScanPanel from './BScanPanel';

type JobStatus = 'idle'|'pending'|'processing'|'complete'|'failed';
type RightPanel = 'properties'|'analysis'|'adjust'|null;
interface Badge { text: string; color: string }

interface CentralColumnProps {
  mapContainerRef:    RefObject<HTMLDivElement | null>;
  activeView:         'cscan' | '3d';
  outputTab:          OutputTab;
  setOutputTab:       (t: OutputTab) => void;
  hasResult:          boolean;
  analysisResult:     AnalysisResult | null;
  condCanvasRef:      RefObject<HTMLCanvasElement | null>;
  rebarCanvasRef:     RefObject<HTMLCanvasElement | null>;
  rebarColorbarRef:   RefObject<HTMLCanvasElement | null>;
  ampCanvasRef:       RefObject<HTMLCanvasElement | null>;
  exportPNG:          () => void;
  condBadge:          () => Badge | null;
  depthBadge:         () => Badge | null;
  ampBadge:           () => Badge | null;
  rightIconOpen:      RightPanel;
  setRightIconOpen:   React.Dispatch<React.SetStateAction<RightPanel>>;
  selectedLayer:      LayerId;
  selectedFileIdx:    number;
  setSelectedFileIdx: (i: number) => void;
  bottomPanelRef:     RefObject<ImperativePanelHandle | null>;
  conditionOpacity:   number;
  setConditionOpacity:(v: number) => void;
  manufacturer:       string;
  frequencyMhz:       number;
  bridgeId:           string;
  inspDate:           string;
  setupDone:          boolean;
  isAnalyzing:        boolean;
  jobStatus:          JobStatus;
  statusMsg:          string;
  errorMsg:           string | null;
  files:              UploadedFile[];
  fileInputRef:       RefObject<HTMLInputElement | null>;
  detectionThreshold: number;
  setDetectionThreshold:(v: number) => void;
  setUseCondCanvas:   (v: boolean) => void;
  dielectricEr:       number;
  setDielectricEr:    (v: number) => void;
  setUseRebarCanvas:  (v: boolean) => void;
  ampClampMin:        number;
  setAmpClampMin:     (v: number) => void;
  ampClampMax:        number;
  setAmpClampMax:     (v: number) => void;
  setUseAmpCanvas:    (v: boolean) => void;
  projectId:          string | null;
  setShowConfirm:     (v: boolean) => void;
  totalFiles:         number;
  bscanCanvasRef:     RefObject<HTMLCanvasElement | null>;
  bottomExpanded:     boolean;
  setBottomExpanded:  React.Dispatch<React.SetStateAction<boolean>>;
  outputTabs: { id: OutputTab; label: string; badge: () => Badge | null }[];
}

export default function CentralColumn(p: CentralColumnProps) {
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <PanelGroup direction="vertical" style={{ flex: 1 }}>
        <Panel defaultSize={75} minSize={35} style={{ display: 'flex', overflow: 'hidden', position: 'relative' }}>

          <div style={{ flex: 1, position: 'relative', overflow: 'hidden', background: RAISED }}>
            <div ref={p.mapContainerRef} style={{ width: '100%', height: '100%', position: 'absolute', inset: 0, opacity: (p.activeView === 'cscan' && p.outputTab === 'gps') ? 1 : 0, pointerEvents: (p.activeView === 'cscan' && p.outputTab === 'gps') ? 'auto' : 'none', zIndex: (p.activeView === 'cscan' && p.outputTab === 'gps') ? 1 : 0, transition: 'opacity 0.2s' }} />
            {!MAPBOX_TOKEN && p.activeView === 'cscan' && p.outputTab === 'gps' && (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: PANEL, zIndex: 2 }}>
                <p style={{ fontSize: 13, color: TEXT2 }}>Set <code style={{ background: RAISED, padding: '2px 6px', fontSize: 11 }}>VITE_MAPBOX_TOKEN</code> in .env.local.</p>
              </div>
            )}
            {p.activeView === '3d' && (
              <div style={{ position: 'absolute', inset: 0, zIndex: 5 }}>
                {p.hasResult ? <ThreeDView perFileSummary={p.analysisResult!.per_file_summary} />
                  : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><p style={{ color: TEXT2, fontSize: 13 }}>Run an analysis to see the 3D view.</p></div>}
              </div>
            )}
            {p.activeView === 'cscan' && (
              <>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10, height: 40, display: 'flex', alignItems: 'stretch', background: PANEL, borderBottom: `1px solid ${BORDER}` }}>
                  {p.outputTabs.map(tab => {
                    const disabled = tab.id !== 'gps' && !p.hasResult;
                    return (
                      <button key={tab.id} onClick={() => !disabled && p.setOutputTab(tab.id)}
                        style={{ padding: '0 18px', fontSize: 13, fontWeight: 600, background: 'none', border: 'none', cursor: disabled ? 'default' : 'pointer', color: p.outputTab === tab.id ? TEXT : (disabled ? '#4a5568' : TEXT2), borderBottom: `2px solid ${p.outputTab === tab.id ? ACCENT : 'transparent'}`, fontFamily: 'Inter, sans-serif', transition: 'color 0.12s', whiteSpace: 'nowrap', letterSpacing: '-0.01em' }}>
                        {tab.label}
                      </button>
                    );
                  })}
                </div>
                {p.outputTab !== 'gps' && (
                  <div style={{ position: 'absolute', inset: 0, top: 40, overflow: 'hidden', background: RAISED, zIndex: 3 }}>
                    <OutputMaps
                      outputTab={p.outputTab} hasResult={p.hasResult} analysisResult={p.analysisResult}
                      condCanvasRef={p.condCanvasRef} rebarCanvasRef={p.rebarCanvasRef}
                      rebarColorbarRef={p.rebarColorbarRef} ampCanvasRef={p.ampCanvasRef}
                      onExport={p.exportPNG} condBadge={p.condBadge} depthBadge={p.depthBadge} ampBadge={p.ampBadge}
                    />
                  </div>
                )}
              </>
            )}
          </div>

          <div style={{ width: 44, flexShrink: 0, background: PANEL, borderLeft: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 10, gap: 4 }}>
            {([
              { id: 'properties' as const, Icon: Layers,           title: 'Properties' },
              { id: 'analysis'   as const, Icon: BarChart2,         title: 'Analysis'   },
              { id: 'adjust'     as const, Icon: SlidersHorizontal, title: 'Adjust'     },
            ]).map(({ id, Icon, title }) => (
              <button key={id} title={title}
                onClick={() => p.setRightIconOpen(prev => prev === id ? null : id)}
                style={{ width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', background: p.rightIconOpen === id ? 'rgba(232,96,28,0.12)' : 'none', border: 'none', borderRadius: 6, cursor: 'pointer', color: p.rightIconOpen === id ? ACCENT : TEXT2, transition: 'background 0.12s, color 0.12s' }}
                onMouseEnter={e => { if (p.rightIconOpen !== id) { e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; e.currentTarget.style.color = TEXT; } }}
                onMouseLeave={e => { if (p.rightIconOpen !== id) { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = TEXT2; } }}
              ><Icon size={16} /></button>
            ))}
          </div>

          <RightSlidePanel
            rightIconOpen={p.rightIconOpen} selectedLayer={p.selectedLayer}
            hasResult={p.hasResult} analysisResult={p.analysisResult}
            selectedFileIdx={p.selectedFileIdx} setSelectedFileIdx={p.setSelectedFileIdx}
            bottomPanelRef={p.bottomPanelRef as any}
            conditionOpacity={p.conditionOpacity} setConditionOpacity={p.setConditionOpacity}
            manufacturer={p.manufacturer} frequencyMhz={p.frequencyMhz}
            bridgeId={p.bridgeId} inspDate={p.inspDate} setupDone={p.setupDone}
            isAnalyzing={p.isAnalyzing} jobStatus={p.jobStatus}
            statusMsg={p.statusMsg} errorMsg={p.errorMsg}
            files={p.files} fileInputRef={p.fileInputRef}
            detectionThreshold={p.detectionThreshold} setDetectionThreshold={p.setDetectionThreshold}
            setUseCondCanvas={p.setUseCondCanvas}
            dielectricEr={p.dielectricEr} setDielectricEr={p.setDielectricEr}
            setUseRebarCanvas={p.setUseRebarCanvas}
            ampClampMin={p.ampClampMin} setAmpClampMin={p.setAmpClampMin}
            ampClampMax={p.ampClampMax} setAmpClampMax={p.setAmpClampMax}
            setUseAmpCanvas={p.setUseAmpCanvas}
            projectId={p.projectId} outputTab={p.outputTab} setShowConfirm={p.setShowConfirm}
          />

        </Panel>

        <PanelResizeHandle style={{ height: 3, background: BORDER, cursor: 'row-resize' }} />

        <BScanPanel
          bottomPanelRef={p.bottomPanelRef}
          hasResult={p.hasResult} totalFiles={p.totalFiles}
          selectedFileIdx={p.selectedFileIdx} setSelectedFileIdx={p.setSelectedFileIdx}
          bscanCanvasRef={p.bscanCanvasRef}
          bottomExpanded={p.bottomExpanded} setBottomExpanded={p.setBottomExpanded}
        />
      </PanelGroup>
    </div>
  );
}
