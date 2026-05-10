/**
 * RightSlidePanel.tsx
 * The right slide-out panel (280px) with three tabs: Properties, Analysis, Adjust.
 * Properties shows per-file scan lines and analysis/setup summaries.
 * Analysis shows model info, file upload prompt, and job status.
 * Adjust renders the AdjustPanel component.
 *
 * Does NOT: manage job state transitions, canvas rendering, or map logic.
 */

import { Loader2, Check, AlertCircle } from 'lucide-react';

import { PANEL, RAISED, BORDER, TEXT, TEXT2, ACCENT, MANUFACTURERS } from './constants';
import type { ManufacturerKey } from './constants';
import type { AnalysisResult, OutputTab } from './types';
import { delamColor } from './utils';
import AdjustPanel from './AdjustPanel';

interface RightSlidePanelProps {
  rightIconOpen: 'properties' | 'analysis' | 'adjust' | null;
  selectedLayer: string;
  hasResult: boolean;
  analysisResult: AnalysisResult | null;
  selectedFileIdx: number;
  setSelectedFileIdx: (i: number) => void;
  bottomPanelRef: React.RefObject<{ expand: () => void }>;
  conditionOpacity: number;
  setConditionOpacity: (v: number) => void;
  manufacturer: ManufacturerKey | '';
  frequencyMhz: number;
  bridgeId: string;
  inspDate: string;
  setupDone: boolean;
  isAnalyzing: boolean;
  jobStatus: 'idle' | 'pending' | 'processing' | 'complete' | 'failed';
  statusMsg: string;
  errorMsg: string | null;
  files: { file: File; name: string }[];
  fileInputRef: React.RefObject<HTMLInputElement>;
  detectionThreshold: number;
  setDetectionThreshold: (v: number) => void;
  setUseCondCanvas: (v: boolean) => void;
  dielectricEr: number;
  setDielectricEr: (v: number) => void;
  setUseRebarCanvas: (v: boolean) => void;
  ampClampMin: number;
  setAmpClampMin: (v: number) => void;
  ampClampMax: number;
  setAmpClampMax: (v: number) => void;
  setUseAmpCanvas: (v: boolean) => void;
  projectId: string | null;
  outputTab: OutputTab;
  setShowConfirm: (v: boolean) => void;
}

export default function RightSlidePanel({
  rightIconOpen, selectedLayer, hasResult, analysisResult,
  selectedFileIdx, setSelectedFileIdx, bottomPanelRef,
  conditionOpacity, setConditionOpacity,
  manufacturer, frequencyMhz, bridgeId, inspDate, setupDone,
  isAnalyzing, jobStatus, statusMsg, errorMsg,
  files, fileInputRef,
  detectionThreshold, setDetectionThreshold, setUseCondCanvas,
  dielectricEr, setDielectricEr, setUseRebarCanvas,
  ampClampMin, setAmpClampMin, ampClampMax, setAmpClampMax, setUseAmpCanvas,
  projectId, outputTab, setShowConfirm,
}: RightSlidePanelProps) {
  return (
    <div style={{ width: rightIconOpen ? 280 : 0, transition: 'width 0.18s ease', flexShrink: 0, overflow: 'hidden', background: PANEL, borderLeft: rightIconOpen ? `1px solid ${BORDER}` : 'none', display: 'flex', flexDirection: 'column' }}>
      <div style={{ width: 280, height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10px 14px 8px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2 }}>
            {rightIconOpen === 'properties' ? 'Properties' : rightIconOpen === 'analysis' ? 'Analysis' : 'Adjust'}
          </span>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
          {rightIconOpen === 'properties' && (
            <PropertiesTab
              selectedLayer={selectedLayer} hasResult={hasResult} analysisResult={analysisResult}
              selectedFileIdx={selectedFileIdx} setSelectedFileIdx={setSelectedFileIdx}
              bottomPanelRef={bottomPanelRef}
              conditionOpacity={conditionOpacity} setConditionOpacity={setConditionOpacity}
              manufacturer={manufacturer} frequencyMhz={frequencyMhz}
              bridgeId={bridgeId} inspDate={inspDate} setupDone={setupDone}
            />
          )}
          {rightIconOpen === 'analysis' && (
            <AnalysisTab
              manufacturer={manufacturer} frequencyMhz={frequencyMhz}
              detectionThreshold={detectionThreshold}
              isAnalyzing={isAnalyzing} jobStatus={jobStatus}
              statusMsg={statusMsg} errorMsg={errorMsg}
              files={files} fileInputRef={fileInputRef}
              hasResult={hasResult} analysisResult={analysisResult}
              setShowConfirm={setShowConfirm}
            />
          )}
          {rightIconOpen === 'adjust' && (
            <AdjustPanel
              hasResult={hasResult} outputTab={outputTab} analysisResult={analysisResult}
              detectionThreshold={detectionThreshold} setDetectionThreshold={setDetectionThreshold} setUseCondCanvas={setUseCondCanvas}
              dielectricEr={dielectricEr} setDielectricEr={setDielectricEr} setUseRebarCanvas={setUseRebarCanvas}
              ampClampMin={ampClampMin} setAmpClampMin={setAmpClampMin}
              ampClampMax={ampClampMax} setAmpClampMax={setAmpClampMax} setUseAmpCanvas={setUseAmpCanvas}
              projectId={projectId}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Internal sub-sections ─────────────────────────────────────────────────────

interface PropertiesTabProps {
  selectedLayer: string;
  hasResult: boolean;
  analysisResult: AnalysisResult | null;
  selectedFileIdx: number;
  setSelectedFileIdx: (i: number) => void;
  bottomPanelRef: React.RefObject<{ expand: () => void }>;
  conditionOpacity: number;
  setConditionOpacity: (v: number) => void;
  manufacturer: ManufacturerKey | '';
  frequencyMhz: number;
  bridgeId: string;
  inspDate: string;
  setupDone: boolean;
}

function PropertiesTab({
  selectedLayer, hasResult, analysisResult, selectedFileIdx, setSelectedFileIdx,
  bottomPanelRef, conditionOpacity, setConditionOpacity,
  manufacturer, frequencyMhz, bridgeId, inspDate, setupDone,
}: PropertiesTabProps) {
  return (
    <>
      {selectedLayer === 'gpr' && (
        <div>
          <div style={{ padding: '4px 14px 8px', fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2 }}>Scan Lines</div>
          {hasResult ? analysisResult!.per_file_summary.map((f, i) => (
            <div key={f.filename} onClick={() => { setSelectedFileIdx(i); bottomPanelRef.current?.expand(); }}
              style={{ padding: '8px 14px', cursor: 'pointer', background: selectedFileIdx === i ? 'rgba(232,96,28,0.08)' : 'none', borderLeft: `2px solid ${selectedFileIdx === i ? ACCENT : 'transparent'}` }}
              onMouseEnter={e => { if (selectedFileIdx !== i) e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; }}
              onMouseLeave={e => { if (selectedFileIdx !== i) e.currentTarget.style.background = 'none'; }}>
              <div style={{ fontSize: 11, color: TEXT, marginBottom: 4, fontFamily: 'monospace', wordBreak: 'break-all' }}>{f.filename}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 10, color: TEXT2 }}>{f.signals != null ? f.signals.toLocaleString() : '--'} signals</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 40, height: 4, background: BORDER, borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${f.delam_pct ?? 0}%`, background: delamColor(f.delam_pct ?? 0), borderRadius: 2 }} />
                  </div>
                  <span style={{ fontSize: 10, color: delamColor(f.delam_pct ?? 0), fontWeight: 700, minWidth: 32, textAlign: 'right' }}>{f.delam_pct != null ? f.delam_pct.toFixed(1) : '--'}%</span>
                </div>
              </div>
              {f.rebar_depth_mean != null && (
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 10 }}>
                  <span style={{ color: TEXT2 }}>Avg Depth</span>
                  <span style={{ color: TEXT, fontWeight: 700 }}>{f.rebar_depth_mean.toFixed(2)}"</span>
                </div>
              )}
            </div>
          )) : <div style={{ padding: '24px 14px', textAlign: 'center' }}><p style={{ fontSize: 12, color: TEXT2 }}>No files analyzed yet.</p></div>}
        </div>
      )}
      {selectedLayer === 'condition' && (
        <div style={{ padding: '4px 14px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 12 }}>Condition Map</div>
          <label style={{ display: 'block', fontSize: 10, color: TEXT2, marginBottom: 6 }}>Opacity: {conditionOpacity}%</label>
          <input type="range" min={0} max={100} value={conditionOpacity} onChange={e => setConditionOpacity(+e.target.value)} style={{ width: '100%', accentColor: ACCENT }} />
        </div>
      )}
      {selectedLayer !== 'gpr' && selectedLayer !== 'condition' && (
        <div style={{ padding: '24px 14px', textAlign: 'center' }}><p style={{ fontSize: 12, color: TEXT2 }}>No configurable properties for this layer.</p></div>
      )}
      {hasResult && (
        <div style={{ margin: '16px 14px 0', padding: '12px', background: RAISED, border: `1px solid ${BORDER}` }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 8 }}>Analysis</div>
          {[
            { label: 'Signals',      value: analysisResult!.signals_analyzed != null ? analysisResult!.signals_analyzed.toLocaleString() : '--' },
            { label: 'Delamination', value: analysisResult!.delamination_pct != null ? `${analysisResult!.delamination_pct.toFixed(1)}%` : '--', color: delamColor(analysisResult!.delamination_pct ?? 0) },
            { label: 'Sound',        value: analysisResult!.sound_pct != null ? `${analysisResult!.sound_pct.toFixed(1)}%` : '--' },
            { label: 'Time',         value: analysisResult!.analysis_time_sec != null ? `${analysisResult!.analysis_time_sec.toFixed(1)}s` : '--' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
              <span style={{ color: TEXT2 }}>{label}</span>
              <span style={{ color: (color as string | undefined) || TEXT, fontWeight: 600 }}>{value}</span>
            </div>
          ))}
        </div>
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
  );
}

interface AnalysisTabProps {
  manufacturer: ManufacturerKey | '';
  frequencyMhz: number;
  detectionThreshold: number;
  isAnalyzing: boolean;
  jobStatus: 'idle' | 'pending' | 'processing' | 'complete' | 'failed';
  statusMsg: string;
  errorMsg: string | null;
  files: { file: File; name: string }[];
  fileInputRef: React.RefObject<HTMLInputElement>;
  hasResult: boolean;
  analysisResult: AnalysisResult | null;
  setShowConfirm: (v: boolean) => void;
}

function AnalysisTab({
  manufacturer, frequencyMhz, detectionThreshold,
  isAnalyzing, jobStatus, statusMsg, errorMsg,
  files, fileInputRef, hasResult, analysisResult, setShowConfirm,
}: AnalysisTabProps) {
  return (
    <div style={{ padding: '0 14px' }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>AI Model</div>
        <div style={{ background: RAISED, padding: '10px 12px', fontSize: 11 }}>
          {[
            { label: 'Equipment', value: MANUFACTURERS.find(m => m.key === manufacturer)?.name || '—' },
            { label: 'Frequency', value: `${frequencyMhz} MHz` },
            { label: 'Standard',  value: 'ASTM D6087' },
            { label: 'Threshold', value: detectionThreshold.toFixed(2) },
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
          style={{ width: '100%', padding: '10px', marginBottom: 16, background: `rgba(232,96,28,0.08)`, border: `1px solid rgba(232,96,28,0.25)`, color: ACCENT, fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
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
          {isAnalyzing && <div style={{ height: 3, background: BORDER, overflow: 'hidden', borderRadius: 2 }}><div style={{ height: '100%', width: '40%', background: ACCENT, borderRadius: 2, animation: 'verus-bar 1.8s ease-in-out infinite' }} /></div>}
          {errorMsg && <p style={{ fontSize: 11, color: '#ef4444', marginTop: 6, lineHeight: 1.5 }}>{errorMsg}</p>}
        </div>
      )}
      {files.length > 0 && jobStatus !== 'pending' && jobStatus !== 'processing' && (
        <button onClick={() => setShowConfirm(true)}
          style={{ width: '100%', padding: '10px', marginBottom: 16, background: ACCENT, border: 'none', color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
          Run Analysis
        </button>
      )}
      {hasResult && (
        <div style={{ background: RAISED, padding: '12px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2, marginBottom: 10 }}>Summary</div>
          {[
            { label: 'Signals',      value: analysisResult!.signals_analyzed != null ? analysisResult!.signals_analyzed.toLocaleString() : '--' },
            { label: 'Delamination', value: analysisResult!.delamination_pct != null ? `${analysisResult!.delamination_pct.toFixed(1)}%` : '--', color: delamColor(analysisResult!.delamination_pct ?? 0) },
            { label: 'Sound',        value: analysisResult!.sound_pct != null ? `${analysisResult!.sound_pct.toFixed(1)}%` : '--' },
            { label: 'Time',         value: analysisResult!.analysis_time_sec != null ? `${analysisResult!.analysis_time_sec.toFixed(1)}s` : '--' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 11 }}>
              <span style={{ color: TEXT2 }}>{label}</span>
              <span style={{ color: (color as string | undefined) || TEXT, fontWeight: 600 }}>{value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
