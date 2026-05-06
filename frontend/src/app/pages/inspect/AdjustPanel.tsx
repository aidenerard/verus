import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { supabase } from '../../../lib/supabase';
import { BORDER, TEXT, TEXT2, ACCENT } from './constants';
import type { AnalysisResult, OutputTab } from './types';

interface Props {
  hasResult: boolean;
  outputTab: OutputTab;
  analysisResult: AnalysisResult | null;
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
}

export default function AdjustPanel({
  hasResult, outputTab, analysisResult,
  detectionThreshold, setDetectionThreshold, setUseCondCanvas,
  dielectricEr, setDielectricEr, setUseRebarCanvas,
  ampClampMin, setAmpClampMin, ampClampMax, setAmpClampMax, setUseAmpCanvas,
  projectId,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!hasResult) return null;

  return (
    <div style={{ borderTop: `1px solid ${BORDER}`, flexShrink: 0 }}>
      <button onClick={() => setExpanded(v => !v)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 14px', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT2 }}>Adjust</span>
        {expanded ? <ChevronUp size={12} style={{ color: TEXT2 }} /> : <ChevronDown size={12} style={{ color: TEXT2 }} />}
      </button>

      {expanded && (
        <div style={{ padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 14 }}>

          {outputTab === 'condition' && analysisResult?.prob_grid && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: TEXT2, marginBottom: 6 }}>
                <span style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Detection Threshold</span>
                <span style={{ color: TEXT, fontWeight: 700 }}>{detectionThreshold.toFixed(2)}</span>
              </div>
              <input type="range" min={0.3} max={0.9} step={0.01} value={detectionThreshold}
                onChange={e => { setDetectionThreshold(+e.target.value); setUseCondCanvas(true); }}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: TEXT2, marginTop: 2 }}>
                <span>0.3 (sensitive)</span><span>0.9 (strict)</span>
              </div>
              <button onClick={async () => {
                if (!projectId) return;
                await supabase.from('projects').update({ project_settings: { detection_threshold: detectionThreshold } }).eq('id', projectId);
              }} style={{ marginTop: 8, width: '100%', padding: '6px', background: 'rgba(232,96,28,0.12)', border: `1px solid rgba(232,96,28,0.3)`, color: ACCENT, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                Apply to Report
              </button>
            </div>
          )}

          {outputTab === 'rebar_depth' && analysisResult?.twt_grid && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: TEXT2, marginBottom: 6 }}>
                <span style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Dielectric Constant εr</span>
                <span style={{ color: TEXT, fontWeight: 700 }}>{dielectricEr.toFixed(1)}</span>
              </div>
              <input type="range" min={4} max={12} step={0.5} value={dielectricEr}
                onChange={e => { setDielectricEr(+e.target.value); setUseRebarCanvas(true); }}
                style={{ width: '100%', accentColor: ACCENT }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: TEXT2, marginTop: 2 }}>
                <span>4 (dry/porous)</span><span>12 (wet/dense)</span>
              </div>
              <div style={{ marginTop: 8, fontSize: 10, color: TEXT2, lineHeight: 1.5 }}>
                v = {(0.3 / Math.sqrt(dielectricEr)).toFixed(3)} m/ns
              </div>
            </div>
          )}

          {outputTab === 'amplitude' && (
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: TEXT2, marginBottom: 8 }}>Color Scale</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: TEXT2, marginBottom: 4 }}>
                <span>Min: {ampClampMin.toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={0.5} step={0.01} value={ampClampMin}
                onChange={e => { setAmpClampMin(+e.target.value); setUseAmpCanvas(true); }}
                style={{ width: '100%', accentColor: ACCENT, marginBottom: 8 }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: TEXT2, marginBottom: 4 }}>
                <span>Max: {ampClampMax.toFixed(2)}</span>
              </div>
              <input type="range" min={0.5} max={1} step={0.01} value={ampClampMax}
                onChange={e => { setAmpClampMax(+e.target.value); setUseAmpCanvas(true); }}
                style={{ width: '100%', accentColor: ACCENT }} />
            </div>
          )}

          {outputTab === 'gps' && (
            <div style={{ fontSize: 11, color: TEXT2 }}>GPS overlay adjustments available in the Layers panel.</div>
          )}

        </div>
      )}
    </div>
  );
}
