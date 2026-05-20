import { useMemo, useRef } from 'react';
import type { Scene } from '../state/types';
import { useScanLine } from '../state/hooks';
import { useInteractiveStore } from '../state/useInteractiveStore';
import BScanCanvas from './BScanCanvas';
import PickDots from './PickDots';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from '../tokens';

interface Props {
  projectId: string;
  scene:     Scene;
}

const PX_PER_TRACE  = 2;
const PX_PER_SAMPLE = 1.5;
const AXIS_W = 56;
const HEADER_H = 36;

export default function BScanPanel({ projectId, scene }: Props) {
  const selectedScanLineId = useInteractiveStore(s => {
    const id = s.selectedPickIds[0];
    if (id) {
      const p = s.picks.get(id);
      if (p) return p.scan_line_id;
    }
    return scene.scan_lines[0]?.id;
  });

  const { traces, isLoading, error } = useScanLine(projectId, selectedScanLineId);
  const allPicks = useInteractiveStore(s => Array.from(s.picks.values()));

  const tracePicks = useMemo(() => {
    if (!traces) return [];
    return allPicks.filter(p => p.scan_line_id === traces.id);
  }, [allPicks, traces]);

  const scanLineLabel =
    scene.scan_lines.find(s => s.id === selectedScanLineId)?.label ?? selectedScanLineId ?? '—';

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ height: HEADER_H, flexShrink: 0, padding: '0 14px', background: PANEL, borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3 }}>
          B-Scan
        </div>
        <div style={{ fontSize: 12, color: TEXT, fontWeight: 700 }}>{scanLineLabel}</div>
        {traces && (
          <div style={{ fontSize: 11, color: TEXT2, fontVariantNumeric: 'tabular-nums' }}>
            {traces.n_traces} traces · {traces.n_samples} samples · {traces.samples_per_ns.toFixed(1)} samples/ns
          </div>
        )}
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: `${AXIS_W}px 1fr`, background: RAISED }}>
        {traces
          ? <YAxis nSamples={traces.n_samples} samplesPerNs={traces.samples_per_ns} epsilonR={scene.epsilon_r} pxPerSample={PX_PER_SAMPLE} />
          : <div />}

        <div style={{ overflowX: 'auto', overflowY: 'auto', position: 'relative' }}>
          {isLoading && <StatusOverlay text="Loading trace data…" />}
          {error    && <StatusOverlay text={`Failed to load: ${error.message}`} accent />}
          {traces && (
            <BScanSurface
              key={traces.id}
              tracesId={traces.id}
              nTraces={traces.n_traces}
              nSamples={traces.n_samples}
            >
              <BScanCanvas traces={traces} pxPerTrace={PX_PER_TRACE} pxPerSample={PX_PER_SAMPLE} />
              <PickDots traces={traces} picks={tracePicks} pxPerTrace={PX_PER_TRACE} pxPerSample={PX_PER_SAMPLE} />
            </BScanSurface>
          )}
        </div>
      </div>
    </div>
  );
}

function BScanSurface({ children, nTraces, nSamples }: { children: React.ReactNode; tracesId: string; nTraces: number; nSamples: number }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <div ref={ref} style={{
      position: 'relative',
      width: nTraces * PX_PER_TRACE,
      height: nSamples * PX_PER_SAMPLE,
    }}>
      {children}
    </div>
  );
}

function YAxis({ nSamples, samplesPerNs, epsilonR, pxPerSample }: { nSamples: number; samplesPerNs: number; epsilonR: number; pxPerSample: number }) {
  const totalNs = nSamples / samplesPerNs;
  const c_ft_per_ns = 0.984252;
  const v = c_ft_per_ns / Math.sqrt(epsilonR);
  const ticks = 6;

  return (
    <div style={{ position: 'relative', background: PANEL, borderRight: `1px solid ${BORDER}`, height: nSamples * pxPerSample, overflow: 'hidden' }}>
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const t = i / ticks;
        const ns = totalNs * t;
        const depthIn = (ns * v) / 2 * 12;
        return (
          <div key={i} style={{
            position: 'absolute', left: 0, right: 0, top: `${t * 100}%`,
            transform: 'translateY(-50%)',
            padding: '0 6px', display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
            color: TEXT2, fontSize: 9, fontFamily: 'ui-monospace, monospace', lineHeight: 1.1,
          }}>
            <span style={{ color: TEXT }}>{ns.toFixed(1)} ns</span>
            <span style={{ color: TEXT3 }}>{depthIn.toFixed(1)} in</span>
          </div>
        );
      })}
    </div>
  );
}

function StatusOverlay({ text, accent }: { text: string; accent?: boolean }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: accent ? '#E8601C' : TEXT2, fontSize: 12, background: PANEL }}>
      {text}
    </div>
  );
}
