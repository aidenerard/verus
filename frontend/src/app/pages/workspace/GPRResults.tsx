import { lazy, Suspense, useMemo, useState } from 'react';
import { ImageOff, LayoutGrid, MousePointer2 } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { SERVER } from '../inspect/constants';
import { useMapbox } from '../inspect/useMapbox';
import { BORDER, BORDER2, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT, ACCENT_SOFT } from './tokens';

const InteractiveView = lazy(() => import('../interactive/InteractiveView'));

interface Props {
  result:     AnalysisResult;
  projectId?: string;
}

type ResultsTab = 'overview' | 'interactive';

interface PanelSpec { title: string; src: string | undefined }

function resolveUrl(src: string | undefined): string | undefined {
  if (!src) return undefined;
  if (src.startsWith('http') || src.startsWith('data:')) return src;
  return `${SERVER}${src.startsWith('/') ? '' : '/'}${src}`;
}

function meanRebarDepth(result: AnalysisResult): number | undefined {
  const samples = result.per_file_summary
    .map(f => f.rebar_depth_mean)
    .filter((n): n is number => typeof n === 'number');
  if (!samples.length) return undefined;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

export default function GPRResults({ result, projectId }: Props) {
  const [tab, setTab] = useState<ResultsTab>('overview');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <ResultsTabs tab={tab} setTab={setTab} interactiveDisabled={!projectId} />
      {tab === 'overview'
        ? <OverviewTab result={result} />
        : <InteractiveTabBody projectId={projectId} />}
    </div>
  );
}

function ResultsTabs({ tab, setTab, interactiveDisabled }:
  { tab: ResultsTab; setTab: (t: ResultsTab) => void; interactiveDisabled: boolean }) {
  const items: { id: ResultsTab; label: string; Icon: typeof LayoutGrid; disabled?: boolean }[] = [
    { id: 'overview',    label: 'Overview',    Icon: LayoutGrid },
    { id: 'interactive', label: 'Interactive', Icon: MousePointer2, disabled: interactiveDisabled },
  ];
  return (
    <div role="tablist" style={{ display: 'flex', gap: 0, background: PANEL, border: `1px solid ${BORDER}`, alignSelf: 'flex-start' }}>
      {items.map(({ id, label, Icon, disabled }) => {
        const active = tab === id;
        const dim = disabled && !active;
        return (
          <button
            key={id} role="tab" aria-selected={active}
            onClick={() => { if (!disabled) setTab(id); }}
            disabled={disabled}
            title={disabled ? 'Save the project first to enable the interactive view' : undefined}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '10px 16px', border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
              background: active ? ACCENT_SOFT : 'transparent',
              color: active ? ACCENT : (dim ? TEXT3 : TEXT2),
              borderRight: id === 'overview' ? `1px solid ${BORDER}` : 'none',
              fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
              fontFamily: 'inherit',
            }}
          >
            <Icon size={13} /> {label}
          </button>
        );
      })}
    </div>
  );
}

function OverviewTab({ result }: { result: AnalysisResult }) {
  const panels: PanelSpec[] = useMemo(() => [
    { title: 'Horizon Picks',   src: resolveUrl(result.horizon_picks   ?? result.cscan_url ?? result.cscan_image) },
    { title: 'Rebar Depth Map', src: resolveUrl(result.rebar_depth_map ?? result.rebar_depth_image) },
    { title: 'Corrosion Risk',  src: resolveUrl(result.corrosion_map   ?? result.amplitude_image) },
  ], [result]);

  const stats = useMemo(() => {
    const mean  = result.mean_depth_inches     ?? meanRebarDepth(result);
    const thick = result.deck_thickness_inches;
    const risk  = result.high_risk_pct         ?? result.delamination_pct;
    return { mean, thick, risk };
  }, [result]);

  const { mapContainerRef } = useMapbox({
    analysisResult: result,
    layerVis:       { gpr: true, condition: true, amplitude: false, satellite: true, annotations: true },
    conditionOpacity: 80,
  });

  const hasGps = result.per_file_summary.some(f => f.gps);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="gpr-panel-grid">
        {panels.map(p => <ResultPanel key={p.title} title={p.title} src={p.src} />)}
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '14px 20px', display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'center' }}>
        <Stat label="Mean Depth"      value={stats.mean !== undefined ? `${stats.mean.toFixed(2)}"` : '—'} />
        <Stat label="Deck Thickness"  value={stats.thick !== undefined ? `${stats.thick.toFixed(2)}"` : '—'} />
        <Stat label="Risk"            value={stats.risk !== undefined ? `${stats.risk.toFixed(0)}%` : '—'} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
        <div style={{ padding: '12px 20px', borderBottom: `1px solid ${BORDER}`, fontSize: 11, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT2 }}>
          Scan Location
        </div>
        <div style={{ position: 'relative', height: 320 }}>
          <div ref={mapContainerRef} style={{ position: 'absolute', inset: 0 }} />
          {!hasGps && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: RAISED, color: TEXT3, fontSize: 12, pointerEvents: 'none' }}>
              No GPS data in scan
            </div>
          )}
        </div>
      </div>

      <style>{`
        .gpr-panel-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
        }
        @media (max-width: 900px) {
          .gpr-panel-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}

function InteractiveTabBody({ projectId }: { projectId: string | undefined }) {
  if (!projectId) {
    return (
      <div style={{ background: PANEL, border: `1px dashed ${BORDER2}`, padding: '40px 24px', textAlign: 'center', color: TEXT2 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: TEXT, marginBottom: 6 }}>Save the project first</div>
        <div style={{ fontSize: 12 }}>The interactive view opens once this analysis is saved and has a project id.</div>
      </div>
    );
  }
  return (
    <Suspense fallback={
      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '40px 24px', textAlign: 'center', color: TEXT2, fontSize: 13 }}>
        Loading interactive view…
      </div>
    }>
      <InteractiveView projectId={projectId} />
    </Suspense>
  );
}

function ResultPanel({ title, src }: PanelSpec) {
  return (
    <div style={{ background: PANEL, border: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 14px', borderBottom: `1px solid ${BORDER}`, fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT2 }}>
        {title}
      </div>
      <div style={{ flex: 1, aspectRatio: '4 / 3', background: RAISED, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
        {src ? (
          <img src={src} alt={title} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: TEXT3 }}>
            <ImageOff size={22} />
            <span style={{ fontSize: 11 }}>No data</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3 }}>{label}</span>
      <span style={{
        fontSize: 16, fontWeight: 700, color: accent ? ACCENT : TEXT,
        padding: accent ? '0 6px' : 0, background: accent ? ACCENT_SOFT : 'transparent',
      }}>{value}</span>
    </div>
  );
}
