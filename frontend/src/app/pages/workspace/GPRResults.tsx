import { useMemo, useState } from 'react';
import { ImageOff, LayoutGrid, Activity } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { useMapbox } from '../inspect/useMapbox';
import { SERVER } from '../inspect/constants';
import BScanViewer from './BScanViewer';
import ConditionMapPanel from './ConditionMapPanel';
import DepthMapCanvas from './DepthMapCanvas';
import { BORDER, BORDER2, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT, ACCENT_SOFT } from './tokens';

interface Props {
  result:     AnalysisResult;
  projectId?: string;
}

type ResultsTab = 'overview' | 'interactive';

interface PanelSpec { title: string; src: string | undefined }

function resolveImageSrc(value: string | undefined | null): string | undefined {
  if (!value) return undefined;
  if (value.startsWith('http') || value.startsWith('data:')) return value;
  // DZT/Proceq pipelines return raw base64 PNG when Supabase upload skipped —
  // wrap with the data URI prefix so <img src=...> renders it.
  return `data:image/png;base64,${value}`;
}

function meanRebarDepth(result: AnalysisResult): number | undefined {
  const samples = (result.per_file_summary ?? [])
    .map(f => f.rebar_depth_mean)
    .filter((n): n is number => typeof n === 'number');
  if (!samples.length) return undefined;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

export default function GPRResults({ result, projectId }: Props) {
  const [tab,        setTab]        = useState<ResultsTab>('overview');
  const [needsRegen, setNeedsRegen] = useState(false);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <ResultsHeader
        name={result.analysis_name}
        notes={result.analysis_notes}
        company={result.company}
        project={result.project}
      />
      <ResultsTabs tab={tab} setTab={setTab} />
      {tab === 'overview'
        ? <OverviewTab
            result={result}
            projectId={projectId}
            needsRegen={needsRegen}
            onRegenerated={() => setNeedsRegen(false)}
          />
        : <InteractiveTabBody
            result={result}
            projectId={projectId}
            onPicksSaved={() => setNeedsRegen(true)}
          />}
    </div>
  );
}

function ResultsHeader({ name, notes, company, project }: {
  name?: string; notes?: string; company?: string; project?: string;
}) {
  const title = (name ?? '').trim() || 'Untitled Analysis';
  const trimmedNotes = (notes ?? '').trim();
  const companyTrimmed = (company ?? '').trim();
  const projectTrimmed = (project ?? '').trim();
  return (
    <div>
      {(companyTrimmed || projectTrimmed) && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
          {companyTrimmed && <HeaderChip>{companyTrimmed}</HeaderChip>}
          {projectTrimmed && <HeaderChip>{projectTrimmed}</HeaderChip>}
        </div>
      )}
      <h2 style={{
        margin: 0, fontSize: 22, fontWeight: 800, color: TEXT,
        letterSpacing: '-0.01em', lineHeight: 1.25,
      }}>
        {title}
      </h2>
      {trimmedNotes && (
        <p style={{
          margin: '6px 0 0', fontSize: 13, color: TEXT2,
          lineHeight: 1.55, whiteSpace: 'pre-wrap',
        }}>
          {trimmedNotes}
        </p>
      )}
    </div>
  );
}

function HeaderChip({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
      color: ACCENT, background: ACCENT_SOFT,
      padding: '4px 8px', borderRadius: 2,
    }}>
      {children}
    </span>
  );
}

function ResultsTabs({ tab, setTab }:
  { tab: ResultsTab; setTab: (t: ResultsTab) => void }) {
  const items: { id: ResultsTab; label: string; Icon: typeof LayoutGrid }[] = [
    { id: 'overview',    label: 'Overview',    Icon: LayoutGrid },
    { id: 'interactive', label: 'Interactive', Icon: Activity },
  ];
  return (
    <div role="tablist" style={{ display: 'flex', gap: 0, background: PANEL, border: `1px solid ${BORDER}`, alignSelf: 'flex-start' }}>
      {items.map(({ id, label, Icon }) => {
        const active = tab === id;
        return (
          <button
            key={id} role="tab" aria-selected={active}
            onClick={() => setTab(id)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '10px 16px', border: 'none', cursor: 'pointer',
              background: active ? ACCENT_SOFT : 'transparent',
              color: active ? ACCENT : TEXT2,
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

interface OverviewTabProps {
  result:        AnalysisResult;
  projectId?:    string;
  needsRegen:    boolean;
  onRegenerated: () => void;
}

function OverviewTab({ result, projectId, needsRegen, onRegenerated }: OverviewTabProps) {
  const amplitudeSrc = useMemo(
    () => resolveImageSrc(result.corrosion_map ?? result.amplitude_image),
    [result],
  );
  const staticDepthB64 = result.rebar_depth_map ?? result.rebar_depth_image;

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

  // per_file_summary is DZT-only; Proceq results don't carry it.
  const hasGps = (result.per_file_summary ?? []).some(f => f.gps);

  const conditionSrc = resolveImageSrc(result.cscan_url ?? result.cscan_image);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {conditionSrc && (
        <ConditionMapPanel result={result} src={conditionSrc} />
      )}
      <div className="gpr-panel-grid">
        <DepthMapCanvas
          jobId={projectId ?? ''}
          serverUrl={SERVER}
          analysisName={result.analysis_name ?? 'Analysis'}
          staticImageB64={staticDepthB64}
          needsRegen={needsRegen}
          onRegenerated={onRegenerated}
        />
        <ResultPanel title="Rebar Reflection Amplitude" src={amplitudeSrc} />
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '14px 20px', display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'center' }}>
        <Stat label="Mean Depth"      value={stats.mean !== undefined ? `${stats.mean.toFixed(2)}"` : '—'} />
        <Stat label="Deck Thickness"  value={stats.thick !== undefined ? `${stats.thick.toFixed(2)}"` : '—'} />
        <Stat label="Risk"            value={stats.risk !== undefined ? `${stats.risk.toFixed(0)}%` : '—'} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
      </div>

      {hasGps && (
        <div style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div style={{ padding: '12px 20px', borderBottom: `1px solid ${BORDER}`, fontSize: 11, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT2 }}>
            Scan Location
          </div>
          <div style={{ position: 'relative', height: 320 }}>
            <div ref={mapContainerRef} style={{ position: 'absolute', inset: 0 }} />
          </div>
        </div>
      )}

      <style>{`
        .gpr-panel-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }
        @media (max-width: 900px) {
          .gpr-panel-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}

interface InteractiveTabBodyProps {
  result:       AnalysisResult;
  projectId?:   string;
  onPicksSaved: () => void;
}

function InteractiveTabBody({ result, projectId, onPicksSaved }: InteractiveTabBodyProps) {
  if (!projectId) {
    return (
      <div style={{ background: PANEL, border: `1px dashed ${BORDER2}`, padding: '40px 24px', textAlign: 'center', color: TEXT2 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: TEXT, marginBottom: 6 }}>Save the project first</div>
        <div style={{ fontSize: 12 }}>The interactive pick editor needs a saved job id to load and persist picks.</div>
      </div>
    );
  }
  return (
    // Give BScanViewer the full viewport below the top header / tabs so the
    // B-scan dominates the workspace like professional GPR software. The
    // ~280px reserve covers the page header + tabs + their margins; the
    // inner viewer uses flex:1 to consume whatever is left.
    <div style={{
      height: 'calc(100vh - 280px)',
      minHeight: 480,
      background: PANEL, border: `1px solid ${BORDER}`, borderRadius: 8,
      padding: 12, display: 'flex', flexDirection: 'column',
    }}>
      <BScanViewer
        bscanData={result.bscan_data ?? []}
        jobId={projectId}
        serverUrl={SERVER}
        onPicksSaved={onPicksSaved}
      />
    </div>
  );
}

function ResultPanel({ title, src }: PanelSpec) {
  return (
    <div style={{
      background: PANEL, border: `1px solid ${BORDER}`,
      display: 'flex', flexDirection: 'column',
      minHeight: 400, overflow: 'hidden', borderRadius: 8,
    }}>
      <div style={{
        flexShrink: 0,
        padding: '10px 14px', borderBottom: `1px solid ${BORDER}`,
        fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT2,
      }}>
        {title}
      </div>
      <div style={{
        flex: 1, minHeight: 0,
        position: 'relative', overflow: 'hidden',
        background: RAISED,
      }}>
        {src ? (
          <img
            src={src}
            alt={title}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'fill', display: 'block' }}
          />
        ) : (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: TEXT3 }}>
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
