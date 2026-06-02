import { useMemo, useState } from 'react';
import { ImageOff, LayoutGrid, Activity, Download } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { useMapbox } from '../inspect/useMapbox';
import { SERVER } from '../inspect/constants';
import BScanViewer from './BScanViewer';
import ResultsExportTab from './ResultsExportTab';
import { BORDER, BORDER2, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT, ACCENT_SOFT } from './tokens';

import type { BScanViewerHandle } from './BScanViewer';

interface Props {
  result:     AnalysisResult;
  projectId?: string;
  tab?:       ResultsTab;
  setTab?:    (t: ResultsTab) => void;
  bscanRef?:  React.Ref<BScanViewerHandle>;
}

export type ResultsTab = 'overview' | 'interactive' | 'export';

interface PanelSpec { title: string; src: string | undefined }

function resolveImageSrc(
  urlField?: string | null,
  b64Field?: string | null,
): string | undefined {
  // Prefer a storage URL (large blobs are offloaded to the job-results
  // bucket); fall back to an inline base64 PNG for older/local results.
  if (urlField) return urlField;
  if (!b64Field) return undefined;
  if (b64Field.startsWith('http') || b64Field.startsWith('data:')) return b64Field;
  return `data:image/png;base64,${b64Field}`;
}

function meanRebarDepth(result: AnalysisResult): number | undefined {
  const samples = (result.per_file_summary ?? [])
    .map(f => f.rebar_depth_mean)
    .filter((n): n is number => typeof n === 'number');
  if (!samples.length) return undefined;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

export default function GPRResults({
  result, projectId, tab: tabProp, setTab: setTabProp, bscanRef,
}: Props) {
  const [tabLocal, setTabLocal] = useState<ResultsTab>('overview');
  const tab    = tabProp    ?? tabLocal;
  const setTab = setTabProp ?? setTabLocal;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <ResultsHeader
        name={result.analysis_name}
        notes={result.analysis_notes}
        company={result.company}
        project={result.project}
      />
      <ResultsTabs tab={tab} setTab={setTab} />
      {tab === 'overview' && (
        <OverviewTab result={result} />
      )}
      {tab === 'interactive' && (
        <InteractiveTabBody
          result={result}
          projectId={projectId}
          bscanRef={bscanRef}
        />
      )}
      {tab === 'export' && (
        <ResultsExportTab result={result} projectId={projectId} />
      )}
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
    { id: 'export',      label: 'Export',      Icon: Download },
  ];
  return (
    <div role="tablist" style={{ display: 'flex', gap: 0, background: PANEL, border: `1px solid ${BORDER}`, alignSelf: 'flex-start' }}>
      {items.map(({ id, label, Icon }, i) => {
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
              borderRight: i < items.length - 1 ? `1px solid ${BORDER}` : 'none',
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
  result: AnalysisResult;
}

function astmStatus(risk?: number): string {
  if (risk === undefined) return '—';
  if (risk < 10) return 'Sound';
  if (risk < 30) return 'Monitor';
  return 'Action';
}

function OverviewTab({ result }: OverviewTabProps) {
  // Both maps come from the SAME canonical backend renderers so the output is
  // fixed and identical across datasets (depth: build_unified_depth_map,
  // YlOrRd_r 3.0–8.5"; corrosion: render_corrosion_db_map). No client-side
  // auto-scaling — see docs/DEPTH_MAP_SPEC.md.
  const depthSrc = useMemo(
    () => resolveImageSrc(
      result.rebar_depth_map_url ?? result.rebar_depth_image_url,
      result.rebar_depth_map ?? result.rebar_depth_image,
    ),
    [result],
  );
  const corrosionSrc = useMemo(
    () => resolveImageSrc(result.corrosion_map_url, result.corrosion_map),
    [result],
  );

  const stats = useMemo(() => {
    const mean     = result.mean_depth_inches ?? meanRebarDepth(result);
    const risk     = result.high_risk_pct     ?? result.delamination_pct;
    const coverage = result.signals_analyzed  ?? result.stats?.n_traces;
    return { mean, risk, coverage };
  }, [result]);

  const { mapContainerRef } = useMapbox({
    analysisResult: result,
    layerVis:       { gpr: true, condition: true, amplitude: false, satellite: true, annotations: true },
    conditionOpacity: 80,
  });

  // per_file_summary is DZT-only; Proceq results don't carry it.
  const hasGps = (result.per_file_summary ?? []).some(f => f.gps);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="gpr-panel-grid">
        <ResultPanel title="Rebar Depth Map" src={depthSrc} />
        <ResultPanel title="Corrosion Risk Map" src={corrosionSrc} />
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '14px 20px', display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'center' }}>
        <Stat label="Mean Depth"  value={stats.mean !== undefined ? `${stats.mean.toFixed(2)}"` : '—'} />
        <Stat label="High Risk"   value={stats.risk !== undefined ? `${stats.risk.toFixed(0)}%` : '—'} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
        <Stat label="ASTM D6087"  value={astmStatus(stats.risk)} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
        <Stat label="Coverage"    value={stats.coverage !== undefined ? `${stats.coverage.toLocaleString()} traces` : '—'} />
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
  result:     AnalysisResult;
  projectId?: string;
  bscanRef?:  React.Ref<BScanViewerHandle>;
}

function InteractiveTabBody({ result, projectId, bscanRef }: InteractiveTabBodyProps) {
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
        ref={bscanRef}
        bscanData={result.bscan_data ?? []}
        bscanUrl={result.bscan_data_url}
        jobId={projectId}
        serverUrl={SERVER}
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
