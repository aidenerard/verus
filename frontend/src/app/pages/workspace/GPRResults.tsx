import { useState } from 'react';
import { LayoutGrid, Activity, Download } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { SERVER } from '../inspect/constants';
import BScanViewer from './BScanViewer';
import ResultsExportTab from './ResultsExportTab';
import OverviewTab from './ResultsOverview';
import { BORDER, BORDER2, PANEL, TEXT, TEXT2, ACCENT, ACCENT_SOFT } from './tokens';

import type { BScanViewerHandle } from './BScanViewer';

interface Props {
  result:     AnalysisResult;
  projectId?: string;
  tab?:       ResultsTab;
  setTab?:    (t: ResultsTab) => void;
  bscanRef?:  React.Ref<BScanViewerHandle>;
}

export type ResultsTab = 'overview' | 'interactive' | 'export';

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
      {tab === 'overview' && <OverviewTab result={result} />}
      {tab === 'interactive' && (
        <InteractiveTabBody result={result} projectId={projectId} bscanRef={bscanRef} />
      )}
      {tab === 'export' && <ResultsExportTab result={result} projectId={projectId} />}
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
    // B-scan dominates the workspace like professional GPR software.
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
