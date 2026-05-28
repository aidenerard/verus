import { useState } from 'react';
import { FolderOpen, Trash2, ArrowUpRight, Radio } from 'lucide-react';
import type { AnalysisJob } from './types';

const PANEL  = '#FFFFFF';
const RAISED = '#F5F3EF';
const BORDER = '#E2DED9';
const TEXT   = '#0A0A0A';
const TEXT2  = '#7A7470';
const TEXT3  = '#B0A9A4';
const ACCENT = '#E8601C';

interface Props {
  jobs:         AnalysisJob[];
  loading:      boolean;
  onView:       (job: AnalysisJob) => void;
  onDelete:     (job: AnalysisJob) => void;
  onStartFirst: () => void;
}

interface Status { label: string; color: string; dot: string }

function statusOf(s: AnalysisJob['status']): Status {
  if (s === 'complete')   return { label: 'Complete',   color: '#2E7D32', dot: '#22c55e' };
  if (s === 'failed')     return { label: 'Failed',     color: '#C0392B', dot: '#ef4444' };
  if (s === 'processing') return { label: 'Processing', color: TEXT2,     dot: '#f59e0b' };
  return                          { label: 'Pending',   color: TEXT3,     dot: TEXT3 };
}

export default function JobCards({ jobs, loading, onView, onDelete, onStartFirst }: Props) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  if (loading) {
    return (
      <div style={{ padding: '64px 24px', textAlign: 'center', color: TEXT3, fontSize: 13 }}>Loading…</div>
    );
  }

  if (jobs.length === 0) {
    return (
      <div style={{ padding: '72px 24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18 }}>
        <FolderOpen className="w-12 h-12" style={{ color: BORDER }} />
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: 15, fontWeight: 700, color: TEXT, margin: '0 0 6px' }}>No inspections yet</p>
          <p style={{ fontSize: 13, color: TEXT2, margin: 0 }}>Run your first inspection to see project history here.</p>
        </div>
        <button onClick={onStartFirst}
          style={{ marginTop: 4, padding: '11px 26px', background: ACCENT, color: '#fff', border: 'none', fontWeight: 700, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
          Start First Inspection
        </button>
      </div>
    );
  }

  return (
    <div className="job-card-grid" style={{ display: 'grid', gap: 16, padding: 20 }}>
      {jobs.map(job => {
        const date = new Date(job.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        const st = statusOf(job.status);
        const isConfirming = confirmingId === job.id;
        const canView = job.status === 'complete';
        const fileCount = job.file_names?.length ?? 0;
        const signals = job.signals_analyzed?.toLocaleString();

        return (
          <article key={job.id}
            style={{
              background: PANEL, border: `1px solid ${BORDER}`, padding: '18px 20px',
              display: 'flex', flexDirection: 'column', gap: 12, minHeight: 168,
              transition: 'border-color 0.15s, box-shadow 0.15s, transform 0.15s',
              opacity: isConfirming ? 0.85 : 1,
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#D4CFC9'; e.currentTarget.style.boxShadow = '0 8px 24px rgba(10,10,10,0.05)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = BORDER; e.currentTarget.style.boxShadow = 'none'; }}>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 9, fontWeight: 800, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT2, padding: '3px 7px', background: RAISED, border: `1px solid ${BORDER}` }}>
                <Radio size={10} /> GPR
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: st.color }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: st.dot }} />
                {st.label}
              </span>
            </div>

            <div>
              <div
                title={job.analysis_name || job.project_name || 'Untitled Analysis'}
                style={{ fontSize: 14, fontWeight: 700, color: TEXT, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              >
                {job.analysis_name?.trim() || job.project_name || 'Untitled Analysis'}
              </div>
              {job.analysis_notes && job.analysis_notes.trim() && (
                <p
                  title={job.analysis_notes}
                  className="job-card-notes"
                  style={{ margin: '6px 0 0', fontSize: 12, color: TEXT2, lineHeight: 1.45 }}
                >
                  {job.analysis_notes.trim()}
                </p>
              )}
              <div style={{ fontSize: 11, color: TEXT2, marginTop: 6 }}>{date}</div>
            </div>

            <div style={{ fontSize: 11, color: TEXT2, fontVariantNumeric: 'tabular-nums' }}>
              {fileCount > 0 ? `${fileCount} file${fileCount !== 1 ? 's' : ''}` : '—'}
              {signals ? ` · ${signals} signals` : ''}
            </div>

            <div style={{ flex: 1 }} />

            {isConfirming ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'space-between' }}>
                <span style={{ fontSize: 11, color: TEXT2, whiteSpace: 'nowrap' }}>Permanently delete?</span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button onClick={() => setConfirmingId(null)}
                    style={{ padding: '6px 12px', fontSize: 10, fontWeight: 600, background: 'none', border: `1px solid ${BORDER}`, color: TEXT2, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                    Cancel
                  </button>
                  <button onClick={() => { onDelete(job); setConfirmingId(null); }}
                    style={{ padding: '6px 14px', fontSize: 10, fontWeight: 700, background: '#ef4444', border: '1px solid #ef4444', color: '#fff', cursor: 'pointer', fontFamily: 'Inter, sans-serif', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    Delete
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
                {canView ? (
                  <button onClick={() => onView(job)}
                    style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: ACCENT, fontFamily: 'Inter, sans-serif' }}>
                    View <ArrowUpRight size={12} />
                  </button>
                ) : (
                  <span style={{ fontSize: 11, color: TEXT3 }}>—</span>
                )}
                <button onClick={() => setConfirmingId(job.id)} aria-label="Delete project"
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT3, padding: 4, display: 'flex' }}
                  onMouseEnter={e => (e.currentTarget.style.color = '#ef4444')}
                  onMouseLeave={e => (e.currentTarget.style.color = TEXT3)}>
                  <Trash2 size={14} />
                </button>
              </div>
            )}
          </article>
        );
      })}

      <style>{`
        .job-card-grid { grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
        @media (max-width: 640px) {
          .job-card-grid { grid-template-columns: 1fr; }
        }
        .job-card-notes {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
          overflow-wrap: anywhere;
        }
      `}</style>
    </div>
  );
}
