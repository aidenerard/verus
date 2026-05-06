import { X, FolderOpen } from 'lucide-react';
import type { AnalysisJob } from './types';

interface Props {
  jobs: AnalysisJob[];
  loading: boolean;
  onView: (job: AnalysisJob) => void;
  onStartFirst: () => void;
}

export default function JobTable({ jobs, loading, onView, onStartFirst }: Props) {
  if (loading) {
    return <div style={{ padding: '48px 24px', textAlign: 'center' }}><p style={{ fontSize: 13, color: '#B0A9A4' }}>Loading…</p></div>;
  }

  if (jobs.length === 0) {
    return (
      <div style={{ padding: '64px 24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
        <FolderOpen className="w-12 h-12" style={{ color: '#E2DED9' }} />
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: 14, fontWeight: 600, color: '#B0A9A4', margin: '0 0 6px' }}>No projects yet</p>
          <p style={{ fontSize: 12, color: '#B0A9A4', margin: 0 }}>Run your first inspection to see project history here.</p>
        </div>
        <button onClick={onStartFirst} style={{ marginTop: 8, padding: '10px 24px', background: '#E8601C', color: '#FFFFFF', border: '2px solid #E8601C', fontWeight: 700, fontSize: 11, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
          Start First Inspection
        </button>
      </div>
    );
  }

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr style={{ background: '#F5F3EF', borderBottom: '2px solid #E2DED9' }}>
          {['Date', 'Files', 'Signals', 'Delamination', 'Status', ''].map(h => (
            <th key={h} style={{ textAlign: h === '' ? 'right' : 'left', padding: '10px 20px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {jobs.map(job => {
          const date = new Date(job.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
          const statusColor = job.status === 'complete' ? '#2E7D32' : job.status === 'failed' ? '#C0392B' : '#B0A9A4';
          const statusLabel = job.status === 'complete' ? 'Complete' : job.status === 'failed' ? 'Failed' : job.status === 'processing' ? 'Processing…' : 'Pending…';
          return (
            <tr key={job.id} style={{ borderTop: '1px solid #E2DED9' }}>
              <td style={{ padding: '12px 20px', fontSize: 12, color: '#0A0A0A' }}>{date}</td>
              <td style={{ padding: '12px 20px', fontSize: 11, color: '#7A7470', maxWidth: 200 }}>
                <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                  {job.file_names?.slice(0, 2).join(', ') ?? '—'}
                  {(job.file_names?.length ?? 0) > 2 && ` +${(job.file_names?.length ?? 0) - 2}`}
                </span>
              </td>
              <td style={{ padding: '12px 20px', fontSize: 12, color: '#7A7470' }}>{job.signals_analyzed?.toLocaleString() ?? '—'}</td>
              <td style={{ padding: '12px 20px', fontSize: 12 }}>
                {job.delamination_pct != null
                  ? <span style={{ fontWeight: 700, color: job.delamination_pct > 10 ? '#C0392B' : '#0A0A0A' }}>{job.delamination_pct.toFixed(1)}%</span>
                  : '—'}
              </td>
              <td style={{ padding: '12px 20px' }}>
                <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: statusColor }}>{statusLabel}</span>
              </td>
              <td style={{ padding: '12px 20px', textAlign: 'right' }}>
                {job.status === 'complete' && (
                  <button onClick={() => onView(job)} style={{ padding: '5px 14px', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', background: 'none', border: '1.5px solid #E2DED9', color: '#7A7470', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                    View
                  </button>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function ViewJobModal({ job, onClose }: { job: AnalysisJob; onClose: () => void }) {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(10,10,10,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }} onClick={onClose}>
      <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9', maxWidth: 720, width: '100%', maxHeight: '90vh', overflowY: 'auto' }} onClick={e => e.stopPropagation()}>
        <div style={{ padding: '16px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A' }}>Analysis Result</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4 }}>
            <X className="w-4 h-4" style={{ color: '#7A7470' }} />
          </button>
        </div>
        <div style={{ padding: 24 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Signals', value: job.signals_analyzed?.toLocaleString() ?? '—' },
              { label: 'Delamination', value: job.delamination_pct != null ? `${job.delamination_pct.toFixed(1)}%` : '—' },
              { label: 'Sound', value: job.sound_pct != null ? `${job.sound_pct.toFixed(1)}%` : '—' },
            ].map(({ label, value }) => (
              <div key={label} style={{ border: '2px solid #E2DED9', padding: '14px 18px' }}>
                <p style={{ margin: '0 0 4px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470' }}>{label}</p>
                <p style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#0A0A0A' }}>{value}</p>
              </div>
            ))}
          </div>
          {job.cscan_url && (
            <div>
              <p style={{ margin: '0 0 10px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470' }}>C-Scan Map</p>
              <img src={job.cscan_url} alt="C-Scan" style={{ width: '100%', display: 'block', border: '2px solid #E2DED9' }} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
