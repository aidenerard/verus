/**
 * ProjectsDrawer.tsx
 * Full-screen overlay with a slide-in right panel listing recent completed
 * analysis jobs. Clicking a job loads it into the workspace.
 *
 * Does NOT: fetch jobs itself or manage analysis state — the parent fetches
 * recentJobs and provides loadJob.
 */

import { X } from 'lucide-react';

import { PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2 } from './constants';
import { delamColor } from './utils';

interface RecentJob {
  id: string;
  created_at: string;
  signals_analyzed?: number;
  delamination_pct?: number;
  result?: unknown;
}

interface ProjectsDrawerProps {
  recentJobs: RecentJob[];
  loadJob: (job: RecentJob) => void;
  setShowProjects: (v: boolean) => void;
}

export default function ProjectsDrawer({ recentJobs, loadJob, setShowProjects }: ProjectsDrawerProps) {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'flex' }} onClick={() => setShowProjects(false)}>
      <div style={{ flex: 1 }} />
      <div style={{ width: 360, height: '100%', background: PANEL, borderLeft: `1px solid ${BORDER2}`, display: 'flex', flexDirection: 'column', boxShadow: '-8px 0 32px rgba(0,0,0,0.12)' }} onClick={e => e.stopPropagation()}>
        <div style={{ padding: '14px 20px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: TEXT }}>My Projects</span>
          <button onClick={() => setShowProjects(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2 }}><X size={16} /></button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {recentJobs.length === 0
            ? <div style={{ padding: 40, textAlign: 'center' }}><p style={{ fontSize: 13, color: TEXT2 }}>No completed analyses yet.</p></div>
            : recentJobs.map(job => (
              <div key={job.id} onClick={() => loadJob(job)}
                style={{ padding: '14px 20px', borderBottom: `1px solid ${BORDER}`, cursor: 'pointer' }}
                onMouseEnter={e => (e.currentTarget.style.background = RAISED)}
                onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
                <div style={{ fontSize: 12, fontWeight: 600, color: TEXT, marginBottom: 4 }}>
                  Analysis — {new Date(job.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </div>
                <div style={{ display: 'flex', gap: 16, fontSize: 11, color: TEXT2 }}>
                  <span>{job.signals_analyzed?.toLocaleString() ?? '—'} signals</span>
                  <span style={{ color: delamColor(job.delamination_pct ?? 0) }}>{job.delamination_pct?.toFixed(1) ?? '—'}% delam</span>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
