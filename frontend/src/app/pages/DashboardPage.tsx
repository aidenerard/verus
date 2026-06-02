import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { Plus } from 'lucide-react';
import Navbar, { NAVBAR_HEIGHT } from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../lib/supabase';
import type { AnalysisJob } from './dashboard/types';
import JobCards from './dashboard/JobCards';

export default function DashboardPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [jobs,        setJobs]        = useState<AnalysisJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;
    setJobsLoading(true);
    (async () => {
      try {
        const { data: jobData } = await supabase.from('analysis_jobs').select('*')
          .eq('user_id', user.id).order('created_at', { ascending: false }).limit(20);
        const loaded = (jobData ?? []) as AnalysisJob[];
        const pids = [...new Set(loaded.map(j => j.project_id).filter(Boolean))] as string[];
        if (pids.length) {
          const { data: projects } = await supabase.from('projects')
            .select('id,name,structure_name').in('id', pids);
          const names = new Map((projects ?? []).map(p => [p.id, (p.name || p.structure_name) ?? 'Untitled Project']));
          setJobs(loaded.map(j => ({ ...j, project_name: j.project_id ? (names.get(j.project_id) ?? 'Untitled Project') : undefined })));
        } else {
          setJobs(loaded);
        }
      } catch {
        setJobs([]);
      }
      setJobsLoading(false);
    })();
  }, [user]);

  const handleDelete = async (job: AnalysisJob) => {
    if (!user) return;
    try {
      const { error: jobErr, count } = await supabase
        .from('analysis_jobs')
        .delete({ count: 'exact' })
        .eq('id', job.id);
      if (jobErr) throw jobErr;
      if (count === 0) throw new Error('Delete was blocked by the database. Run migration 007 in your Supabase SQL Editor, then try again.');

      if (job.project_id) {
        const { count: remaining } = await supabase
          .from('analysis_jobs')
          .select('*', { count: 'exact', head: true })
          .eq('project_id', job.project_id);
        if (remaining === 0) {
          const { error: projErr } = await supabase
            .from('projects')
            .delete()
            .eq('id', job.project_id);
          if (projErr) throw projErr;
        }
        if (job.project_id === localStorage.getItem('verus_project_id')) {
          localStorage.removeItem('verus_project_id');
        }
      }

      setJobs(prev => prev.filter(j => j.id !== job.id));
    } catch (err) {
      console.error('[delete] failed:', err);
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete. Please try again.');
      setTimeout(() => setDeleteError(null), 8000);
    }
  };

  const startNew = () => navigate('/workspace');

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#F5F3EF', fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif' }}>
      <Navbar />

      <main style={{ flex: 1, width: '100%', maxWidth: 1280, margin: '0 auto', padding: `${NAVBAR_HEIGHT + 40}px 40px 64px` }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 24, marginBottom: 32, flexWrap: 'wrap' }}>
          <div>
            <h1 style={{ fontSize: 28, fontWeight: 800, color: '#0A0A0A', margin: '0 0 6px', letterSpacing: '-0.02em' }}>
              Your Inspections
            </h1>
            <p style={{ fontSize: 14, color: '#7A7470', margin: 0 }}>
              {jobs.length === 0 ? 'Start a new inspection to populate this page.' : `${jobs.length} project${jobs.length !== 1 ? 's' : ''} on file.`}
            </p>
          </div>
          <button onClick={startNew}
            style={{
              padding: '11px 22px', background: '#E8601C', color: '#FFFFFF', border: 'none',
              fontWeight: 700, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase',
              cursor: 'pointer', fontFamily: 'Inter, sans-serif',
              display: 'inline-flex', alignItems: 'center', gap: 8,
              boxShadow: '0 4px 12px rgba(232,96,28,0.25)',
            }}>
            <Plus size={14} /> New Inspection
          </button>
        </div>

        <section style={{ background: '#FFFFFF', border: '2px solid #E2DED9' }}>
          <JobCards
            jobs={jobs}
            loading={jobsLoading}
            onView={job => navigate(`/workspace/em/gpr?project_id=${job.id}`)}
            onDelete={handleDelete}
            onStartFirst={startNew}
          />
        </section>
      </main>

      {deleteError && (
        <div style={{ position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', background: '#1a1a1a', color: '#fff', padding: '12px 20px', fontSize: 12, fontWeight: 600, borderRadius: 6, boxShadow: '0 8px 24px rgba(0,0,0,0.2)', zIndex: 400, whiteSpace: 'nowrap' }}>
          {deleteError}
        </div>
      )}

      <Footer />
    </div>
  );
}
