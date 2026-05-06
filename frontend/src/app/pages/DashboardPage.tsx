import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
import { Radio, Waves, Thermometer } from 'lucide-react';
import VerusLogo from '../components/VerusLogo';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../lib/supabase';
import type { AnalysisJob, InspectionModule } from './dashboard/types';
import ComingSoonModal from './dashboard/ComingSoonModal';
import JobTable from './dashboard/JobTable';

const MODULES: InspectionModule[] = [
  { id: 'gpr',      name: 'GPR',      fullName: 'Ground-Penetrating Radar',               status: 'available',      icon: Radio,       standard: 'ASTM D6087', description: 'Detects subsurface delamination from electromagnetic reflection patterns in GPR A-scan waveforms.' },
  { id: 'masw',     name: 'MASW',     fullName: 'Multichannel Analysis of Surface Waves', status: 'in-development', icon: Waves,       standard: 'ASTM D7400', description: 'Detects subsurface anomalies and layer stiffness from Rayleigh wave dispersion curves.' },
  { id: 'infrared', name: 'Infrared', fullName: 'Infrared Thermography',                  status: 'in-development', icon: Thermometer, standard: 'ASTM D4788', description: 'Detects subsurface delamination and moisture intrusion from thermal gradient patterns.' },
];

const INSPECT_ROUTES: Record<string, string> = {
  gpr:      '/inspect/gpr',
  masw:     '/inspect/masw',
  infrared: '/inspect/ir',
};

export default function DashboardPage() {
  const { auth, logout, user } = useAuth();
  const navigate = useNavigate();
  const [modalModule,   setModalModule]   = useState<InspectionModule | null>(null);
  const [jobs,          setJobs]          = useState<AnalysisJob[]>([]);
  const [jobsLoading,   setJobsLoading]   = useState(true);
  const [deleteTarget,  setDeleteTarget]  = useState<AnalysisJob | null>(null);
  const [deleteError,   setDeleteError]   = useState<string | null>(null);

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

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    const target = deleteTarget;
    setDeleteTarget(null);
    setJobs(prev => prev.filter(j => j.id !== target.id));
    try {
      const { error } = await supabase.from('analysis_jobs').delete().eq('id', target.id);
      if (error) throw error;
      if (target.project_id) {
        await supabase.from('projects').delete().eq('id', target.project_id);
      }
    } catch {
      setJobs(prev => [...prev, target].sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ));
      setDeleteError('Failed to delete project. Please try again.');
      setTimeout(() => setDeleteError(null), 4000);
    }
  };

  const handleModuleClick = (module: InspectionModule) => {
    const route = INSPECT_ROUTES[module.id];
    if (route) {
      if (module.id === 'gpr') localStorage.removeItem('verus_project_id');
      navigate(route);
    } else {
      setModalModule(module);
    }
  };

  const initials = auth.user?.name
    ? auth.user.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <div style={{ minHeight: '100vh', background: '#F5F3EF', fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif' }}>

      {/* Header */}
      <header style={{ background: '#FFFFFF', borderBottom: '2px solid #E2DED9', padding: '0 40px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <VerusLogo size={36} wordmarkColor="#0A0A0A" />
        </Link>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 34, height: 34, background: '#E8601C', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, color: '#FFFFFF', letterSpacing: '0.04em' }}>
              {initials}
            </div>
            <div>
              <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: '#0A0A0A' }}>{auth.user?.name || 'User'}</p>
              <p style={{ margin: 0, fontSize: 11, color: '#7A7470' }}>{auth.user?.email}</p>
            </div>
          </div>
          <div style={{ width: 1, height: 32, background: '#E2DED9' }} />
          <button onClick={() => { logout(); navigate('/', { replace: true }); }}
            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600, color: '#7A7470', letterSpacing: '0.04em', padding: '6px 0', fontFamily: 'Inter, sans-serif' }}>
            Log Out
          </button>
        </div>
      </header>

      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 40px' }}>

        {/* Welcome */}
        <div style={{ marginBottom: 48 }}>
          <h1 style={{ fontSize: 26, fontWeight: 800, color: '#0A0A0A', margin: '0 0 6px', letterSpacing: '-0.02em' }}>
            Welcome back{auth.user?.name ? `, ${auth.user.name.split(' ')[0]}` : ''}.
          </h1>
          <p style={{ fontSize: 14, color: '#7A7470', margin: 0 }}>Select an inspection method below to start a new analysis.</p>
        </div>

        {/* Module selector */}
        <section style={{ marginBottom: 56 }}>
          <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9' }}>
            <div style={{ padding: '14px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF' }}>
              <h2 style={{ margin: 0, fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7A7470' }}>Start a New Inspection</h2>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 0 }}>
              {MODULES.map((module, i) => {
                const Icon = module.icon;
                const isAvailable = module.status === 'available';
                const isLast = i === MODULES.length - 1;
                return (
                  <button key={module.id} onClick={() => handleModuleClick(module)}
                    style={{ width: '100%', textAlign: 'left', padding: '24px', background: '#FFFFFF', border: 'none', borderRight: isLast ? 'none' : '1px solid #E2DED9', borderLeft: '3px solid transparent', cursor: 'pointer', transition: 'border-color 0.15s, background 0.15s', fontFamily: 'Inter, sans-serif' }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderLeftColor = '#E8601C'; (e.currentTarget as HTMLElement).style.background = '#F5F3EF'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderLeftColor = 'transparent'; (e.currentTarget as HTMLElement).style.background = '#FFFFFF'; }}
                  >
                    <Icon style={{ color: '#0A0A0A', marginBottom: 14 }} className="w-6 h-6" />
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: '#0A0A0A' }}>{module.name}</span>
                      <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', padding: '3px 7px', flexShrink: 0, background: isAvailable ? '#2E7D32' : '#F5F3EF', color: isAvailable ? '#FFFFFF' : '#0A0A0A' }}>
                        {isAvailable ? 'Available' : 'In Development'}
                      </span>
                    </div>
                    <p style={{ fontSize: 11, color: '#7A7470', margin: '0 0 8px', lineHeight: 1.5 }}>{module.fullName}</p>
                    <p style={{ fontSize: 10, color: '#B0A9A4', margin: 0, fontWeight: 600, letterSpacing: '0.03em' }}>{module.standard}</p>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        {/* Recent projects */}
        <section>
          <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9' }}>
            <div style={{ padding: '14px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <h2 style={{ margin: 0, fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7A7470' }}>Recent Projects</h2>
              {jobs.length > 0 && <span style={{ fontSize: 11, color: '#B0A9A4' }}>{jobs.length} job{jobs.length !== 1 ? 's' : ''}</span>}
            </div>
            <JobTable jobs={jobs} loading={jobsLoading} onView={job => navigate(`/inspect/gpr?project_id=${job.id}`)} onDelete={setDeleteTarget} onStartFirst={() => navigate('/analyze')} />
          </div>
        </section>

      </main>

      {modalModule && <ComingSoonModal module={modalModule} onClose={() => setModalModule(null)} />}

      {deleteTarget && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 300, background: 'rgba(10,10,10,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}
          onClick={() => setDeleteTarget(null)}>
          <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9', width: 440, padding: 32 }} onClick={e => e.stopPropagation()}>
            <h3 style={{ margin: '0 0 12px', fontSize: 16, fontWeight: 800, color: '#0A0A0A', letterSpacing: '-0.01em' }}>Delete Project?</h3>
            <p style={{ margin: '0 0 28px', fontSize: 13, color: '#7A7470', lineHeight: 1.65 }}>
              This will permanently delete{' '}
              <strong style={{ color: '#0A0A0A' }}>{deleteTarget.project_name ?? 'Untitled Project'}</strong>
              {' '}and all associated analysis results. This cannot be undone.
            </p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
              <button onClick={() => setDeleteTarget(null)}
                style={{ padding: '9px 20px', background: 'none', border: '1.5px solid #E2DED9', color: '#7A7470', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
                Cancel
              </button>
              <button onClick={handleDeleteConfirm}
                style={{ padding: '9px 20px', background: '#ef4444', border: 'none', color: '#fff', fontSize: 12, fontWeight: 700, cursor: 'pointer', fontFamily: 'Inter, sans-serif', letterSpacing: '0.04em' }}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {deleteError && (
        <div style={{ position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', background: '#1a1a1a', color: '#fff', padding: '12px 20px', fontSize: 12, fontWeight: 600, borderRadius: 6, boxShadow: '0 8px 24px rgba(0,0,0,0.2)', zIndex: 400, whiteSpace: 'nowrap' }}>
          {deleteError}
        </div>
      )}
    </div>
  );
}
