import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
import { Radio, Waves, Thermometer, FolderOpen, X } from 'lucide-react';
import VerusLogo from '../components/VerusLogo';
import { useAuth } from '../../context/AuthContext';
import { supabase } from '../../lib/supabase';

interface AnalysisJob {
  id: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  created_at: string;
  completed_at?: string;
  signals_analyzed?: number;
  delamination_pct?: number;
  sound_pct?: number;
  analysis_time_sec?: number;
  cscan_url?: string;
  file_names?: string[];
  error_msg?: string;
}

// ── Inspection methods (mirrors the sidebar in App.tsx) ───────────────────────

interface InspectionModule {
  id: string;
  name: string;
  fullName: string;
  status: 'available' | 'in-development';
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  standard: string;
  description: string;
}

const MODULES: InspectionModule[] = [
  {
    id: 'gpr',
    name: 'GPR',
    fullName: 'Ground-Penetrating Radar',
    status: 'available',
    icon: Radio,
    standard: 'ASTM D6087',
    description: 'Detects subsurface delamination from electromagnetic reflection patterns in GPR A-scan waveforms.',
  },
  {
    id: 'masw',
    name: 'MASW',
    fullName: 'Multichannel Analysis of Surface Waves',
    status: 'in-development',
    icon: Waves,
    standard: 'ASTM D7400',
    description: 'Detects subsurface anomalies and layer stiffness from Rayleigh wave dispersion curves.',
  },
  {
    id: 'infrared',
    name: 'Infrared',
    fullName: 'Infrared Thermography',
    status: 'in-development',
    icon: Thermometer,
    standard: 'ASTM D4788',
    description: 'Detects subsurface delamination and moisture intrusion from thermal gradient patterns.',
  },
];

// ── Coming-soon modal ─────────────────────────────────────────────────────────

function ComingSoonModal({ module, onClose }: {
  module: InspectionModule;
  onClose: () => void;
}) {
  const Icon = module.icon;
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 200,
      background: 'rgba(10,10,10,0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }} onClick={onClose}>
      <div
        style={{
          background: '#FFFFFF', border: '2px solid #E2DED9',
          maxWidth: 440, width: '100%',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Modal header */}
        <div style={{
          padding: '16px 24px', borderBottom: '2px solid #E2DED9',
          background: '#F5F3EF', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Icon style={{ color: '#0A0A0A' }} className="w-4 h-4" />
            <span style={{ fontSize: 13, fontWeight: 700, color: '#0A0A0A', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {module.fullName}
            </span>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4 }}>
            <X className="w-4 h-4" style={{ color: '#7A7470' }} />
          </button>
        </div>

        {/* Modal body */}
        <div style={{ padding: '28px 24px' }}>
          <div style={{
            display: 'inline-block',
            background: '#F5F3EF', color: '#0A0A0A',
            padding: '4px 10px', marginBottom: 16,
            fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
          }}>
            In Development
          </div>
          <p style={{ fontSize: 14, color: '#0A0A0A', lineHeight: 1.6, margin: '0 0 16px' }}>
            This module is coming soon. Verus is training a dedicated AI model
            for <strong>{module.fullName}</strong> analysis.
          </p>
          <p style={{ fontSize: 13, color: '#7A7470', lineHeight: 1.6, margin: '0 0 24px' }}>
            {module.description} You'll be notified when it launches.
          </p>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: '#B0A9A4' }}>Standard: {module.standard}</span>
            <button
              onClick={onClose}
              style={{
                padding: '9px 22px',
                background: '#E8601C', color: '#FFFFFF',
                border: '2px solid #E8601C',
                fontWeight: 700, fontSize: 11,
                letterSpacing: '0.07em', textTransform: 'uppercase',
                cursor: 'pointer', fontFamily: 'Inter, sans-serif',
              }}
            >
              Got It
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { auth, logout, user } = useAuth();
  const navigate = useNavigate();
  const [modalModule, setModalModule] = useState<InspectionModule | null>(null);
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [viewJob, setViewJob] = useState<AnalysisJob | null>(null);

  useEffect(() => {
    if (!user) return;
    const loadJobs = async () => {
      setJobsLoading(true);
      try {
        const { data } = await supabase
          .from('analysis_jobs')
          .select('*')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(20);
        setJobs(data ?? []);
      } catch {
        setJobs([]);
      } finally {
        setJobsLoading(false);
      }
    };
    loadJobs();
  }, [user]);

  const handleLogout = () => {
    logout();
    navigate('/', { replace: true });
  };

  const INSPECT_ROUTES: Record<string, string> = {
    'gpr':          '/inspect/gpr',
    'masw':         '/inspect/masw',
    'infrared':     '/inspect/ir',
  };

  const handleModuleClick = (module: InspectionModule) => {
    const route = INSPECT_ROUTES[module.id];
    if (route) {
      navigate(route);
    } else {
      setModalModule(module);
    }
  };

  const initials = auth.user?.name
    ? auth.user.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <div style={{
      minHeight: '100vh',
      background: '#F5F3EF',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      {/* Top navigation */}
      <header style={{
        background: '#FFFFFF',
        borderBottom: '2px solid #E2DED9',
        padding: '0 40px',
        height: 64,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <VerusLogo size={36} wordmarkColor="#0A0A0A" />
        </Link>

        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          {/* Avatar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 34, height: 34,
              background: '#E8601C',
              borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 700, color: '#FFFFFF',
              letterSpacing: '0.04em',
            }}>
              {initials}
            </div>
            <div>
              <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: '#0A0A0A' }}>
                {auth.user?.name || 'User'}
              </p>
              <p style={{ margin: 0, fontSize: 11, color: '#7A7470' }}>
                {auth.user?.email}
              </p>
            </div>
          </div>

          <div style={{ width: 1, height: 32, background: '#E2DED9' }} />

          <button
            onClick={handleLogout}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              fontSize: 12, fontWeight: 600, color: '#7A7470',
              letterSpacing: '0.04em', padding: '6px 0',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            Log Out
          </button>
        </div>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 40px' }}>

        {/* Welcome banner */}
        <div style={{ marginBottom: 48 }}>
          <h1 style={{
            fontSize: 26, fontWeight: 800, color: '#0A0A0A',
            margin: '0 0 6px', letterSpacing: '-0.02em',
          }}>
            Welcome back{auth.user?.name ? `, ${auth.user.name.split(' ')[0]}` : ''}.
          </h1>
          <p style={{ fontSize: 14, color: '#7A7470', margin: 0 }}>
            Select an inspection method below to start a new analysis.
          </p>
        </div>

        {/* ── Inspection method selector ── */}
        <section style={{ marginBottom: 56 }}>
          <div style={{
            background: '#FFFFFF', border: '2px solid #E2DED9',
          }}>
            {/* Section header */}
            <div style={{
              padding: '14px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF',
            }}>
              <h2 style={{
                margin: 0, fontSize: 11, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7A7470',
              }}>
                Start a New Inspection
              </h2>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 0,
            }}>
              {MODULES.map((module, i) => {
                const Icon = module.icon;
                const isAvailable = module.status === 'available';
                const isLast = i === MODULES.length - 1;
                return (
                  <button
                    key={module.id}
                    onClick={() => handleModuleClick(module)}
                    style={{
                      width: '100%', textAlign: 'left', padding: '24px',
                      background: '#FFFFFF',
                      border: 'none',
                      borderRight: isLast ? 'none' : '1px solid #E2DED9',
                      borderLeft: '3px solid transparent',
                      cursor: 'pointer',
                      transition: 'border-color 0.15s, background 0.15s',
                      fontFamily: 'Inter, sans-serif',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.borderLeftColor = '#E8601C';
                      (e.currentTarget as HTMLElement).style.background = '#F5F3EF';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.borderLeftColor = 'transparent';
                      (e.currentTarget as HTMLElement).style.background = '#FFFFFF';
                    }}
                  >
                    <Icon style={{ color: '#0A0A0A', marginBottom: 14 }} className="w-6 h-6" />

                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: '#0A0A0A' }}>
                        {module.name}
                      </span>
                      <span style={{
                        fontSize: 9, fontWeight: 700, letterSpacing: '0.07em',
                        textTransform: 'uppercase', padding: '3px 7px', flexShrink: 0,
                        background: isAvailable ? '#2E7D32' : '#F5F3EF',
                        color: isAvailable ? '#FFFFFF' : '#0A0A0A',
                      }}>
                        {isAvailable ? 'Available' : 'In Development'}
                      </span>
                    </div>

                    <p style={{ fontSize: 11, color: '#7A7470', margin: '0 0 8px', lineHeight: 1.5 }}>
                      {module.fullName}
                    </p>
                    <p style={{ fontSize: 10, color: '#B0A9A4', margin: 0, fontWeight: 600, letterSpacing: '0.03em' }}>
                      {module.standard}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        {/* ── Recent Projects ── */}
        <section>
          <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9' }}>
            <div style={{
              padding: '14px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <h2 style={{
                margin: 0, fontSize: 11, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7A7470',
              }}>
                Recent Projects
              </h2>
              {jobs.length > 0 && (
                <span style={{ fontSize: 11, color: '#B0A9A4' }}>{jobs.length} job{jobs.length !== 1 ? 's' : ''}</span>
              )}
            </div>

            {jobsLoading ? (
              <div style={{ padding: '48px 24px', textAlign: 'center' }}>
                <p style={{ fontSize: 13, color: '#B0A9A4' }}>Loading…</p>
              </div>
            ) : jobs.length === 0 ? (
              /* Empty state */
              <div style={{
                padding: '64px 24px',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 16,
              }}>
                <FolderOpen className="w-12 h-12" style={{ color: '#E2DED9' }} />
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 14, fontWeight: 600, color: '#B0A9A4', margin: '0 0 6px' }}>
                    No projects yet
                  </p>
                  <p style={{ fontSize: 12, color: '#B0A9A4', margin: 0 }}>
                    Run your first inspection to see project history here.
                  </p>
                </div>
                <button
                  onClick={() => navigate('/analyze')}
                  style={{
                    marginTop: 8, padding: '10px 24px',
                    background: '#E8601C', color: '#FFFFFF',
                    border: '2px solid #E8601C',
                    fontWeight: 700, fontSize: 11,
                    letterSpacing: '0.07em', textTransform: 'uppercase',
                    cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  }}
                >
                  Start First Inspection
                </button>
              </div>
            ) : (
              /* Job table */
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#F5F3EF', borderBottom: '2px solid #E2DED9' }}>
                    {['Date', 'Files', 'Signals', 'Delamination', 'Status', ''].map(h => (
                      <th key={h} style={{
                        textAlign: h === '' ? 'right' : 'left',
                        padding: '10px 20px', fontSize: 10, fontWeight: 700,
                        textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470',
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(job => {
                    const date = new Date(job.created_at).toLocaleDateString('en-US', {
                      month: 'short', day: 'numeric', year: 'numeric',
                    });
                    const statusColor = job.status === 'complete'
                      ? '#2E7D32' : job.status === 'failed'
                      ? '#C0392B' : '#B0A9A4';
                    const statusLabel = job.status === 'complete' ? 'Complete'
                      : job.status === 'failed' ? 'Failed'
                      : job.status === 'processing' ? 'Processing…' : 'Pending…';
                    return (
                      <tr key={job.id} style={{ borderTop: '1px solid #E2DED9' }}>
                        <td style={{ padding: '12px 20px', fontSize: 12, color: '#0A0A0A' }}>{date}</td>
                        <td style={{ padding: '12px 20px', fontSize: 11, color: '#7A7470', maxWidth: 200 }}>
                          <span style={{ fontFamily: 'monospace', fontSize: 10 }}>
                            {job.file_names?.slice(0, 2).join(', ') ?? '—'}
                            {(job.file_names?.length ?? 0) > 2 && ` +${(job.file_names?.length ?? 0) - 2}`}
                          </span>
                        </td>
                        <td style={{ padding: '12px 20px', fontSize: 12, color: '#7A7470' }}>
                          {job.signals_analyzed?.toLocaleString() ?? '—'}
                        </td>
                        <td style={{ padding: '12px 20px', fontSize: 12 }}>
                          {job.delamination_pct != null ? (
                            <span style={{ fontWeight: 700, color: job.delamination_pct > 10 ? '#C0392B' : '#0A0A0A' }}>
                              {job.delamination_pct.toFixed(1)}%
                            </span>
                          ) : '—'}
                        </td>
                        <td style={{ padding: '12px 20px' }}>
                          <span style={{
                            fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                            textTransform: 'uppercase', color: statusColor,
                          }}>
                            {statusLabel}
                          </span>
                        </td>
                        <td style={{ padding: '12px 20px', textAlign: 'right' }}>
                          {job.status === 'complete' && (
                            <button
                              onClick={() => setViewJob(job)}
                              style={{
                                padding: '5px 14px', fontSize: 10, fontWeight: 700,
                                letterSpacing: '0.06em', textTransform: 'uppercase',
                                background: 'none', border: '1.5px solid #E2DED9',
                                color: '#7A7470', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                              }}
                            >
                              View
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* ── View Job Modal ── */}
        {viewJob && (
          <div style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(10,10,10,0.55)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24,
          }} onClick={() => setViewJob(null)}>
            <div
              style={{ background: '#FFFFFF', border: '2px solid #E2DED9', maxWidth: 720, width: '100%', maxHeight: '90vh', overflowY: 'auto' }}
              onClick={e => e.stopPropagation()}
            >
              <div style={{
                padding: '16px 24px', borderBottom: '2px solid #E2DED9',
                background: '#F5F3EF', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <span style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A' }}>
                  Analysis Result
                </span>
                <button onClick={() => setViewJob(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4 }}>
                  <X className="w-4 h-4" style={{ color: '#7A7470' }} />
                </button>
              </div>
              <div style={{ padding: 24 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
                  {[
                    { label: 'Signals', value: viewJob.signals_analyzed?.toLocaleString() ?? '—' },
                    { label: 'Delamination', value: viewJob.delamination_pct != null ? `${viewJob.delamination_pct.toFixed(1)}%` : '—' },
                    { label: 'Sound', value: viewJob.sound_pct != null ? `${viewJob.sound_pct.toFixed(1)}%` : '—' },
                  ].map(({ label, value }) => (
                    <div key={label} style={{ border: '2px solid #E2DED9', padding: '14px 18px' }}>
                      <p style={{ margin: '0 0 4px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470' }}>{label}</p>
                      <p style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#0A0A0A' }}>{value}</p>
                    </div>
                  ))}
                </div>
                {viewJob.cscan_url && (
                  <div>
                    <p style={{ margin: '0 0 10px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#7A7470' }}>C-Scan Map</p>
                    <img src={viewJob.cscan_url} alt="C-Scan" style={{ width: '100%', display: 'block', border: '2px solid #E2DED9' }} />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Coming-soon modal */}
      {modalModule && (
        <ComingSoonModal module={modalModule} onClose={() => setModalModule(null)} />
      )}
    </div>
  );
}
