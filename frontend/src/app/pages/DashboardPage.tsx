import { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { Radio, Waves, Thermometer, Speaker, FolderOpen, X } from 'lucide-react';
import VerusLogo from '../components/VerusLogo';
import { useAuth } from '../../context/AuthContext';

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
    id: 'impact-echo',
    name: 'Impact-Echo',
    fullName: 'Impact-Echo Acoustic Testing',
    status: 'in-development',
    icon: Waves,
    standard: 'ASTM C1383',
    description: 'Detects voids, delamination, and thickness anomalies from stress wave reflections.',
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
  {
    id: 'sounding',
    name: 'Sounding',
    fullName: 'Rapid Automated Sounding',
    status: 'in-development',
    icon: Speaker,
    standard: 'ASTM D4580',
    description: 'Detects hollow or delaminated areas from acoustic response patterns in sounding signals.',
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
  const { auth, logout } = useAuth();
  const navigate = useNavigate();
  const [modalModule, setModalModule] = useState<InspectionModule | null>(null);

  const handleLogout = () => {
    logout();
    navigate('/', { replace: true });
  };

  const handleModuleClick = (module: InspectionModule) => {
    if (module.status === 'available') {
      navigate('/analyze');
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
            }}>
              <h2 style={{
                margin: 0, fontSize: 11, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.08em', color: '#7A7470',
              }}>
                Recent Projects
              </h2>
            </div>

            {/* Empty state */}
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
                  marginTop: 8,
                  padding: '10px 24px',
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
          </div>
        </section>
      </main>

      {/* Coming-soon modal */}
      {modalModule && (
        <ComingSoonModal module={modalModule} onClose={() => setModalModule(null)} />
      )}
    </div>
  );
}
