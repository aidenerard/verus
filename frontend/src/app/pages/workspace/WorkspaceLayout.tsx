import { useEffect, useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router';
import { ArrowLeft, Menu } from 'lucide-react';
import { useAuth } from '../../../context/AuthContext';
import { ProcessingOptionsProvider } from './ProcessingOptionsContext';
import WorkspaceSidebar from './WorkspaceSidebar';
import {
  BG, BORDER, PANEL, TEXT, TEXT2, ACCENT,
  SIDEBAR_WIDTH, TOPBAR_HEIGHT, FONT_FAMILY,
} from './tokens';
import { findMethod } from './modules';

export default function WorkspaceLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { auth } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);

  useEffect(() => {
    if (window.innerWidth < 768) setSidebarOpen(false);
  }, [location.pathname]);

  const current = findMethod(location.pathname);
  const initials = auth.user?.name
    ? auth.user.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <ProcessingOptionsProvider>
      <div
        className="workspace-shell"
        data-sidebar-open={sidebarOpen}
        style={{
          minHeight: '100vh', background: BG, color: TEXT, fontFamily: FONT_FAMILY,
          display: 'grid', gridTemplateColumns: `${SIDEBAR_WIDTH}px 1fr`,
        }}
      >
        <div className="workspace-sidebar-host" style={{ minWidth: 0 }}>
          <WorkspaceSidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <header style={{
            height: TOPBAR_HEIGHT, flexShrink: 0, background: PANEL,
            borderBottom: `1px solid ${BORDER}`, padding: '0 20px',
            display: 'flex', alignItems: 'center', gap: 14,
          }}>
            <button
              className="workspace-menu-btn"
              onClick={() => setSidebarOpen(v => !v)}
              aria-label="Toggle sidebar"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 6, display: 'none' }}
            >
              <Menu size={18} />
            </button>

            <button
              onClick={() => navigate('/dashboard')}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, padding: '4px 8px' }}
            >
              <ArrowLeft size={14} /> Dashboard
            </button>

            <div style={{ width: 1, height: 20, background: BORDER }} />

            <div style={{ flex: 1, minWidth: 0, fontSize: 13, fontWeight: 600, color: TEXT, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {current ? `${current.module.label} · ${current.method.fullName}` : 'Workspace'}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 28, height: 28, background: ACCENT, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>
                {initials}
              </div>
            </div>
          </header>

          <main style={{ flex: 1, minWidth: 0, overflowY: 'auto' }}>
            <Outlet />
          </main>
        </div>

        {sidebarOpen && (
          <div
            className="workspace-scrim"
            onClick={() => setSidebarOpen(false)}
            style={{ display: 'none' }}
          />
        )}

        <style>{`
          .workspace-shell .workspace-sidebar-host { grid-column: 1; grid-row: 1; }
          @media (max-width: 767px) {
            .workspace-shell { grid-template-columns: 1fr !important; }
            .workspace-shell .workspace-sidebar-host {
              position: fixed; inset: 0 auto 0 0; width: 280px; z-index: 60;
              transform: translateX(-100%); transition: transform 0.2s ease;
              box-shadow: 0 0 24px rgba(0,0,0,0.18);
            }
            .workspace-shell[data-sidebar-open="true"] .workspace-sidebar-host { transform: translateX(0); }
            .workspace-shell .workspace-menu-btn { display: flex !important; }
            .workspace-shell .workspace-scrim {
              display: block !important; position: fixed; inset: 0; z-index: 55;
              background: rgba(10,10,10,0.45);
            }
          }
          @keyframes spin { to { transform: rotate(360deg); } }
        `}</style>
      </div>
    </ProcessingOptionsProvider>
  );
}
