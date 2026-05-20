import { Link, Outlet, useLocation, useNavigate } from 'react-router';
import { ArrowLeft, ChevronRight } from 'lucide-react';
import VerusLogo from '../../components/VerusLogo';
import { useAuth } from '../../../context/AuthContext';
import { ProcessingOptionsProvider } from './ProcessingOptionsContext';
import { BG, BORDER, PANEL, TEXT, TEXT2, TEXT3, ACCENT, TOPBAR_HEIGHT, FONT_FAMILY } from './tokens';
import { findMethod } from './modules';

export default function WorkspaceLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { auth } = useAuth();

  const current = findMethod(location.pathname);
  const moduleHref = current ? `/workspace/${current.module.id}` : '/workspace';
  const initials = auth.user?.name
    ? auth.user.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <ProcessingOptionsProvider>
      <div style={{
        minHeight: '100vh', background: BG, color: TEXT, fontFamily: FONT_FAMILY,
        display: 'flex', flexDirection: 'column',
      }}>
        <header style={{
          height: TOPBAR_HEIGHT, flexShrink: 0, background: PANEL,
          borderBottom: `1px solid ${BORDER}`, padding: '0 24px',
          display: 'flex', alignItems: 'center', gap: 16,
        }}>
          <button
            onClick={() => navigate(moduleHref)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, padding: '4px 8px', fontFamily: 'inherit' }}
          >
            <ArrowLeft size={14} /> Back
          </button>

          <div style={{ width: 1, height: 20, background: BORDER }} />

          <Link to="/workspace" style={{ display: 'flex', textDecoration: 'none' }}>
            <VerusLogo size={22} wordmarkColor={TEXT} />
          </Link>

          <div style={{ width: 1, height: 20, background: BORDER }} />

          <nav aria-label="Breadcrumb" style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, overflow: 'hidden' }}>
            <Crumb to="/workspace" label="Workspace" />
            {current && (
              <>
                <ChevronRight size={12} style={{ color: TEXT3, flexShrink: 0 }} />
                <Crumb to={moduleHref} label={current.module.label} />
                <ChevronRight size={12} style={{ color: TEXT3, flexShrink: 0 }} />
                <span style={{ color: ACCENT, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {current.method.name}
                </span>
              </>
            )}
          </nav>

          <div style={{ width: 28, height: 28, background: ACCENT, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#fff' }}>
            {initials}
          </div>
        </header>

        <main style={{ flex: 1, minWidth: 0, overflowY: 'auto' }}>
          <Outlet />
        </main>

        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
        `}</style>
      </div>
    </ProcessingOptionsProvider>
  );
}

function Crumb({ to, label }: { to: string; label: string }) {
  return (
    <Link
      to={to}
      style={{
        color: TEXT2, textDecoration: 'none', fontWeight: 600,
        padding: '2px 4px', borderRadius: 2,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0,
      }}
      onMouseEnter={e => (e.currentTarget.style.color = TEXT)}
      onMouseLeave={e => (e.currentTarget.style.color = TEXT2)}
    >
      {label}
    </Link>
  );
}
