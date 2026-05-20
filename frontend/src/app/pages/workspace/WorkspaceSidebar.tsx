import { NavLink } from 'react-router';
import { X } from 'lucide-react';
import VerusLogo from '../../components/VerusLogo';
import { MODULES } from './modules';
import { ACCENT, ACCENT_SOFT, BORDER, RAISED, SIDEBAR_BG, TEXT, TEXT2, TEXT3, FONT_FAMILY } from './tokens';
import ProcessingOptionsPanel from './ProcessingOptionsPanel';

interface Props {
  open:    boolean;
  onClose: () => void;
}

export default function WorkspaceSidebar({ open, onClose }: Props) {
  return (
    <aside
      data-open={open}
      style={{
        background: SIDEBAR_BG,
        borderRight: `1px solid ${BORDER}`,
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        overflow: 'hidden',
        fontFamily: FONT_FAMILY,
      }}
    >
      <div style={{ padding: '16px 20px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <NavLink to="/dashboard" style={{ textDecoration: 'none', display: 'flex' }}>
          <VerusLogo size={26} wordmarkColor={TEXT} />
        </NavLink>
        <button
          onClick={onClose}
          aria-label="Close sidebar"
          className="workspace-sidebar-close"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'none', padding: 4 }}
        >
          <X size={18} />
        </button>
      </div>

      <nav style={{ padding: '12px 0', flex: '1 1 auto', overflowY: 'auto' }}>
        {MODULES.map(mod => (
          <div key={mod.id} style={{ marginBottom: 18 }}>
            <div style={{ padding: '0 20px 8px', fontSize: 11, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3 }}>
              {mod.label}
            </div>
            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {mod.methods.map(m => {
                const Icon = m.Icon;
                const comingSoon = m.status === 'coming-soon';
                return (
                  <li key={m.id}>
                    <NavLink
                      to={m.path}
                      onClick={onClose}
                      style={({ isActive }) => ({
                        textDecoration: 'none',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                        padding: '9px 20px 9px 28px',
                        fontSize: 13,
                        fontWeight: isActive ? 700 : 500,
                        color: isActive ? ACCENT : (comingSoon ? TEXT3 : TEXT2),
                        background: isActive ? ACCENT_SOFT : 'transparent',
                        borderLeft: isActive ? `3px solid ${ACCENT}` : '3px solid transparent',
                        transition: 'background 0.15s, color 0.15s',
                      })}
                    >
                      <Icon size={14} />
                      <span style={{ flex: 1 }}>{m.name}</span>
                      {comingSoon && (
                        <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', padding: '2px 6px', background: RAISED, color: TEXT3, border: `1px solid ${BORDER}` }}>
                          Soon
                        </span>
                      )}
                    </NavLink>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}

        <div style={{ borderTop: `1px solid ${BORDER}`, margin: '12px 20px 0', paddingTop: 12 }}>
          <ProcessingOptionsPanel />
        </div>
      </nav>

      <style>{`
        @media (max-width: 767px) {
          .workspace-sidebar-close { display: flex !important; }
        }
      `}</style>
    </aside>
  );
}
