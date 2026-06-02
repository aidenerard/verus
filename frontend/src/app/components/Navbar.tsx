import { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router';
import { ChevronDown } from 'lucide-react';
import VerusLogo from './VerusLogo';
import { useAuth } from '../../context/AuthContext';

export const NAVBAR_HEIGHT = 58;

const NAV_CSS = `
  .vr-nav-link {
    position: relative; text-decoration: none;
    color: rgba(240,237,232,0.68); font-size: 13px;
    font-weight: 500; letter-spacing: 0.01em; transition: color 0.2s;
  }
  .vr-nav-link::after {
    content:''; position:absolute; bottom:-3px; left:0;
    width:0; height:1px; background:#E8601C; transition: width 0.25s ease;
  }
  .vr-nav-link:hover { color:#F0EDE8; }
  .vr-nav-link:hover::after { width:100%; }
  .vr-nav-login {
    font-size:12px; font-weight:600; color:rgba(240,237,232,0.68);
    text-decoration:none; letter-spacing:0.03em; transition:color 0.2s;
  }
  .vr-nav-login:hover { color:#F0EDE8; }
  .vr-nav-signup {
    display:inline-block; padding:8px 20px; background:#E8601C; color:#fff;
    font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:none; transition: background 0.18s, transform 0.18s;
  }
  .vr-nav-signup:hover { background:#D4521A; transform:translateY(-1px); }

  .vr-profile { position: relative; }
  .vr-profile-btn {
    display:flex; align-items:center; gap:7px; background:none; border:none;
    cursor:pointer; padding:0; font-family:inherit;
  }
  .vr-avatar {
    width:32px; height:32px; background:#E8601C; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:700; color:#fff; letter-spacing:0.04em;
  }
  .vr-chev { color:rgba(240,237,232,0.55); transition: transform 0.2s, color 0.2s; }
  .vr-profile-btn:hover .vr-chev { color:#F0EDE8; }
  .vr-chev.open { transform: rotate(180deg); }
  .vr-menu {
    position:absolute; right:0; top:calc(100% + 14px); min-width:212px;
    background:#141210; border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 14px 36px rgba(0,0,0,0.45); padding:6px; z-index:120;
  }
  .vr-menu-head {
    padding:10px 12px 12px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:6px;
  }
  .vr-menu-name { margin:0; font-size:13px; font-weight:600; color:#F0EDE8; }
  .vr-menu-email { margin:2px 0 0; font-size:11px; color:rgba(240,237,232,0.5); overflow:hidden; text-overflow:ellipsis; }
  .vr-menu-item {
    display:block; width:100%; box-sizing:border-box; text-align:left; padding:9px 12px;
    background:none; border:none; cursor:pointer; font-family:inherit;
    font-size:13px; font-weight:500; color:rgba(240,237,232,0.75);
    text-decoration:none; letter-spacing:0.01em; transition: background 0.15s, color 0.15s;
  }
  .vr-menu-item:hover { background:rgba(255,255,255,0.05); color:#F0EDE8; }
  .vr-menu-logout {
    color:rgba(240,237,232,0.6); border-top:1px solid rgba(255,255,255,0.08); margin-top:6px;
  }
  .vr-menu-logout:hover { color:#E8601C; background:rgba(232,96,28,0.06); }
`;

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const { isAuthenticated, auth, logout } = useAuth();
  const navigate = useNavigate();
  const profileRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', h, { passive: true });
    return () => window.removeEventListener('scroll', h);
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const h = (e: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [menuOpen]);

  const initials = auth.user?.name
    ? auth.user.name.split(' ').map((w: string) => w[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <>
      <style>{NAV_CSS}</style>
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100, height: NAVBAR_HEIGHT,
        background: '#0A0A0A', display: 'flex', alignItems: 'center', padding: '0 48px',
        borderBottom: `1px solid ${scrolled ? 'rgba(255,255,255,0.08)' : 'transparent'}`,
        boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.5)' : 'none',
        transition: 'border-color 0.3s, box-shadow 0.3s',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      }}>
        <Link to="/" style={{ textDecoration: 'none', flexShrink: 0 }}>
          <VerusLogo size={30} wordmarkColor="#F0EDE8" />
        </Link>
        <div style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: 32 }}>
          <Link to="/" className="vr-nav-link">Home</Link>
          <Link to="/experience" className="vr-nav-link">Experience</Link>
          <Link to="/team" className="vr-nav-link">Team</Link>
          {isAuthenticated && <Link to="/dashboard" className="vr-nav-link">Dashboard</Link>}
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          {isAuthenticated ? (
            <div className="vr-profile" ref={profileRef}>
              <button className="vr-profile-btn" onClick={() => setMenuOpen(o => !o)} aria-label="Account menu">
                <div className="vr-avatar" title={auth.user?.name || auth.user?.email}>{initials}</div>
                <ChevronDown size={14} className={`vr-chev${menuOpen ? ' open' : ''}`} />
              </button>
              {menuOpen && (
                <div className="vr-menu">
                  <div className="vr-menu-head">
                    <p className="vr-menu-name">{auth.user?.name}</p>
                    <p className="vr-menu-email">{auth.user?.email}</p>
                  </div>
                  <Link to="/account" className="vr-menu-item" onClick={() => setMenuOpen(false)}>Account</Link>
                  <button
                    className="vr-menu-item vr-menu-logout"
                    onClick={() => { setMenuOpen(false); logout(); navigate('/', { replace: true }); }}
                  >
                    Log Out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <>
              <Link to="/login" className="vr-nav-login">Log In</Link>
              <Link to="/signup" className="vr-nav-signup">Sign Up</Link>
            </>
          )}
        </div>
      </nav>
    </>
  );
}
