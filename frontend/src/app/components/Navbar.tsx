import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
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
`;

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const { isAuthenticated, auth, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', h, { passive: true });
    return () => window.removeEventListener('scroll', h);
  }, []);

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
          <Link to="/team" className="vr-nav-link">Team</Link>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16, flexShrink: 0 }}>
          {isAuthenticated ? (
            <>
              <Link to="/dashboard" className="vr-nav-login">Dashboard</Link>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div
                  style={{ width: 32, height: 32, background: '#E8601C', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: '#FFFFFF', letterSpacing: '0.04em', cursor: 'pointer' }}
                  title={auth.user?.name || auth.user?.email}
                >
                  {initials}
                </div>
                <button
                  onClick={() => { logout(); navigate('/', { replace: true }); }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.55)', letterSpacing: '0.04em', padding: 0, fontFamily: 'Inter, sans-serif' }}
                >
                  Log Out
                </button>
              </div>
            </>
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
