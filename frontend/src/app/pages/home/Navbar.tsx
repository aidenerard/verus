import { useState, useEffect } from 'react';
import { Link } from 'react-router';
import VerusLogo from '../../components/VerusLogo';
import { C } from './tokens';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', h, { passive: true });
    return () => window.removeEventListener('scroll', h);
  }, []);

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100, height: 58,
      background: C.black, display: 'flex', alignItems: 'center', padding: '0 48px',
      borderBottom: `1px solid ${scrolled ? 'rgba(255,255,255,0.08)' : 'transparent'}`,
      boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.5)' : 'none',
      transition: 'border-color 0.3s, box-shadow 0.3s',
    }}>
      <Link to="/" style={{ textDecoration: 'none', flexShrink: 0 }}>
        <VerusLogo size={30} wordmarkColor="#F0EDE8" />
      </Link>
      <div style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: 32 }}>
        <Link to="/" className="vr-nav-link">Home</Link>
        <Link to="/team" className="vr-nav-link">Team</Link>
      </div>
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 22, flexShrink: 0 }}>
        <Link to="/login" className="vr-nav-login">Log In</Link>
        <Link to="/signup" className="vr-nav-signup">Sign Up</Link>
      </div>
    </nav>
  );
}
