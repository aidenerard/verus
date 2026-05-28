import { Link } from 'react-router';
import VerusLogo from '../../components/VerusLogo';
import { C } from './tokens';

export default function Footer() {
  return (
    <footer style={{ background: C.black, borderTop: `1px solid rgba(255,255,255,0.07)`, padding: '24px 48px' }}>
      <div style={{ maxWidth: 1080, margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24, flexWrap: 'wrap' }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <VerusLogo size={26} wordmarkColor="rgba(240,237,232,0.8)" />
        </Link>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
          <Link to="/" className="vr-footer-link">Home</Link>
          <Link to="/team" className="vr-footer-link">Team</Link>
          <Link to="/login" className="vr-footer-link">Log In</Link>
          <Link to="/signup" className="vr-footer-link">Sign Up</Link>
          <Link to="/privacy" className="vr-footer-link">Privacy</Link>
          <Link to="/terms" className="vr-footer-link">Terms</Link>
          <a href="mailto:hello@verus.ai" className="vr-footer-link">hello@verus.ai</a>
        </div>
        <p style={{ fontSize: 11, color: 'rgba(240,237,232,0.55)', margin: 0 }}>
          © {new Date().getFullYear()} Verus Technologies, Inc.
        </p>
      </div>
    </footer>
  );
}
