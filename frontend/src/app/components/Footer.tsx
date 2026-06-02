import { Link } from 'react-router';

const BG          = '#0A0A0A';
const BORDER      = 'rgba(255,255,255,0.08)';
const LINK        = 'rgba(240,237,232,0.68)';
const LINK_HOVER  = '#F0EDE8';
const COPYRIGHT   = 'rgba(240,237,232,0.45)';

export default function Footer() {
  return (
    <footer style={{
      background: BG,
      borderTop: `1px solid ${BORDER}`,
      padding: '20px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      fontSize: 12,
      color: COPYRIGHT,
      flexWrap: 'wrap',
      gap: 12,
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      <span>© {new Date().getFullYear()} Verus Technologies, Inc. All rights reserved.</span>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'center' }}>
        <FooterLink to="/">Home</FooterLink>
        <FooterLink to="/experience">Experience</FooterLink>
        <FooterLink to="/team">Team</FooterLink>
        <FooterLink to="/login">Log In</FooterLink>
        <FooterLink to="/signup">Sign Up</FooterLink>
        <FooterLink to="/privacy">Privacy</FooterLink>
        <FooterLink to="/terms">Terms</FooterLink>
        <FooterMail />
      </div>
    </footer>
  );
}

function FooterLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      style={{ color: LINK, textDecoration: 'none' }}
      onMouseEnter={e => (e.currentTarget.style.color = LINK_HOVER)}
      onMouseLeave={e => (e.currentTarget.style.color = LINK)}
    >
      {children}
    </Link>
  );
}

function FooterMail() {
  return (
    <a
      href="mailto:info@verus.com"
      style={{ color: LINK, textDecoration: 'none' }}
      onMouseEnter={e => (e.currentTarget.style.color = LINK_HOVER)}
      onMouseLeave={e => (e.currentTarget.style.color = LINK)}
    >
      info@verus.com
    </a>
  );
}
