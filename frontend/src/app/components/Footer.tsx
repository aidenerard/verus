import { Link } from 'react-router';

const ACCENT     = '#E8601C';
const MUTED      = '#6b7280';
const MUTED_DIM  = '#9ca3af';
const BORDER     = '#e5e7eb';

export default function Footer() {
  return (
    <footer style={{
      borderTop: `1px solid ${BORDER}`,
      padding: '20px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      fontSize: 12,
      color: MUTED_DIM,
      flexWrap: 'wrap',
      gap: 12,
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      <span>© {new Date().getFullYear()} Verus Technologies, Inc. All rights reserved.</span>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'center' }}>
        <FooterLink to="/">Home</FooterLink>
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
      style={{ color: MUTED, textDecoration: 'none' }}
      onMouseEnter={e => (e.currentTarget.style.color = ACCENT)}
      onMouseLeave={e => (e.currentTarget.style.color = MUTED)}
    >
      {children}
    </Link>
  );
}

function FooterMail() {
  return (
    <a
      href="mailto:info@verus.com"
      style={{ color: MUTED, textDecoration: 'none' }}
      onMouseEnter={e => (e.currentTarget.style.color = ACCENT)}
      onMouseLeave={e => (e.currentTarget.style.color = MUTED)}
    >
      info@verus.com
    </a>
  );
}
