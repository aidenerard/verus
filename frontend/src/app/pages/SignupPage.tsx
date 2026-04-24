import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import VerusLogo from '../components/VerusLogo';
import { useAuth } from '../../context/AuthContext';

export default function SignupPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [company, setCompany]   = useState('');
  const [agreed, setAgreed]     = useState(false);
  const [error, setError]       = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name.trim())            { setError('Please enter your name.'); return; }
    if (!email.includes('@'))    { setError('Please enter a valid email address.'); return; }
    if (password.length < 6)     { setError('Password must be at least 6 characters.'); return; }
    if (!agreed)                 { setError('Please agree to the Terms of Service.'); return; }

    // No real auth yet — accept any credentials
    login(email, name.trim());
    navigate('/dashboard', { replace: true });
  };

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '10px 14px',
    border: '2px solid #E2DED9', background: '#FFFFFF',
    fontSize: 14, color: '#0A0A0A', outline: 'none',
    boxSizing: 'border-box',
    fontFamily: 'Inter, sans-serif',
  };

  const labelStyle: React.CSSProperties = {
    display: 'block', fontSize: 11, fontWeight: 700,
    textTransform: 'uppercase', letterSpacing: '0.06em',
    color: '#7A7470', marginBottom: 8,
  };

  return (
    <div style={{
      minHeight: '100vh', background: '#F5F3EF',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '40px 24px',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      {/* Logo */}
      <Link to="/" style={{ textDecoration: 'none', marginBottom: 40 }}>
        <VerusLogo size={40} wordmarkColor="#0A0A0A" />
      </Link>

      {/* Card */}
      <div style={{ width: '100%', maxWidth: 460, background: '#FFFFFF', border: '2px solid #E2DED9' }}>
        {/* Header */}
        <div style={{ padding: '18px 28px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF' }}>
          <h1 style={{
            margin: 0, fontSize: 14, fontWeight: 700,
            textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A',
          }}>
            Create Your Account
          </h1>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} style={{ padding: '28px' }}>
          {error && (
            <div style={{
              padding: '10px 14px', marginBottom: 20,
              background: '#FFF5F5', border: '1.5px solid #E74C3C',
              fontSize: 12, color: '#742A2A',
            }}>
              {error}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div>
              <label style={labelStyle}>Full Name</label>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="Jane Smith"
                autoComplete="name"
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Company <span style={{ color: '#B0A9A4', fontWeight: 400 }}>(optional)</span></label>
              <input
                type="text"
                value={company}
                onChange={e => setCompany(e.target.value)}
                placeholder="ACME Engineering"
                autoComplete="organization"
                style={inputStyle}
              />
            </div>
          </div>

          <div style={{ marginBottom: 20 }}>
            <label style={labelStyle}>Email Address</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@company.com"
              autoComplete="email"
              style={inputStyle}
            />
          </div>

          <div style={{ marginBottom: 24 }}>
            <label style={labelStyle}>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="At least 6 characters"
              autoComplete="new-password"
              style={inputStyle}
            />
          </div>

          {/* Terms checkbox */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 28 }}>
            <input
              type="checkbox"
              id="terms"
              checked={agreed}
              onChange={e => setAgreed(e.target.checked)}
              style={{ marginTop: 2, accentColor: '#E8601C', cursor: 'pointer', flexShrink: 0 }}
            />
            <label htmlFor="terms" style={{
              fontSize: 12, color: '#7A7470', cursor: 'pointer', lineHeight: 1.5,
            }}>
              I agree to the{' '}
              <a href="#" style={{ color: '#E8601C', textDecoration: 'none', fontWeight: 600 }}>
                Terms of Service
              </a>
            </label>
          </div>

          <button
            type="submit"
            style={{
              width: '100%', padding: '13px',
              background: '#E8601C', color: '#FFFFFF',
              border: '2px solid #E8601C',
              fontWeight: 700, fontSize: 12,
              letterSpacing: '0.08em', textTransform: 'uppercase',
              cursor: 'pointer',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            Create Account
          </button>

          <p style={{ marginTop: 24, textAlign: 'center', fontSize: 13, color: '#7A7470' }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#E8601C', fontWeight: 600, textDecoration: 'none' }}>
              Log In
            </Link>
          </p>
        </form>
      </div>

      <p style={{ marginTop: 32, fontSize: 11, color: '#B0A9A4', textAlign: 'center' }}>
        © {new Date().getFullYear()} Verus · ASTM D6087 Compliant
      </p>
    </div>
  );
}
