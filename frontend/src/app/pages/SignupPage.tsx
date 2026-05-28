import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import Navbar, { NAVBAR_HEIGHT } from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../../context/AuthContext';

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();

  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [company, setCompany]   = useState('');
  const [agreed, setAgreed]     = useState(false);
  const [error, setError]       = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name.trim())         { setError('Please enter your name.'); return; }
    if (!company.trim())      { setError('Company or agency name is required.'); return; }
    if (!email.includes('@')) { setError('Please enter a valid email address.'); return; }
    if (password.length < 6)  { setError('Password must be at least 6 characters.'); return; }
    if (!agreed)              { setError('Please agree to the Terms of Use and Privacy Policy.'); return; }

    const { error: authError } = await signup(email, password, name.trim(), company.trim());
    if (authError) {
      setError(authError);
      return;
    }
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
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      <Navbar />
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: `${NAVBAR_HEIGHT + 40}px 24px 40px`,
      }}>

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
              <label style={labelStyle}>
                Company / Agency <span style={{ color: '#E8601C' }}>*</span>
              </label>
              <input
                type="text"
                value={company}
                onChange={e => setCompany(e.target.value)}
                placeholder="Caltrans, Terracon, WISDOT"
                autoComplete="organization"
                required
                maxLength={60}
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
              <Link to="/terms" style={{ color: '#E8601C', textDecoration: 'none', fontWeight: 600 }}>
                Terms of Use
              </Link>
              {' '}and{' '}
              <Link to="/privacy" style={{ color: '#E8601C', textDecoration: 'none', fontWeight: 600 }}>
                Privacy Policy
              </Link>.
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

          <p style={{ fontSize: 12, color: '#9ca3af', textAlign: 'center', marginTop: 16 }}>
            By creating an account, you agree to our{' '}
            <Link to="/terms" style={{ color: '#E8601C' }}>Terms of Use</Link>
            {' '}and{' '}
            <Link to="/privacy" style={{ color: '#E8601C' }}>Privacy Policy</Link>.
          </p>

          <p style={{ marginTop: 24, textAlign: 'center', fontSize: 13, color: '#7A7470' }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#E8601C', fontWeight: 600, textDecoration: 'none' }}>
              Log In
            </Link>
          </p>
        </form>
      </div>
      </div>
      <Footer />
    </div>
  );
}
