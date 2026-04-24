import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router';
import VerusLogo from '../components/VerusLogo';
import { useAuth } from '../../context/AuthContext';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!email.trim() || !password.trim()) {
      setError('Please enter your email and password.');
      return;
    }
    if (!email.includes('@')) {
      setError('Please enter a valid email address.');
      return;
    }

    // No real auth yet — accept any credentials
    login(email, email.split('@')[0]);
    navigate('/dashboard', { replace: true });
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
      <div style={{
        width: '100%', maxWidth: 420,
        background: '#FFFFFF', border: '2px solid #E2DED9',
      }}>
        {/* Card header */}
        <div style={{
          padding: '18px 28px', borderBottom: '2px solid #E2DED9',
          background: '#F5F3EF',
        }}>
          <h1 style={{
            margin: 0, fontSize: 14, fontWeight: 700,
            textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A',
          }}>
            Log In to Verus
          </h1>
        </div>

        {/* Card body */}
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

          <div style={{ marginBottom: 20 }}>
            <label style={{
              display: 'block', fontSize: 11, fontWeight: 700,
              textTransform: 'uppercase', letterSpacing: '0.06em',
              color: '#7A7470', marginBottom: 8,
            }}>
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@company.com"
              autoComplete="email"
              style={{
                width: '100%', padding: '10px 14px',
                border: '2px solid #E2DED9', background: '#FFFFFF',
                fontSize: 14, color: '#0A0A0A', outline: 'none',
                boxSizing: 'border-box',
                fontFamily: 'Inter, sans-serif',
              }}
            />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label style={{
              display: 'block', fontSize: 11, fontWeight: 700,
              textTransform: 'uppercase', letterSpacing: '0.06em',
              color: '#7A7470', marginBottom: 8,
            }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete="current-password"
              style={{
                width: '100%', padding: '10px 14px',
                border: '2px solid #E2DED9', background: '#FFFFFF',
                fontSize: 14, color: '#0A0A0A', outline: 'none',
                boxSizing: 'border-box',
                fontFamily: 'Inter, sans-serif',
              }}
            />
          </div>

          <div style={{ textAlign: 'right', marginBottom: 24 }}>
            <a href="#" style={{ fontSize: 12, color: '#7A7470', textDecoration: 'none' }}>
              Forgot password?
            </a>
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
            Log In
          </button>

          <p style={{
            marginTop: 24, textAlign: 'center',
            fontSize: 13, color: '#7A7470',
          }}>
            Don't have an account?{' '}
            <Link to="/signup" style={{ color: '#E8601C', fontWeight: 600, textDecoration: 'none' }}>
              Sign Up
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
