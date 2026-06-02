import { useState, FormEvent, useEffect } from 'react';
import Navbar, { NAVBAR_HEIGHT } from '../components/Navbar';
import Footer from '../components/Footer';
import { useAuth } from '../../context/AuthContext';

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '10px 14px',
  border: '2px solid #E2DED9', background: '#FFFFFF',
  fontSize: 14, color: '#0A0A0A', outline: 'none',
  boxSizing: 'border-box', fontFamily: 'Inter, sans-serif',
};

const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 11, fontWeight: 700,
  textTransform: 'uppercase', letterSpacing: '0.06em',
  color: '#7A7470', marginBottom: 8,
};

export default function AccountPage() {
  const { user, auth, updateAccount } = useAuth();

  const [name, setName]         = useState('');
  const [company, setCompany]   = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm]   = useState('');
  const [saving, setSaving]     = useState(false);
  const [error, setError]       = useState('');
  const [success, setSuccess]   = useState('');

  useEffect(() => {
    document.title = 'Account — Verus Technologies';
    setName(auth.user?.name ?? '');
    setCompany(auth.user?.company ?? '');
    setEmail(user?.email ?? '');
  }, [user, auth.user?.name, auth.user?.company]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!name.trim())    { setError('Please enter your name.'); return; }
    if (!company.trim()) { setError('Company or agency name is required.'); return; }
    if (!email.includes('@')) { setError('Please enter a valid email address.'); return; }
    if (password && password.length < 6) { setError('Password must be at least 6 characters.'); return; }
    if (password && password !== confirm) { setError('Passwords do not match.'); return; }

    const fields: { name?: string; company?: string; email?: string; password?: string } = {
      name: name.trim(),
      company: company.trim(),
    };
    const emailChanged = email.trim() !== (user?.email ?? '');
    if (emailChanged) fields.email = email.trim();
    if (password) fields.password = password;

    setSaving(true);
    const { error: updateError } = await updateAccount(fields);
    setSaving(false);

    if (updateError) { setError(updateError); return; }

    setPassword('');
    setConfirm('');
    setSuccess(
      emailChanged
        ? 'Saved. Check your new email address for a confirmation link to finish the change.'
        : 'Your account details have been updated.',
    );
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
        alignItems: 'center', padding: `${NAVBAR_HEIGHT + 48}px 24px 56px`,
      }}>
        <div style={{ width: '100%', maxWidth: 520, background: '#FFFFFF', border: '2px solid #E2DED9' }}>
          <div style={{ padding: '18px 28px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF' }}>
            <h1 style={{
              margin: 0, fontSize: 14, fontWeight: 700,
              textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A',
            }}>
              Account Settings
            </h1>
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#7A7470' }}>
              Update your profile and sign-in details.
            </p>
          </div>

          <form onSubmit={handleSubmit} style={{ padding: '28px' }}>
            {error && (
              <div style={{ padding: '10px 14px', marginBottom: 20, background: '#FFF5F5', border: '1.5px solid #E74C3C', fontSize: 12, color: '#742A2A' }}>
                {error}
              </div>
            )}
            {success && (
              <div style={{ padding: '10px 14px', marginBottom: 20, background: '#F0FAF3', border: '1.5px solid #2ECC71', fontSize: 12, color: '#1E5E36' }}>
                {success}
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
              <div>
                <label style={labelStyle}>Full Name</label>
                <input type="text" value={name} onChange={e => setName(e.target.value)} autoComplete="name" style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Company / Agency</label>
                <input type="text" value={company} onChange={e => setCompany(e.target.value)} autoComplete="organization" maxLength={60} style={inputStyle} />
              </div>
            </div>

            <div style={{ marginBottom: 24 }}>
              <label style={labelStyle}>Email Address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} autoComplete="email" style={inputStyle} />
            </div>

            <div style={{ borderTop: '1px solid #E2DED9', paddingTop: 22, marginBottom: 20 }}>
              <p style={{ margin: '0 0 16px', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#0A0A0A' }}>
                Change Password
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <label style={labelStyle}>New Password</label>
                  <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Leave blank to keep" autoComplete="new-password" style={inputStyle} />
                </div>
                <div>
                  <label style={labelStyle}>Confirm Password</label>
                  <input type="password" value={confirm} onChange={e => setConfirm(e.target.value)} placeholder="Re-enter new password" autoComplete="new-password" style={inputStyle} />
                </div>
              </div>
            </div>

            <button type="submit" disabled={saving}
              style={{
                width: '100%', padding: '12px', background: saving ? '#B0A9A4' : '#E8601C',
                color: '#FFFFFF', border: 'none', fontWeight: 700, fontSize: 12,
                letterSpacing: '0.08em', textTransform: 'uppercase',
                cursor: saving ? 'default' : 'pointer', fontFamily: 'Inter, sans-serif',
              }}>
              {saving ? 'Saving…' : 'Save Changes'}
            </button>
          </form>
        </div>
      </div>
      <Footer />
    </div>
  );
}
