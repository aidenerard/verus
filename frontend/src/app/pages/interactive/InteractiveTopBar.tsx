import { Link, useNavigate } from 'react-router';
import { ArrowLeft, Download } from 'lucide-react';
import VerusLogo from '../../components/VerusLogo';
import { useInteractiveStore } from './state/useInteractiveStore';
import type { ViewMode } from './state/types';
import { ACCENT, ACCENT_SOFT, BORDER, PANEL, TEXT, TEXT2, TEXT3, TOP_BAR_HEIGHT } from './tokens';

interface Props {
  projectId:   string;
  projectName: string | undefined;
  backHref:    string;
}

const VIEW_MODES: { value: ViewMode; label: string }[] = [
  { value: 'top',     label: 'Top' },
  { value: 'three_d', label: '3D' },
  { value: 'fixed',   label: 'Fixed' },
];

export default function InteractiveTopBar({ projectId, projectName, backHref }: Props) {
  const navigate = useNavigate();
  const viewMode    = useInteractiveStore(s => s.viewMode);
  const setViewMode = useInteractiveStore(s => s.setViewMode);

  return (
    <header style={{
      height: TOP_BAR_HEIGHT, flexShrink: 0, background: PANEL,
      borderBottom: `1px solid ${BORDER}`, padding: '0 16px',
      display: 'flex', alignItems: 'center', gap: 14,
    }}>
      <button
        onClick={() => navigate(backHref)}
        style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, padding: '4px 8px', fontFamily: 'inherit' }}
      >
        <ArrowLeft size={14} /> Back to Results
      </button>

      <div style={{ width: 1, height: 20, background: BORDER }} />

      <Link to="/dashboard" style={{ display: 'flex', textDecoration: 'none' }}>
        <VerusLogo size={20} wordmarkColor={TEXT} />
      </Link>

      <div style={{ width: 1, height: 20, background: BORDER }} />

      <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, overflow: 'hidden' }}>
        <span style={{ color: TEXT3, fontWeight: 600 }}>Interactive</span>
        <span style={{ color: TEXT3 }}>·</span>
        <span style={{ color: TEXT, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {projectName ?? projectId}
        </span>
      </div>

      <ViewToggle value={viewMode} onChange={setViewMode} />

      <button
        onClick={() => alert('Export from interactive view — not implemented')}
        style={{ background: 'none', border: `1px solid ${BORDER}`, cursor: 'pointer', color: TEXT2, display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', padding: '6px 12px', fontFamily: 'inherit' }}
      >
        <Download size={12} /> Export
      </button>
    </header>
  );
}

function ViewToggle({ value, onChange }: { value: ViewMode; onChange: (v: ViewMode) => void }) {
  return (
    <div role="tablist" style={{ display: 'flex', border: `1px solid ${BORDER}`, padding: 2 }}>
      {VIEW_MODES.map(m => {
        const active = value === m.value;
        return (
          <button
            key={m.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(m.value)}
            style={{
              border: 'none', cursor: 'pointer',
              background: active ? ACCENT_SOFT : 'transparent',
              color: active ? ACCENT : TEXT2,
              padding: '6px 12px', fontSize: 11, fontWeight: 700,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              fontFamily: 'inherit',
            }}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
