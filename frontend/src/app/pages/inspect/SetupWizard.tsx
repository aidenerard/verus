import {
  MANUFACTURERS, FREQ_OPTIONS,
  PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2, ACCENT,
} from './constants';
import type { ManufacturerKey } from './constants';
import VerusLogo from '../../components/VerusLogo';

interface Props {
  setupStep: 1 | 2 | 3;
  setSetupStep: (s: 1 | 2 | 3) => void;
  manufacturer: ManufacturerKey | '';
  setManufacturer: (m: ManufacturerKey) => void;
  frequencyMhz: number;
  setFrequencyMhz: (v: number) => void;
  customFreq: string;
  setCustomFreq: (v: string) => void;
  useCustomFreq: boolean;
  setUseCustomFreq: (v: boolean) => void;
  structureName: string;
  setStructureName: (v: string) => void;
  projectName: string;
  setProjectName: (v: string) => void;
  bridgeId: string;
  setBridgeId: (v: string) => void;
  inspDate: string;
  setInspDate: (v: string) => void;
  notes: string;
  setNotes: (v: string) => void;
  onComplete: () => void;
}

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '9px 12px', background: RAISED,
  border: `1px solid ${BORDER2}`, color: TEXT, fontSize: 12,
  fontFamily: 'Inter, sans-serif', outline: 'none', boxSizing: 'border-box',
};

const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 10, fontWeight: 700,
  letterSpacing: '0.07em', textTransform: 'uppercase', color: TEXT2, marginBottom: 6,
};

const nextBtnStyle = (enabled: boolean): React.CSSProperties => ({
  padding: '9px 24px', background: enabled ? ACCENT : BORDER, border: 'none',
  color: enabled ? '#fff' : TEXT2, fontSize: 11, fontWeight: 700,
  letterSpacing: '0.07em', textTransform: 'uppercase',
  cursor: enabled ? 'pointer' : 'default', fontFamily: 'Inter, sans-serif',
});

const backBtnStyle: React.CSSProperties = {
  padding: '9px 20px', background: 'none', border: `1px solid ${BORDER}`,
  color: TEXT2, fontSize: 11, fontWeight: 700, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
};

export default function SetupWizard({
  setupStep, setSetupStep,
  manufacturer, setManufacturer,
  frequencyMhz, setFrequencyMhz,
  customFreq, setCustomFreq,
  useCustomFreq, setUseCustomFreq,
  structureName, setStructureName,
  projectName, setProjectName,
  bridgeId, setBridgeId,
  inspDate, setInspDate,
  notes, setNotes,
  onComplete,
}: Props) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(10,10,10,0.5)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        width: 560, background: PANEL, border: `1px solid ${BORDER2}`,
        boxShadow: '0 24px 80px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column',
      }}>

        {/* Header */}
        <div style={{ padding: '20px 24px 16px', borderBottom: `1px solid ${BORDER}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <VerusLogo size={20} wordmarkColor="#0A0A0A" />
            <span style={{ fontSize: 12, color: TEXT2 }}>New Inspection Setup</span>
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            {[1, 2, 3].map(s => (
              <div key={s} style={{ height: 3, flex: 1, borderRadius: 2, background: s <= setupStep ? ACCENT : BORDER }} />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
            {['Equipment', 'Antenna', 'Project'].map((label, i) => (
              <span key={label} style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase',
                color: i + 1 === setupStep ? ACCENT : TEXT2,
              }}>{label}</span>
            ))}
          </div>
        </div>

        {/* Step 1: Manufacturer */}
        {setupStep === 1 && (
          <div style={{ padding: 24 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: TEXT, marginBottom: 4 }}>Select Equipment Manufacturer</p>
            <p style={{ fontSize: 11, color: TEXT2, marginBottom: 16 }}>
              This sets which file formats are accepted and which converter is used.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {MANUFACTURERS.map(m => (
                <button key={m.key} onClick={() => setManufacturer(m.key as ManufacturerKey)}
                  style={{
                    padding: '12px 14px', textAlign: 'left',
                    background: manufacturer === m.key ? 'rgba(232,96,28,0.08)' : RAISED,
                    border: `1.5px solid ${manufacturer === m.key ? ACCENT : BORDER}`,
                    cursor: 'pointer', transition: 'border 0.12s, background 0.12s',
                  }}
                  onMouseEnter={e => { if (manufacturer !== m.key) e.currentTarget.style.background = 'rgba(0,0,0,0.04)'; }}
                  onMouseLeave={e => { if (manufacturer !== m.key) e.currentTarget.style.background = RAISED; }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, color: TEXT, marginBottom: 2 }}>{m.name}</div>
                  <div style={{ fontSize: 10, color: ACCENT, fontFamily: 'monospace', marginBottom: 2 }}>{m.formats}</div>
                  <div style={{ fontSize: 10, color: TEXT2 }}>{m.series}</div>
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 20 }}>
              <button disabled={!manufacturer} onClick={() => setSetupStep(2)} style={nextBtnStyle(!!manufacturer)}>
                Next →
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Antenna Frequency */}
        {setupStep === 2 && (
          <div style={{ padding: 24 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: TEXT, marginBottom: 4 }}>Select Antenna Frequency</p>
            <p style={{ fontSize: 11, color: TEXT2, marginBottom: 16 }}>Used to calculate rebar depth from signal travel time.</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {FREQ_OPTIONS.map(f => (
                <label key={f.mhz} style={{
                  display: 'flex', alignItems: 'flex-start', gap: 10, padding: '10px 14px',
                  background: (!useCustomFreq && frequencyMhz === f.mhz) ? 'rgba(232,96,28,0.08)' : RAISED,
                  border: `1.5px solid ${(!useCustomFreq && frequencyMhz === f.mhz) ? ACCENT : BORDER}`,
                  cursor: 'pointer',
                }}>
                  <input type="radio" name="freq" checked={!useCustomFreq && frequencyMhz === f.mhz}
                    onChange={() => { setFrequencyMhz(f.mhz); setUseCustomFreq(false); }}
                    style={{ accentColor: ACCENT, marginTop: 2 }} />
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: TEXT }}>{f.label}</div>
                    <div style={{ fontSize: 10, color: TEXT2, marginTop: 2 }}>{f.desc}</div>
                  </div>
                </label>
              ))}
              <label style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
                background: useCustomFreq ? 'rgba(232,96,28,0.08)' : RAISED,
                border: `1.5px solid ${useCustomFreq ? ACCENT : BORDER}`, cursor: 'pointer',
              }}>
                <input type="radio" name="freq" checked={useCustomFreq}
                  onChange={() => setUseCustomFreq(true)} style={{ accentColor: ACCENT }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: TEXT }}>Custom</span>
                {useCustomFreq && (
                  <input autoFocus value={customFreq} onChange={e => setCustomFreq(e.target.value)}
                    placeholder="MHz"
                    style={{ width: 80, padding: '4px 8px', background: PANEL, border: `1px solid ${BORDER2}`, color: TEXT, fontSize: 12, fontFamily: 'Inter, sans-serif', outline: 'none' }} />
                )}
              </label>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 20 }}>
              <button onClick={() => setSetupStep(1)} style={backBtnStyle}>← Back</button>
              <button onClick={() => setSetupStep(3)} style={nextBtnStyle(true)}>Next →</button>
            </div>
          </div>
        )}

        {/* Step 3: Project Details */}
        {setupStep === 3 && (
          <div style={{ padding: 24 }}>
            <p style={{ fontSize: 13, fontWeight: 600, color: TEXT, marginBottom: 4 }}>Project Details</p>
            <p style={{ fontSize: 11, color: TEXT2, marginBottom: 16 }}>
              Saved with the inspection record. Structure name is required.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <label style={labelStyle}>Project Name</label>
                <input value={projectName} onChange={e => setProjectName(e.target.value)}
                  placeholder="e.g. Manhattan Bridge Inspection 2026" style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Structure Name *</label>
                <input value={structureName} onChange={e => setStructureName(e.target.value)}
                  placeholder="e.g. Bridge B440029 — Deck A" style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Bridge ID / Asset Number</label>
                <input value={bridgeId} onChange={e => setBridgeId(e.target.value)}
                  placeholder="Optional" style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Inspection Date</label>
                <input type="date" value={inspDate} onChange={e => setInspDate(e.target.value)} style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Notes</label>
                <textarea value={notes} onChange={e => setNotes(e.target.value)}
                  rows={3} placeholder="Optional"
                  style={{ ...inputStyle, resize: 'vertical' }} />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 20 }}>
              <button onClick={() => setSetupStep(2)} style={backBtnStyle}>← Back</button>
              <button disabled={!structureName.trim()} onClick={onComplete}
                style={nextBtnStyle(!!structureName.trim())}>
                Begin Inspection
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
