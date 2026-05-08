import { X } from 'lucide-react';
import { PANEL, RAISED, BORDER, BORDER2, TEXT, TEXT2, ACCENT, MANUFACTURERS } from './constants';
import type { ManufacturerKey } from './constants';
import type { UploadedFile } from './types';
import { estimateAnalysisSeconds } from './utils';

interface Props {
  files: UploadedFile[];
  manufacturer: ManufacturerKey | '';
  frequencyMhz: number;
  onConfirm: () => void;
  onCancel: () => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.ceil(bytes / 1e3)} KB`;
}

function formatEstimate(secs: number): string {
  if (secs >= 60) {
    const mins = Math.ceil(secs / 60);
    return `~${mins} minute${mins !== 1 ? 's' : ''}`;
  }
  return `~${secs} seconds`;
}

export default function ConfirmAnalysisModal({ files, manufacturer, frequencyMhz, onConfirm, onCancel }: Props) {
  const totalBytes = files.reduce((s, f) => s + f.file.size, 0);
  const mfrName = MANUFACTURERS.find(m => m.key === manufacturer)?.name ?? (manufacturer || '—');
  const estimatedSecs = estimateAnalysisSeconds(files.map(f => f.file), manufacturer);
  const showServerNote = estimatedSecs > 300;

  const rows = [
    { label: 'Files',     value: `${files.length} file${files.length !== 1 ? 's' : ''} · ${formatSize(totalBytes)}` },
    { label: 'Equipment', value: mfrName },
    { label: 'Frequency', value: `${frequencyMhz} MHz` },
    { label: 'Model',     value: 'model_v13.pth' },
    { label: 'Standard',  value: 'ASTM D6087' },
  ];

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(10,10,10,0.65)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        width: 440, background: PANEL, border: `1px solid ${BORDER2}`,
        boxShadow: '0 24px 64px rgba(0,0,0,0.28)',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      }}>
        <div style={{ padding: '18px 24px 14px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 15, fontWeight: 700, color: TEXT }}>Ready to Analyze</span>
          <button onClick={onCancel} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}>
            <X size={16} />
          </button>
        </div>

        <div style={{ padding: '20px 24px' }}>
          <div style={{ background: RAISED, border: `1px solid ${BORDER}`, padding: '14px 16px', marginBottom: 12 }}>
            {rows.map(({ label, value }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                <span style={{ color: TEXT2 }}>{label}</span>
                <span style={{ color: TEXT, fontWeight: 600 }}>{value}</span>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: 20, paddingLeft: 2 }}>
            <span style={{ fontSize: 11, color: TEXT2 }}>
              Estimated analysis time: <strong style={{ color: TEXT, fontWeight: 600 }}>{formatEstimate(estimatedSecs)}</strong>
            </span>
            {showServerNote && (
              <div style={{ marginTop: 4, fontSize: 10, color: TEXT2, opacity: 0.75 }}>
                Large uploads may take longer on the free server tier.
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
            <button onClick={onCancel}
              style={{ padding: '9px 20px', background: 'none', border: `1px solid ${BORDER2}`, color: TEXT2, fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
              Cancel
            </button>
            <button onClick={onConfirm}
              style={{ padding: '9px 24px', background: ACCENT, border: 'none', color: '#fff', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
              Begin Analysis
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
