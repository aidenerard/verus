import { useRef, useState } from 'react';
import { UploadCloud, X, ChevronDown, ChevronRight, SlidersHorizontal, Play } from 'lucide-react';
import type { UploadedFile } from '../inspect/types';
import { MANUFACTURERS, FREQ_OPTIONS, GPR_EXTS, type ManufacturerKey } from '../inspect/constants';
import { ACCENT, BORDER, BORDER2, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';
import ProcessingOptionsPanel from './ProcessingOptionsPanel';
import ConfirmSwipeModal from './ConfirmSwipeModal';

interface Props {
  files:            UploadedFile[];
  setFiles:         React.Dispatch<React.SetStateAction<UploadedFile[]>>;
  manufacturer:     ManufacturerKey | '';
  setManufacturer:  (m: ManufacturerKey | '') => void;
  frequencyMhz:     number;
  setFrequencyMhz:  (f: number) => void;
  analysisName:     string;
  setAnalysisName:  (v: string) => void;
  analysisNotes:    string;
  setAnalysisNotes: (v: string) => void;
  onStart:          () => void;
  busy:             boolean;
}

export default function GPRUploadCard({
  files, setFiles, manufacturer, setManufacturer, frequencyMhz, setFrequencyMhz,
  analysisName, setAnalysisName, analysisNotes, setAnalysisNotes,
  onStart, busy,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [confirmOpen,  setConfirmOpen]  = useState(false);

  const disabled = !files.length || busy;

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const accepted = Array.from(incoming).filter(f =>
      GPR_EXTS.has(f.name.slice(f.name.lastIndexOf('.')).toLowerCase()));
    if (!accepted.length) return;
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...accepted.filter(f => !existing.has(f.name)).map(f => ({ file: f, name: f.name }))];
    });
  };

  const removeFile = (name: string) =>
    setFiles(prev => prev.filter(f => f.name !== name));

  const onDrop: React.DragEventHandler<HTMLDivElement> = e => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  };

  return (
    <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '24px 28px' }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: TEXT, marginBottom: 4 }}>Upload GPR Scan Files</div>
      <div style={{ fontSize: 12, color: TEXT2, marginBottom: 18 }}>
        Upload all files from your Proceq dataset folder — PRC scan files, .pos GPS files, and .CScan amplitude files. The system groups them into swaths automatically. (Also accepts GSSI .dzt, S&S .dt1, MALA .rd3/.rd7, SEG-Y, and .csv.)
      </div>

      <div
        onDragOver={e => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${BORDER2}`, padding: '32px 24px',
          background: RAISED, textAlign: 'center', cursor: 'pointer',
          marginBottom: 18,
        }}
      >
        <UploadCloud size={28} style={{ color: TEXT3, marginBottom: 8 }} />
        <div style={{ fontSize: 13, color: TEXT, fontWeight: 600 }}>Click to browse or drop files here</div>
        <div style={{ fontSize: 11, color: TEXT3, marginTop: 4 }}>Up to ~500 MB per upload</div>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv,.dzt,.dt1,.rd3,.rd7,.segy,.sgy,.dzg,.hd,.rad,.dt,.gec,.iprb,.iprh,.scan,.pos,.cscan,.CScan"
        onChange={e => { addFiles(e.target.files); e.target.value = ''; }}
        style={{ display: 'none' }}
      />

      {files.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT3, marginBottom: 6 }}>
            {files.length} file{files.length !== 1 ? 's' : ''} ready
          </div>
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, maxHeight: 120, overflowY: 'auto', border: `1px solid ${BORDER}` }}>
            {files.map(f => (
              <li key={f.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 10px', fontSize: 12, color: TEXT, background: PANEL, borderBottom: `1px solid ${BORDER}` }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                <button onClick={e => { e.stopPropagation(); removeFile(f.name); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, padding: 2, display: 'flex' }}>
                  <X size={12} />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 18 }}>
        <FieldLabel text="Analysis Name">
          <input
            type="text"
            value={analysisName}
            onChange={e => setAnalysisName(e.target.value)}
            placeholder="e.g. Bridge B440029 — October 2024"
            maxLength={100}
            style={{
              width: '100%', padding: '8px 10px',
              background: RAISED, border: `1px solid ${BORDER}`,
              color: TEXT, fontSize: 12, fontFamily: 'inherit', outline: 'none',
            }}
          />
        </FieldLabel>

        <FieldLabel text="Notes" optional>
          <textarea
            value={analysisNotes}
            onChange={e => setAnalysisNotes(e.target.value)}
            placeholder="Any relevant notes about this scan…"
            maxLength={500}
            rows={2}
            style={{
              width: '100%', padding: '8px 10px',
              background: RAISED, border: `1px solid ${BORDER}`,
              color: TEXT, fontSize: 12, fontFamily: 'inherit', outline: 'none',
              resize: 'vertical', minHeight: 44,
            }}
          />
        </FieldLabel>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 18 }}>
        <LabelledSelect
          label="Equipment"
          value={manufacturer}
          onChange={v => setManufacturer(v as ManufacturerKey | '')}
          options={[{ value: '', label: 'Auto-detect' }, ...MANUFACTURERS.map(m => ({ value: m.key, label: m.name }))]}
        />
        <LabelledSelect
          label="Frequency"
          value={String(frequencyMhz)}
          onChange={v => setFrequencyMhz(parseInt(v))}
          options={FREQ_OPTIONS.map(f => ({ value: String(f.mhz), label: f.label }))}
        />
      </div>

      <div style={{ borderTop: `1px solid ${BORDER}`, margin: '0 -28px 18px', padding: '14px 28px 0' }}>
        <button
          type="button"
          onClick={() => setAdvancedOpen(v => !v)}
          aria-expanded={advancedOpen}
          style={{
            width: '100%', background: 'none', border: 'none', padding: 0, cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 11, fontWeight: 800, letterSpacing: '0.10em', textTransform: 'uppercase',
            color: TEXT2, fontFamily: 'inherit',
          }}
        >
          {advancedOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <SlidersHorizontal size={12} />
          Advanced Options
        </button>

        {advancedOpen && (
          <div style={{ marginTop: 16, padding: '16px 18px', background: RAISED, border: `1px solid ${BORDER}` }}>
            <ProcessingOptionsPanel />
          </div>
        )}
      </div>

      <button
        type="button"
        onClick={() => setConfirmOpen(true)}
        disabled={disabled}
        style={{
          width: '100%', padding: '12px 16px',
          background: disabled ? BORDER2 : ACCENT, color: '#fff', border: 'none',
          cursor: disabled ? 'not-allowed' : 'pointer',
          fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
          fontFamily: 'inherit',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        }}
      >
        <Play size={14} /> Start Analysis
      </button>

      <ConfirmSwipeModal
        open={confirmOpen}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => { setConfirmOpen(false); onStart(); }}
        rows={summaryRows(files.length, manufacturer, frequencyMhz, analysisName, analysisNotes)}
      />
    </div>
  );
}

function summaryRows(
  fileCount: number, manufacturer: ManufacturerKey | '', frequencyMhz: number,
  analysisName: string, analysisNotes: string,
) {
  const mfr = MANUFACTURERS.find(m => m.key === manufacturer);
  return [
    { label: 'Analysis Name', value: analysisName.trim()  || 'Untitled Analysis' },
    { label: 'Notes',         value: analysisNotes.trim() || '—' },
    { label: 'Files',         value: `${fileCount} file${fileCount !== 1 ? 's' : ''} selected` },
    { label: 'Method',        value: 'GPR' },
    { label: 'Frequency',     value: `${frequencyMhz} MHz` },
    { label: 'Equipment',     value: mfr?.name ?? 'Auto-detect' },
  ];
}

function FieldLabel({ text, optional, children }: { text: string; optional?: boolean; children: React.ReactNode }) {
  return (
    <label style={{ display: 'block' }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT3, marginBottom: 4, display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span>{text}</span>
        {optional && <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.04em', textTransform: 'none', color: TEXT3, opacity: 0.7 }}>(optional)</span>}
      </div>
      {children}
    </label>
  );
}

function LabelledSelect({
  label, value, onChange, options,
}: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <label style={{ display: 'block' }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT3, marginBottom: 4 }}>
        {label}
      </div>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{ width: '100%', padding: '8px 10px', background: RAISED, border: `1px solid ${BORDER}`, color: TEXT, fontSize: 12, fontFamily: 'inherit', outline: 'none' }}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  );
}
