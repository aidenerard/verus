import { Plus, Trash2 } from 'lucide-react';
import type { Pick, Scene } from '../state/types';
import { useInteractiveStore } from '../state/useInteractiveStore';
import { patchPickAndRevalidate } from '../state/hooks';
import { Section, Row, Button } from './fields';
import { ACCENT, ACCENT_SOFT, BORDER, RAISED, TEXT, TEXT2, TEXT3 } from '../tokens';

interface Props { projectId: string; scene: Scene }

export default function InspectorTab({ projectId, scene }: Props) {
  const selectedPickIds = useInteractiveStore(s => s.selectedPickIds);
  const picks           = useInteractiveStore(s => s.picks);

  const id   = selectedPickIds[0];
  const pick = id ? picks.get(id) : undefined;

  if (selectedPickIds.length === 0) {
    return (
      <Empty hint="Click a rebar marker in the 3D scene to inspect it.">
        <AddPickStub />
      </Empty>
    );
  }
  if (selectedPickIds.length > 1) {
    return <Empty hint={`${selectedPickIds.length} picks selected. Pick a single marker to edit.`} />;
  }
  if (!pick) {
    return <Empty hint="Selected pick is no longer present." />;
  }

  const scanLine = scene.scan_lines.find(s => s.id === pick.scan_line_id);

  const onDelete = async () => {
    await patchPickAndRevalidate(projectId, pick.id, { is_deleted: true });
  };

  return (
    <div>
      <Section title="Identity">
        <Row label="Pick id"><Mono value={pick.id} /></Row>
        <Row label="Scan line"><Mono value={scanLine?.label ?? pick.scan_line_id} /></Row>
        <Row label="Trace #"><Mono value={String(pick.trace_idx)} /></Row>
      </Section>

      <Section title="Position">
        <Row label="x (ft)"><Mono value={pick.x_ft.toFixed(2)} /></Row>
        <Row label="y (ft)"><Mono value={pick.y_ft.toFixed(2)} /></Row>
        <Row label="lat"><Mono value={pick.lat?.toFixed(6) ?? '—'} /></Row>
        <Row label="lon"><Mono value={pick.lon?.toFixed(6) ?? '—'} /></Row>
      </Section>

      <Section title="Signal">
        <Row label="Depth (in)"><Mono value={pick.depth_in.toFixed(2)} /></Row>
        <Row label="Amplitude"><Mono value={pick.amplitude.toFixed(3)} /></Row>
        <Row label="Confidence"><Mono value={`${(pick.confidence * 100).toFixed(0)}%`} /></Row>
      </Section>

      <NotEditableNote />

      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <Button variant="danger" onClick={onDelete}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <Trash2 size={11} /> Delete pick
          </span>
        </Button>
      </div>
    </div>
  );
}

function NotEditableNote() {
  return (
    <div style={{
      background: RAISED, border: `1px solid ${BORDER}`,
      padding: '8px 10px', marginBottom: 14,
      fontSize: 11, color: TEXT2, lineHeight: 1.5,
    }}>
      Position and depth are derived from survey + velocity and aren't user-editable.
      Adjust <strong style={{ color: TEXT }}>Velocity</strong> in the Processing tab to recompute depths globally.
    </div>
  );
}

function AddPickStub() {
  return (
    <div
      role="note"
      style={{
        marginTop: 16, padding: '14px 14px',
        border: `1px dashed ${ACCENT}`, background: ACCENT_SOFT,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <Plus size={12} style={{ color: ACCENT }} />
        <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.10em', textTransform: 'uppercase', color: ACCENT }}>
          Add pick
        </span>
      </div>
      <div style={{ fontSize: 11, color: TEXT2, lineHeight: 1.5 }}>
        Click anywhere on the deck surface in the 3D scene to drop a new pick at that location.
        <span style={{ color: TEXT3 }}> (wiring lands once the backend exposes POST&nbsp;/picks)</span>
      </div>
    </div>
  );
}

function Empty({ hint, children }: { hint: string; children?: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3, marginBottom: 8 }}>
        No pick selected
      </div>
      <div style={{ fontSize: 12, color: TEXT2, lineHeight: 1.55 }}>
        {hint}
      </div>
      {children}
    </div>
  );
}

function Mono({ value }: { value: string }) {
  return (
    <span style={{ fontSize: 11, color: TEXT, fontFamily: 'ui-monospace, monospace', fontVariantNumeric: 'tabular-nums' }}>
      {value}
    </span>
  );
}
