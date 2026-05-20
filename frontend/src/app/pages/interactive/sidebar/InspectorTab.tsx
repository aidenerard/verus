import { useEffect, useRef, useState } from 'react';
import { Trash2, RotateCcw } from 'lucide-react';
import type { Scene, Pick } from '../state/types';
import { useInteractiveStore } from '../state/useInteractiveStore';
import { patchPickAndRevalidate } from '../state/hooks';
import { interactiveApi } from '../state/api';
import { Section, Row, NumberField, Button } from './fields';
import { TEXT, TEXT2, TEXT3 } from '../tokens';

interface Props { projectId: string; scene: Scene }

const REGRID_DEBOUNCE_MS = 400;

export default function InspectorTab({ projectId, scene }: Props) {
  const selectedPickIds = useInteractiveStore(s => s.selectedPickIds);
  const picks           = useInteractiveStore(s => s.picks);
  const bumpSurface     = useInteractiveStore(s => s.bumpSurfaceCache);
  const regridTimer     = useRef<number | null>(null);

  const id   = selectedPickIds[0];
  const pick = id ? picks.get(id) : undefined;

  useEffect(() => () => { if (regridTimer.current) window.clearTimeout(regridTimer.current); }, []);

  if (selectedPickIds.length === 0) {
    return <EmptyInspector hint="Click a rebar marker in the 3D scene to inspect it." />;
  }
  if (selectedPickIds.length > 1) {
    return <EmptyInspector hint={`${selectedPickIds.length} picks selected. Pick a single marker to edit.`} />;
  }
  if (!pick) {
    return <EmptyInspector hint="Selected pick is no longer present." />;
  }

  const scanLine = scene.scan_lines.find(s => s.id === pick.scan_line_id);

  const patch = async (delta: Partial<Pick>) => {
    await patchPickAndRevalidate(projectId, pick.id, delta);
    scheduleRegrid();
  };

  const scheduleRegrid = () => {
    if (regridTimer.current) window.clearTimeout(regridTimer.current);
    regridTimer.current = window.setTimeout(async () => {
      await interactiveApi.regrid(projectId);
      bumpSurface();
    }, REGRID_DEBOUNCE_MS);
  };

  const onDelete = () => patch({ is_deleted: true });
  const onReset  = () => patch({ is_edited: false });

  return (
    <div>
      <Section title="Identity">
        <Row label="Pick id"><Mono value={pick.id} /></Row>
        <Row label="Scan line"><Mono value={scanLine?.label ?? pick.scan_line_id} /></Row>
        <Row label="Trace #"><Mono value={String(pick.trace_idx)} /></Row>
      </Section>

      <Section title="Position">
        <Row label="x (ft)">
          <NumberField value={pick.x_ft} step={0.05} onChange={v => patch({ x_ft: v })} />
        </Row>
        <Row label="y (ft)">
          <NumberField value={pick.y_ft} step={0.05} onChange={v => patch({ y_ft: v })} />
        </Row>
        <Row label="lat"><Mono value={pick.lat?.toFixed(6) ?? '—'} /></Row>
        <Row label="lon"><Mono value={pick.lon?.toFixed(6) ?? '—'} /></Row>
      </Section>

      <Section title="Signal">
        <Row label="Depth (in)">
          <NumberField value={pick.depth_in} step={0.05} min={0} onChange={v => patch({ depth_in: v })} />
        </Row>
        <Row label="Amplitude"><Mono value={pick.amplitude.toFixed(3)} /></Row>
        <Row label="Confidence"><Mono value={`${(pick.confidence * 100).toFixed(0)}%`} /></Row>
        <Row label="Status">
          <span style={{ fontSize: 11, color: pick.is_edited ? '#fbbf24' : TEXT3 }}>
            {pick.is_edited ? 'Edited' : 'Original'}
          </span>
        </Row>
      </Section>

      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        {pick.is_edited && (
          <Button variant="ghost" onClick={onReset}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <RotateCcw size={11} /> Reset
            </span>
          </Button>
        )}
        <Button variant="danger" onClick={onDelete}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <Trash2 size={11} /> Delete pick
          </span>
        </Button>
      </div>
    </div>
  );
}

function EmptyInspector({ hint }: { hint: string }) {
  return (
    <div style={{ fontSize: 12, color: TEXT2, lineHeight: 1.55 }}>
      <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3, marginBottom: 8 }}>
        No pick selected
      </div>
      {hint}
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
