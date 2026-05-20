import { useEffect, useRef, useState } from 'react';
import { useProcessing, saveProcessing } from '../state/hooks';
import { interactiveApi } from '../state/api';
import { useInteractiveStore } from '../state/useInteractiveStore';
import type { FilterStep, ProcessingConfig, Scene } from '../state/types';
import { Section, Row, Slider, NumberField, Button, Select } from './fields';
import FilterRow from './FilterRow';
import { TEXT2, TEXT3 } from '../tokens';

interface Props { projectId: string; scene: Scene }

export default function ProcessingTab({ projectId, scene }: Props) {
  const { data } = useProcessing(projectId);
  const bumpSurface = useInteractiveStore(s => s.bumpSurfaceCache);
  const selectedPick = useInteractiveStore(s => {
    const id = s.selectedPickIds[0];
    return id ? s.picks.get(id) : undefined;
  });

  const [cfg, setCfg] = useState<ProcessingConfig | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const dragId = useRef<string | null>(null);

  useEffect(() => { if (data && !cfg) setCfg(data); }, [data, cfg]);

  if (!cfg) return <div style={{ color: TEXT3, fontSize: 12 }}>Loading processing config…</div>;

  const focusedScanLineId =
    selectedPick?.scan_line_id ?? scene.scan_lines[0]?.id ?? '';
  const scanLineLabel =
    scene.scan_lines.find(s => s.id === focusedScanLineId)?.label ?? focusedScanLineId;
  const currentShift = cfg.time_zero_shifts[focusedScanLineId] ?? 0;

  const setShift = (v: number) => setCfg({
    ...cfg, time_zero_shifts: { ...cfg.time_zero_shifts, [focusedScanLineId]: v },
  });

  const setFilter = (next: FilterStep) => setCfg({
    ...cfg, filters: cfg.filters.map(f => f.id === next.id ? next : f),
  });

  const reorderTo = (overId: string) => {
    if (!dragId.current || dragId.current === overId) return;
    const ordered = [...cfg.filters];
    const from = ordered.findIndex(f => f.id === dragId.current);
    const to   = ordered.findIndex(f => f.id === overId);
    if (from < 0 || to < 0) return;
    const [moved] = ordered.splice(from, 1);
    ordered.splice(to, 0, moved);
    setCfg({ ...cfg, filters: ordered });
  };

  const apply = async () => {
    setBusy(true);
    try {
      await saveProcessing(projectId, cfg);
      await interactiveApi.reprocess(projectId);
      bumpSurface();
    } finally { setBusy(false); }
  };

  const reset = () => { if (data) setCfg(data); };

  return (
    <div>
      <Section title={`Time-zero (${scanLineLabel})`}>
        <Row label="Shift (ns)">
          <Slider value={currentShift} onChange={setShift} min={-2} max={2} step={0.05} />
        </Row>
        <div style={{ fontSize: 10, color: TEXT3 }}>
          Adjusts the t=0 offset for the currently-inspected scan line.
        </div>
      </Section>

      <Section title="Filter Chain">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {cfg.filters.map(f => (
            <FilterRow
              key={f.id}
              step={f}
              onChange={setFilter}
              onDragStart={id => { dragId.current = id; }}
              onDragOver={reorderTo}
              onDrop={() => { dragId.current = null; }}
            />
          ))}
        </div>
        <Row label="Add">
          <Select<FilterStep['type']>
            value="bandpass"
            onChange={t => setCfg({
              ...cfg,
              filters: [...cfg.filters, { id: `f-${Date.now()}`, type: t, enabled: true, params: defaultParams(t) }],
            })}
            options={[
              { value: 'bandpass',           label: '+ Bandpass' },
              { value: 'background_removal', label: '+ Background Removal' },
              { value: 'gain',               label: '+ Gain' },
              { value: 'agc',                label: '+ AGC' },
              { value: 'hilbert',            label: '+ Hilbert Envelope' },
            ]}
          />
        </Row>
      </Section>

      <Section title="GPS">
        <Row label="Latency (ms)">
          <NumberField
            value={cfg.gps_latency_ms} step={1}
            onChange={v => setCfg({ ...cfg, gps_latency_ms: v })}
          />
        </Row>
      </Section>

      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', borderTop: '1px solid #2A2D32', paddingTop: 12 }}>
        <Button variant="ghost" onClick={reset}>Reset</Button>
        <Button variant="primary" onClick={apply} disabled={busy}>
          {busy ? 'Applying…' : 'Apply'}
        </Button>
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: TEXT2 }}>
        Apply re-runs the pipeline and refreshes the surface.
      </div>
    </div>
  );
}

function defaultParams(type: FilterStep['type']): FilterStep['params'] {
  switch (type) {
    case 'bandpass':           return { low_mhz: 800, high_mhz: 2400, order: 4 };
    case 'background_removal': return { window: 50 };
    case 'gain':               return { gain_db: 6 };
    case 'agc':                return { window_ns: 8 };
    case 'hilbert':            return {};
  }
}
