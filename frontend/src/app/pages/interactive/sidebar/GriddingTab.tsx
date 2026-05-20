import { useEffect, useState } from 'react';
import { useGridding, saveGridding } from '../state/hooks';
import { interactiveApi } from '../state/api';
import { useInteractiveStore } from '../state/useInteractiveStore';
import type { GriddingAlgorithm, GriddingConfig } from '../state/types';
import { Section, Row, Slider, Select, Toggle, Button } from './fields';
import { TEXT2, TEXT3 } from '../tokens';

const ALGO_OPTIONS: { value: GriddingAlgorithm; label: string }[] = [
  { value: 'nearest_neighbor',   label: 'Nearest Neighbor' },
  { value: 'idw',                label: 'Inverse Distance Weighting' },
  { value: 'natural_neighbor',   label: 'Natural Neighbor' },
  { value: 'minimum_curvature',  label: 'Minimum Curvature' },
  { value: 'kriging',            label: 'Kriging' },
];

export default function GriddingTab({ projectId }: { projectId: string }) {
  const { data } = useGridding(projectId);
  const bumpSurface = useInteractiveStore(s => s.bumpSurfaceCache);
  const [cfg, setCfg] = useState<GriddingConfig | undefined>(undefined);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (data && !cfg) setCfg(data); }, [data, cfg]);

  if (!cfg) return <div style={{ color: TEXT3, fontSize: 12 }}>Loading gridding config…</div>;

  const showAniso = cfg.algorithm === 'kriging' || cfg.algorithm === 'minimum_curvature';

  const regrid = async () => {
    setBusy(true);
    try {
      await saveGridding(projectId, cfg);
      await interactiveApi.regrid(projectId);
      bumpSurface();
    } finally { setBusy(false); }
  };

  return (
    <div>
      <Section title="Algorithm">
        <Row label="Method">
          <Select<GriddingAlgorithm>
            value={cfg.algorithm}
            onChange={v => setCfg({ ...cfg, algorithm: v })}
            options={ALGO_OPTIONS}
          />
        </Row>
      </Section>

      <Section title="Search">
        <Row label="Radius (ft)">
          <Slider
            value={cfg.search_radius_ft}
            onChange={v => setCfg({ ...cfg, search_radius_ft: v })}
            min={0.5} max={20} step={0.25}
          />
        </Row>
        <Row label="Edge clip">
          <Toggle checked={cfg.edge_clip} onChange={v => setCfg({ ...cfg, edge_clip: v })} />
        </Row>
        <Row label="Cell size (ft)">
          <Slider
            value={cfg.cell_size_ft}
            onChange={v => setCfg({ ...cfg, cell_size_ft: v })}
            min={0.1} max={3} step={0.05}
          />
        </Row>
      </Section>

      {showAniso && (
        <Section title="Anisotropy">
          <Row label="Angle (°)">
            <Slider
              value={cfg.anisotropy.angle_deg}
              onChange={v => setCfg({ ...cfg, anisotropy: { ...cfg.anisotropy, angle_deg: v } })}
              min={0} max={180} step={1}
            />
          </Row>
          <Row label="Ratio">
            <Slider
              value={cfg.anisotropy.ratio}
              onChange={v => setCfg({ ...cfg, anisotropy: { ...cfg.anisotropy, ratio: v } })}
              min={0.25} max={4} step={0.05}
            />
          </Row>
        </Section>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid #2A2D32', paddingTop: 12 }}>
        <Button variant="primary" onClick={regrid} disabled={busy}>
          {busy ? 'Re-gridding…' : 'Re-grid'}
        </Button>
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: TEXT2 }}>
        Re-grid recomputes the surface from current picks.
      </div>
    </div>
  );
}
