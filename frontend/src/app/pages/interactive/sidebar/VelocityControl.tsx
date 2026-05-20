import { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { mutate } from 'swr';
import { interactiveApi, picksKey, sceneKey } from '../state/api';
import { useProcessing, saveProcessing } from '../state/hooks';
import { useInteractiveStore } from '../state/useInteractiveStore';
import { Section, Row, Slider } from './fields';
import { ACCENT, BORDER, RAISED, TEXT, TEXT2, TEXT3 } from '../tokens';

interface Props { projectId: string }

const DEBOUNCE_MS    = 350;
const MIN_VELOCITY   = 0.05;
const MAX_VELOCITY   = 0.15;
const STEP_VELOCITY  = 0.005;
const DEFAULT_VEL    = 0.10;

export default function VelocityControl({ projectId }: Props) {
  const { data: cfg } = useProcessing(projectId);
  const bumpSurface   = useInteractiveStore(s => s.bumpSurfaceCache);
  const initial       = cfg?.velocity_m_per_ns ?? DEFAULT_VEL;
  const [velocity, setVelocity] = useState(initial);
  const [busy,     setBusy]     = useState(false);
  const debounce = useRef<number | null>(null);
  const synced   = useRef(false);

  useEffect(() => {
    if (!synced.current && cfg) {
      setVelocity(cfg.velocity_m_per_ns ?? DEFAULT_VEL);
      synced.current = true;
    }
  }, [cfg]);

  useEffect(() => () => { if (debounce.current) window.clearTimeout(debounce.current); }, []);

  const onChange = (v: number) => {
    setVelocity(v);
    if (debounce.current) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(async () => {
      if (!cfg) return;
      setBusy(true);
      try {
        await saveProcessing(projectId, { ...cfg, velocity_m_per_ns: v });
        await interactiveApi.reprocess(projectId);
        await Promise.all([mutate(picksKey(projectId)), mutate(sceneKey(projectId))]);
        bumpSurface();
      } finally {
        setBusy(false);
      }
    }, DEBOUNCE_MS);
  };

  return (
    <Section title="Velocity">
      <Row label="m / ns">
        <Slider value={velocity} onChange={onChange} min={MIN_VELOCITY} max={MAX_VELOCITY} step={STEP_VELOCITY} />
      </Row>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '8px 10px', background: RAISED, border: `1px solid ${BORDER}`,
        fontSize: 11, lineHeight: 1.5,
        color: busy ? ACCENT : TEXT2,
      }}>
        {busy ? (
          <>
            <Loader2 size={12} className="velocity-spin" style={{ color: ACCENT }} />
            <span>Recomputing depths…</span>
          </>
        ) : (
          <>
            <span style={{ color: TEXT, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
              {velocity.toFixed(3)} m/ns
            </span>
            <span style={{ color: TEXT3 }}>· depth = v · t / 2</span>
          </>
        )}
      </div>
      <style>{`
        @keyframes velocity-spin-kf { to { transform: rotate(360deg); } }
        .velocity-spin { animation: velocity-spin-kf 0.9s linear infinite; }
      `}</style>
    </Section>
  );
}
