import type { Pick, ScanLineTraces } from '../state/types';
import { useInteractiveStore } from '../state/useInteractiveStore';

interface Props {
  traces:      ScanLineTraces;
  picks:       Pick[];
  pxPerTrace:  number;
  pxPerSample: number;
}

const DOT_R = 4;
const HALO_R = 8;

export default function PickDots({ traces, picks, pxPerTrace, pxPerSample }: Props) {
  const selectedPickIds = useInteractiveStore(s => s.selectedPickIds);
  const selectPick      = useInteractiveStore(s => s.selectPick);
  const selectedSet     = new Set(selectedPickIds);

  const w = traces.n_traces  * pxPerTrace;
  const h = traces.n_samples * pxPerSample;

  return (
    <svg
      width={w} height={h}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {picks.map(p => {
        const cx = p.trace_idx  * pxPerTrace + pxPerTrace / 2;
        const cy = p.sample_idx * pxPerSample + pxPerSample / 2;
        const sel = selectedSet.has(p.id);
        return (
          <g key={p.id} style={{ pointerEvents: 'auto', cursor: 'pointer' }}
             onClick={(e) => selectPick(p.id, e.shiftKey)}>
            {sel && (
              <circle cx={cx} cy={cy} r={HALO_R} fill="none" stroke="#E8601C" strokeWidth={1.5} opacity={0.9} />
            )}
            <circle
              cx={cx} cy={cy} r={DOT_R}
              fill={sel ? '#fff' : '#ff3da8'}
              stroke={sel ? '#E8601C' : 'rgba(0,0,0,0.4)'}
              strokeWidth={sel ? 2 : 1}
            />
          </g>
        );
      })}
    </svg>
  );
}
