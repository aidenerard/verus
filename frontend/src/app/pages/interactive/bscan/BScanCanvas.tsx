import { useEffect, useRef } from 'react';
import type { ScanLineTraces } from '../state/types';

interface Props {
  traces:    ScanLineTraces;
  pxPerTrace: number;
  pxPerSample: number;
}

export default function BScanCanvas({ traces, pxPerTrace, pxPerSample }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const w = traces.n_traces;
    const h = traces.n_samples;
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext('2d', { willReadFrequently: false });
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let s = 0; s < h; s++) {
      for (let t = 0; t < w; t++) {
        const v = traces.data[t]?.[s] ?? 0;
        const g = Math.round(((v + 128) / 255) * 255);
        const i = (s * w + t) * 4;
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [traces]);

  const displayW = traces.n_traces * pxPerTrace;
  const displayH = traces.n_samples * pxPerSample;

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: 'block',
        width:  `${displayW}px`,
        height: `${displayH}px`,
        imageRendering: 'pixelated',
      }}
    />
  );
}
