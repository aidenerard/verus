const SPECTRAL: [number, number, number][] = [
  [0.6196, 0.0039, 0.2588],
  [0.8353, 0.2431, 0.3098],
  [0.9569, 0.4275, 0.2627],
  [0.9922, 0.6824, 0.3804],
  [0.9961, 0.8784, 0.5451],
  [1.0000, 1.0000, 0.7490],
  [0.9020, 0.9608, 0.5961],
  [0.6706, 0.8667, 0.6431],
  [0.4000, 0.7608, 0.6471],
  [0.1961, 0.5333, 0.7412],
  [0.3686, 0.3098, 0.6353],
];

function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function spectral01(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (SPECTRAL.length - 1);
  const lo  = Math.floor(idx);
  const hi  = Math.min(SPECTRAL.length - 1, lo + 1);
  return mix(SPECTRAL[lo], SPECTRAL[hi], idx - lo);
}

export function depthToColor(depth: number, range: [number, number]): [number, number, number] {
  const [lo, hi] = range;
  if (hi <= lo) return [0.5, 0.5, 0.5];
  return spectral01(1 - (depth - lo) / (hi - lo));
}

export function spectralStops(count = 7, range: [number, number] = [0, 1]) {
  const [lo, hi] = range;
  const stops: { color: string; label: string }[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    const [r, g, b] = spectral01(1 - t);
    const value = lo + t * (hi - lo);
    stops.push({
      color: `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`,
      label: value.toFixed(1),
    });
  }
  return stops;
}
