import { useMemo } from 'react';
import * as THREE from 'three';
import { Line } from '@react-three/drei';
import type { ScanLineMeta } from '../state/types';

interface Props {
  scanLines: ScanLineMeta[];
}

const COLORS = ['#9C5BFF', '#3B8EEA', '#1ABC9C', '#F1C40F', '#E67E22', '#E74C3C', '#7F8C8D'];

export default function ScanLines({ scanLines }: Props) {
  const lines = useMemo(() => scanLines.map((sl, i) => ({
    id:     sl.id,
    color:  COLORS[i % COLORS.length],
    points: sl.points.map(([x, y]) => new THREE.Vector3(x, 0.01, y)) as [THREE.Vector3, ...THREE.Vector3[]],
  })), [scanLines]);

  return (
    <>
      {lines.map(l => (
        <Line
          key={l.id}
          points={l.points}
          color={l.color}
          lineWidth={1.5}
          dashed={false}
          transparent
          opacity={0.7}
        />
      ))}
    </>
  );
}
