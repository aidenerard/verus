import { useMemo } from 'react';
import * as THREE from 'three';
import type { Scene } from '../state/types';
import { useInteractiveStore } from '../state/useInteractiveStore';
import { spectral01 } from './colormap';

interface Props {
  scene: Scene;
}

const TEX_W = 128;
const TEX_H = 64;

export default function BridgeDeckSurface({ scene }: Props) {
  const cacheBust = useInteractiveStore(s => s.surfaceTextureCacheBust);
  const width  = scene.bbox.max[0] - scene.bbox.min[0];
  const height = scene.bbox.max[1] - scene.bbox.min[1];
  const cx     = (scene.bbox.min[0] + scene.bbox.max[0]) / 2;
  const cy     = (scene.bbox.min[1] + scene.bbox.max[1]) / 2;

  const texture = useMemo(() => makeSurfaceTexture(cacheBust),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cacheBust]);

  return (
    <group position={[cx, 0, cy]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, height, scene.surface.subdivisions.nx, scene.surface.subdivisions.ny]} />
        <meshStandardMaterial
          map={texture}
          transparent
          opacity={0.82}
          side={THREE.DoubleSide}
          roughness={0.85}
          metalness={0.0}
        />
      </mesh>
      <gridHelper
        args={[Math.max(width, height) * 1.05, 12, '#3A3E45', '#24262B']}
        position={[0, 0.002, 0]}
      />
    </group>
  );
}

function makeSurfaceTexture(seed: number): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = TEX_W;
  canvas.height = TEX_H;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(TEX_W, TEX_H);
  const phase = seed * 0.21;
  for (let y = 0; y < TEX_H; y++) {
    for (let x = 0; x < TEX_W; x++) {
      const fx = x / TEX_W;
      const fy = y / TEX_H;
      const v  =
        0.5 +
        0.30 * Math.sin((fx * 6.2) + phase) * Math.cos((fy * 4.1) - phase * 0.6) +
        0.15 * Math.sin((fx * 14.0) + (fy * 9.0) + phase * 1.4);
      const t = Math.max(0, Math.min(1, v));
      const [r, g, b] = spectral01(t);
      const i = (y * TEX_W + x) * 4;
      img.data[i + 0] = Math.round(r * 255);
      img.data[i + 1] = Math.round(g * 255);
      img.data[i + 2] = Math.round(b * 255);
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}
