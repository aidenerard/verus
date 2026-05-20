import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import type { Scene, ViewMode } from '../state/types';
import { useInteractiveStore, loadCameraState, saveCameraState } from '../state/useInteractiveStore';
import BridgeDeckSurface from './BridgeDeckSurface';
import RebarPicks from './RebarPicks';
import ScanLines from './ScanLines';

interface Props {
  projectId: string;
  scene:     Scene;
}

export default function SceneCanvas({ projectId, scene }: Props) {
  const viewMode = useInteractiveStore(s => s.viewMode);
  const picks    = useInteractiveStore(s => Array.from(s.picks.values()));
  const clear    = useInteractiveStore(s => s.clearSelection);

  const cx = (scene.bbox.min[0] + scene.bbox.max[0]) / 2;
  const cy = (scene.bbox.min[1] + scene.bbox.max[1]) / 2;
  const span = Math.max(scene.bbox.max[0] - scene.bbox.min[0], scene.bbox.max[1] - scene.bbox.min[1]);

  const initialCamera = useMemo(() => {
    const saved = loadCameraState(projectId);
    if (saved) return saved;
    return {
      position: [cx, span * 0.9, cy + span * 0.6] as [number, number, number],
      target:   [cx, 0, cy] as [number, number, number],
      mode:     'three_d' as ViewMode,
    };
  }, [projectId, cx, cy, span]);

  return (
    <Canvas
      onPointerMissed={() => clear()}
      camera={{ position: initialCamera.position, fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false }}
      style={{ background: '#0F0F11' }}
    >
      <SceneLights />
      <BridgeDeckSurface scene={scene} />
      <ScanLines scanLines={scene.scan_lines} />
      <RebarPicks picks={picks} scene={scene} />
      <axesHelper args={[1.5]} />
      <CameraRig
        projectId={projectId}
        viewMode={viewMode}
        target={[cx, 0, cy]}
        span={span}
      />
    </Canvas>
  );
}

function SceneLights() {
  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[20, 40, 30]} intensity={1.05} />
      <directionalLight position={[-15, 25, -10]} intensity={0.45} color="#cdd6ff" />
    </>
  );
}

interface RigProps {
  projectId: string;
  viewMode:  ViewMode;
  target:    [number, number, number];
  span:      number;
}

function CameraRig({ projectId, viewMode, target, span }: RigProps) {
  const { camera } = useThree();
  const controlsRef = useRef<OrbitControlsImpl | null>(null);

  useEffect(() => {
    const c = camera as THREE.PerspectiveCamera;
    switch (viewMode) {
      case 'top':
        c.position.set(target[0], span * 1.4, target[2] + 0.001);
        break;
      case 'fixed':
        c.position.set(target[0] - span * 0.4, span * 0.7, target[2] + span * 0.9);
        break;
      case 'three_d':
      default:
        c.position.set(target[0], span * 0.9, target[2] + span * 0.6);
    }
    c.lookAt(target[0], target[1], target[2]);
    controlsRef.current?.target.set(target[0], target[1], target[2]);
    controlsRef.current?.update();
  }, [viewMode, target, span, camera]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      const c = controlsRef.current;
      if (!c) return;
      saveCameraState(projectId, {
        position: [c.object.position.x, c.object.position.y, c.object.position.z],
        target:   [c.target.x, c.target.y, c.target.z],
        mode:     viewMode,
      });
    }, 1500);
    return () => window.clearInterval(interval);
  }, [projectId, viewMode]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI * 0.49}
      minDistance={span * 0.15}
      maxDistance={span * 3}
      target={target}
      enableRotate={viewMode !== 'top'}
    />
  );
}
