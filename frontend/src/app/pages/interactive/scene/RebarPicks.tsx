import { useMemo } from 'react';
import * as THREE from 'three';
import { Instances, Instance } from '@react-three/drei';
import { ThreeEvent } from '@react-three/fiber';
import type { Pick, Scene } from '../state/types';
import { useInteractiveStore } from '../state/useInteractiveStore';
import { depthToColor } from './colormap';

interface Props {
  picks: Pick[];
  scene: Scene;
}

const RADIUS = 0.18;
const SELECT_RAISE = 0.25;

export default function RebarPicks({ picks, scene }: Props) {
  const selectedPickIds = useInteractiveStore(s => s.selectedPickIds);
  const selectPick      = useInteractiveStore(s => s.selectPick);

  const selectedSet = useMemo(() => new Set(selectedPickIds), [selectedPickIds]);
  const range       = scene.surface.depth_range_in;

  const onClick = (id: string) => (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    selectPick(id, e.shiftKey);
  };

  return (
    <>
      <Instances limit={Math.max(picks.length, 1)} castShadow={false}>
        <sphereGeometry args={[RADIUS, 14, 12]} />
        <meshStandardMaterial roughness={0.4} metalness={0.05} />
        {picks.map(p => {
          const [r, g, b] = depthToColor(p.depth_in, range);
          const z = -p.depth_in / 12;
          return (
            <Instance
              key={p.id}
              position={[p.x_ft, z, p.y_ft]}
              color={new THREE.Color(r, g, b)}
              onClick={onClick(p.id)}
            />
          );
        })}
      </Instances>

      {picks.filter(p => selectedSet.has(p.id)).map(p => {
        const z = -p.depth_in / 12 + SELECT_RAISE;
        return (
          <group key={`sel-${p.id}`} position={[p.x_ft, z, p.y_ft]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[RADIUS * 1.8, RADIUS * 2.4, 24]} />
              <meshBasicMaterial color="#E8601C" transparent opacity={0.9} side={THREE.DoubleSide} />
            </mesh>
            <mesh>
              <sphereGeometry args={[RADIUS * 1.1, 14, 12]} />
              <meshBasicMaterial color="#FFFFFF" transparent opacity={0.85} />
            </mesh>
          </group>
        );
      })}
    </>
  );
}
