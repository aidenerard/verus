import { Radio, Waves, Compass, Activity, Zap } from 'lucide-react';
import type { ModuleMeta } from './types';

export const MODULES: ModuleMeta[] = [
  {
    id: 'em',
    label: 'Electromagnetic',
    methods: [
      { id: 'gpr',          name: 'GPR',          fullName: 'Ground-Penetrating Radar', path: '/workspace/em/gpr',          status: 'available',   Icon: Radio  },
      { id: 'fdem',         name: 'FDEM',         fullName: 'Frequency-Domain EM',      path: '/workspace/em/fdem',         status: 'coming-soon', Icon: Zap    },
      { id: 'magnetometer', name: 'Magnetometer', fullName: 'Magnetic Gradiometry',     path: '/workspace/em/magnetometer', status: 'coming-soon', Icon: Compass },
    ],
  },
  {
    id: 'seismic',
    label: 'Seismic',
    methods: [
      { id: 'masw',         name: 'MASW',        fullName: 'Multichannel Analysis of Surface Waves', path: '/workspace/seismic/masw',         status: 'coming-soon', Icon: Waves    },
      { id: 'impact-echo',  name: 'Impact Echo', fullName: 'Impact-Echo Spectral Analysis',          path: '/workspace/seismic/impact-echo',  status: 'coming-soon', Icon: Activity },
    ],
  },
];

export function findMethod(path: string) {
  for (const mod of MODULES) {
    const m = mod.methods.find(x => x.path === path);
    if (m) return { module: mod, method: m };
  }
  return null;
}
