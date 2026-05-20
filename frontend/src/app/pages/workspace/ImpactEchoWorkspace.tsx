import { Activity } from 'lucide-react';
import PlaceholderWorkspace from './PlaceholderWorkspace';

export default function ImpactEchoWorkspace() {
  return (
    <PlaceholderWorkspace
      Icon={Activity}
      content={{
        name: 'Impact Echo',
        standard: 'ASTM C1383 · Impact-Echo Spectral Analysis',
        description:
          'Impact echo applies a short mechanical impact to a structure and measures the resulting stress-wave reflections. Spectral peaks in the response identify thickness, delamination, and voids in concrete.',
        useCases: [
          'Plate thickness verification of slabs, walls, and bridge decks',
          'Detection of delamination, honeycombing, and grouting flaws',
          'Crack-depth estimation and void detection behind concrete liners',
        ],
      }}
    />
  );
}
