import { Waves } from 'lucide-react';
import PlaceholderWorkspace from './PlaceholderWorkspace';

export default function MASWWorkspace() {
  return (
    <PlaceholderWorkspace
      Icon={Waves}
      content={{
        name: 'MASW',
        standard: 'ASTM D7400 · Multichannel Analysis of Surface Waves',
        description:
          'MASW analyzes the dispersion of Rayleigh surface waves across a linear receiver array to invert for a depth-dependent shear-wave velocity profile of the subsurface.',
        useCases: [
          '1D and 2D Vs profiles for foundation and seismic site classification',
          'Detection of voids, soft zones, and stiffness anomalies under pavement',
          'Bedrock mapping and characterization of fill thickness',
        ],
      }}
    />
  );
}
