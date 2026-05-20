import { Compass } from 'lucide-react';
import PlaceholderWorkspace from './PlaceholderWorkspace';

export default function MagWorkspace() {
  return (
    <PlaceholderWorkspace
      Icon={Compass}
      content={{
        name: 'Magnetometer',
        standard: 'Total-Field & Gradient Magnetic Survey',
        description:
          'Magnetometry measures spatial variations in the Earth\'s magnetic field caused by ferrous objects, geological contrasts, or anthropogenic disturbances near the surface.',
        useCases: [
          'Locating buried steel drums, tanks, and unexploded ordnance',
          'Mapping ferrous utilities, well casings, and abandoned infrastructure',
          'Archaeological prospection of foundations, hearths, and iron artifacts',
        ],
      }}
    />
  );
}
