import { Zap } from 'lucide-react';
import PlaceholderWorkspace from './PlaceholderWorkspace';

export default function FDEMWorkspace() {
  return (
    <PlaceholderWorkspace
      Icon={Zap}
      content={{
        name: 'FDEM',
        standard: 'Frequency-Domain Electromagnetic Induction',
        description:
          'FDEM measures the apparent electrical conductivity and magnetic susceptibility of the subsurface from the in-phase and out-of-phase response to a transmitted electromagnetic field.',
        useCases: [
          'Locating buried metallic and non-metallic utilities at depth',
          'Mapping conductive plumes and saline groundwater intrusion',
          'Delineating fill, voids, and disturbed soil over large survey areas',
        ],
      }}
    />
  );
}
