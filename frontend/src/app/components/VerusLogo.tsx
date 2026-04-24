interface VerusLogoProps {
  size?: number;
  wordmarkColor?: string;
  showWordmark?: boolean;
}

export default function VerusLogo({
  size = 40,
  wordmarkColor = '#2C3E50',
  showWordmark = true,
}: VerusLogoProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <img
        src="/verus-logo.png"
        alt="Verus"
        style={{
          width: size,
          height: size,
          objectFit: 'contain',
          borderRadius: 7,
          display: 'block',
          flexShrink: 0,
        }}
      />
      {showWordmark && (
        <span style={{
          fontWeight: 800,
          fontSize: Math.round(size * 0.44),
          letterSpacing: '0.1em',
          color: wordmarkColor,
          fontFamily: 'Inter, -apple-system, sans-serif',
          lineHeight: 1,
        }}>
          VERUS
        </span>
      )}
    </div>
  );
}
