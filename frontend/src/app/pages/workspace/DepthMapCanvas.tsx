import { useCallback, useEffect, useState } from 'react';
import { Download, ImageOff, RotateCcw } from 'lucide-react';
import { supabase } from '../../../lib/supabase';
import { ACCENT, BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

interface DepthMapCanvasProps {
  jobId:           string;
  serverUrl:       string;
  analysisName:    string;
  staticImageB64?: string;
  needsRegen?:     boolean;
  onRegenerated?:  (newImageB64: string) => void;
}

interface RegenAck {
  rebar_depth_map: string;
  picks_used?:     number;
}

function asDataUrl(src: string | undefined | null): string | undefined {
  if (!src) return undefined;
  if (src.startsWith('data:') || src.startsWith('http')) return src;
  return `data:image/png;base64,${src}`;
}

function safeFilename(name: string): string {
  const cleaned = (name ?? '').replace(/[^\w\s-]/g, '').replace(/\s+/g, '_').trim();
  return cleaned || 'rebar_depth_map';
}

export default function DepthMapCanvas({
  jobId, serverUrl, analysisName,
  staticImageB64, needsRegen, onRegenerated,
}: DepthMapCanvasProps) {
  const [currentImage, setCurrentImage] = useState<string | null>(asDataUrl(staticImageB64) ?? null);
  const [regenerating, setRegenerating] = useState(false);
  const [regenMsg,     setRegenMsg]     = useState<string | null>(null);
  const [picksUsed,    setPicksUsed]    = useState<number | null>(null);

  useEffect(() => {
    const next = asDataUrl(staticImageB64);
    if (next && !currentImage) setCurrentImage(next);
  }, [staticImageB64, currentImage]);

  const regenerateMap = useCallback(async () => {
    if (!jobId) return;
    setRegenerating(true);
    setRegenMsg('Regenerating depth map from picks…');
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch(`${serverUrl}/job/${jobId}/regenerate`, {
        method:  'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`Regeneration failed: HTTP ${res.status}`);
      const data = await res.json() as RegenAck;
      const newSrc = asDataUrl(data.rebar_depth_map) ?? null;
      if (newSrc) setCurrentImage(newSrc);
      setPicksUsed(data.picks_used ?? null);
      setRegenMsg(`✓ Updated from ${data.picks_used ?? 0} picks`);
      onRegenerated?.(data.rebar_depth_map);
    } catch (e) {
      setRegenMsg(`✗ ${e instanceof Error ? e.message : 'Regeneration failed'}`);
    } finally {
      setRegenerating(false);
    }
  }, [jobId, serverUrl, onRegenerated]);

  // External trigger: parent flips needsRegen=true after the analyst saves picks.
  useEffect(() => {
    if (needsRegen && jobId && !regenerating) regenerateMap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsRegen, jobId]);

  const downloadMap = useCallback(() => {
    if (!currentImage) return;
    const a = document.createElement('a');
    a.href     = currentImage;
    a.download = `${safeFilename(analysisName)}_rebar_depth.png`;
    a.click();
  }, [currentImage, analysisName]);

  const noJobTitle = !jobId ? 'Save the project first to regenerate' : undefined;
  const msgIsErr   = regenMsg?.startsWith('✗');
  const msgIsOk    = regenMsg?.startsWith('✓');

  return (
    <div style={{
      background: PANEL, border: `1px solid ${BORDER}`,
      display: 'flex', flexDirection: 'column',
      minHeight: 400, overflow: 'hidden', borderRadius: 8,
    }}>
      <div style={{
        flexShrink: 0,
        padding: '8px 14px', borderBottom: `1px solid ${BORDER}`,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{
          fontSize: 10, fontWeight: 800, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: TEXT2,
        }}>
          Rebar Depth Map
        </span>
        <span style={{ flex: 1, fontSize: 11, color: TEXT3, fontStyle: 'italic' }}>
          {picksUsed != null && picksUsed > 0
            ? `Generated from ${picksUsed} analyst pick${picksUsed === 1 ? '' : 's'}`
            : 'Automated analysis output'}
        </span>

        <button
          onClick={regenerateMap}
          disabled={regenerating || !jobId}
          title={noJobTitle}
          style={iconBtn(regenerating || !jobId)}
        >
          <RotateCcw size={11} /> {regenerating ? 'Regenerating…' : 'Regenerate'}
        </button>

        <button
          onClick={downloadMap}
          disabled={!currentImage}
          style={iconBtn(!currentImage)}
          title="Download PNG"
        >
          <Download size={11} /> PNG
        </button>
      </div>

      {regenMsg && (
        <div style={{
          flexShrink: 0, padding: '6px 14px', fontSize: 11,
          color:      msgIsOk  ? '#15803d' : msgIsErr ? '#b91c1c' : TEXT2,
          background: msgIsOk  ? '#f0fdf4' : msgIsErr ? '#fef2f2' : RAISED,
          borderBottom: `1px solid ${BORDER}`,
        }}>
          {regenMsg}
        </div>
      )}

      <div style={{
        flex: 1, minHeight: 0,
        position: 'relative', overflow: 'hidden',
        background: RAISED,
      }}>
        {regenerating && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 2,
            background: 'rgba(255,255,255,0.78)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: 10, color: TEXT2, fontSize: 13,
          }}>
            <span style={{
              width: 14, height: 14, border: `2px solid ${BORDER}`,
              borderTopColor: ACCENT, borderRadius: '50%',
              animation: 'depthmap-spin 0.8s linear infinite',
            }} />
            Regenerating depth map…
          </div>
        )}

        {currentImage ? (
          <img
            src={currentImage}
            alt="Rebar Depth Map"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'fill', display: 'block' }}
          />
        ) : (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: TEXT3 }}>
            <ImageOff size={22} />
            <span style={{ fontSize: 11 }}>No depth map yet</span>
          </div>
        )}
      </div>

      <style>{`
        @keyframes depthmap-spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

function iconBtn(disabled: boolean): React.CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: '5px 10px', fontSize: 10, fontWeight: 700,
    letterSpacing: '0.08em', textTransform: 'uppercase',
    border: `1px solid ${BORDER}`, borderRadius: 4,
    background: disabled ? RAISED : PANEL,
    color: disabled ? TEXT3 : TEXT,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'inherit',
  };
}
