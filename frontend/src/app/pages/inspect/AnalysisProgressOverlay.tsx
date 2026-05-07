import { useEffect, useRef, useState } from 'react';
import { ACCENT, BORDER, TEXT, TEXT2 } from './constants';

interface Props {
  structureName: string;
  estimatedSecs: number;
  jobProgress:   number;
  jobStage:      string;
  jobStatus: 'pending' | 'processing' | 'complete' | 'failed';
  errorMsg: string | null;
}

function fmtCountdown(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function AnalysisProgressOverlay({
  structureName, estimatedSecs, jobProgress, jobStage, jobStatus, errorMsg,
}: Props) {
  const startRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  const isDone    = jobStatus === 'complete' || jobStatus === 'failed';
  const isSuccess = jobStatus === 'complete';

  useEffect(() => {
    if (isDone) return;
    const iv = setInterval(
      () => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)),
      1000,
    );
    return () => clearInterval(iv);
  }, [isDone]);

  // Honest remaining time: rate-based when we have real progress, else use estimate
  const remaining = (() => {
    if (isDone) return 0;
    if (jobProgress > 5 && elapsed > 0) {
      return Math.max(0, Math.round(elapsed * (100 - jobProgress) / jobProgress));
    }
    return Math.max(0, estimatedSecs - elapsed);
  })();

  const barPct    = isDone ? 100 : Math.max(0, jobProgress);
  const barColor  = isDone && !isSuccess ? '#ef4444' : ACCENT;
  const stageText = isDone
    ? (isSuccess ? 'Analysis complete!' : (errorMsg ?? 'Analysis failed'))
    : (jobStage || 'Connecting…');

  const timeLabel = isDone
    ? (isSuccess ? `Completed in ${elapsed}s` : 'See error below')
    : remaining > 0
      ? <>Time remaining: <strong style={{ color: TEXT }}>{fmtCountdown(remaining)}</strong></>
      : 'Finishing up…';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1001,
      background: 'rgba(0,0,0,0.82)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      <div style={{
        width: 480, background: '#FFFFFF', padding: '48px 52px',
        boxShadow: '0 32px 96px rgba(0,0,0,0.5)', textAlign: 'center',
      }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: TEXT, marginBottom: 8 }}>
          Analyzing {structureName}
        </div>
        <div style={{ fontSize: 12, color: TEXT2, marginBottom: 32 }}>
          {timeLabel}
        </div>

        <div style={{ height: 6, background: BORDER, borderRadius: 3, overflow: 'hidden', marginBottom: 8 }}>
          <div style={{
            height: '100%', borderRadius: 3, background: barColor,
            width: `${barPct}%`,
            transition: isDone ? 'width 0.3s ease' : 'width 0.9s ease',
          }} />
        </div>

        <div style={{ fontSize: 10, color: TEXT2, marginBottom: 24, textAlign: 'right' }}>
          {barPct}%
        </div>

        <div style={{ fontSize: 12, color: isDone && !isSuccess ? '#ef4444' : TEXT2, minHeight: 18 }}>
          {stageText}
        </div>
      </div>
    </div>
  );
}
