import { useEffect, useRef, useState } from 'react';
import { ACCENT, BORDER, TEXT, TEXT2 } from './constants';

interface Props {
  structureName: string;
  estimatedSecs: number;
  fileCount:     number;
  fileFormat:    string;
  jobStatus: 'pending' | 'processing' | 'complete' | 'failed';
  errorMsg: string | null;
}

function formatTimeRemaining(seconds: number): string {
  if (seconds > 60) {
    const minutes = Math.ceil(seconds / 60);
    return `~${minutes} minute${minutes !== 1 ? 's' : ''} remaining`;
  } else if (seconds > 10) {
    return `${seconds} seconds remaining`;
  } else if (seconds > 0) {
    return 'Almost done...';
  } else {
    return 'Finalizing...';
  }
}

function getStatusMessage(elapsed: number, total: number, fileCount: number, fileFormat: string): string {
  const pct = elapsed / Math.max(total, 1);
  if (pct < 0.1) return `Uploading ${fileCount} ${fileFormat} file${fileCount !== 1 ? 's' : ''}...`;
  if (pct < 0.25) return 'Converting data format...';
  if (pct < 0.5) return 'Running AI delamination analysis...';
  if (pct < 0.7) return 'Computing rebar depth estimates...';
  if (pct < 0.85) return 'Generating condition maps...';
  return 'Almost done — finalizing report...';
}

export default function AnalysisProgressOverlay({
  structureName, estimatedSecs, fileCount, fileFormat, jobStatus, errorMsg,
}: Props) {
  const [elapsed, setElapsed] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isDone    = jobStatus === 'complete' || jobStatus === 'failed';
  const isSuccess = jobStatus === 'complete';

  useEffect(() => {
    if (isDone) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => setElapsed(prev => prev + 1), 1000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isDone]);

  const timeRemaining = Math.max(0, estimatedSecs - elapsed);

  const progress = isSuccess
    ? 100
    : Math.min(90, ((estimatedSecs - timeRemaining) / Math.max(estimatedSecs, 1)) * 90);

  const barColor = isSuccess ? '#22c55e' : isDone ? '#ef4444' : ACCENT;

  const timeLabel = isDone
    ? (isSuccess ? `Completed in ${elapsed}s` : 'See error below')
    : formatTimeRemaining(timeRemaining);

  const statusText = isDone
    ? (isSuccess ? 'Analysis complete!' : (errorMsg ?? 'Analysis failed'))
    : getStatusMessage(elapsed, estimatedSecs, fileCount, fileFormat);

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
            height: '100%', borderRadius: 3,
            background: barColor,
            width: `${progress}%`,
            transition: 'width 1s linear, background 0.3s ease',
          }} />
        </div>

        <div style={{ fontSize: 10, color: TEXT2, marginBottom: 24, textAlign: 'right' }}>
          {Math.round(progress)}%
        </div>

        <div style={{ fontSize: 12, color: isDone && !isSuccess ? '#ef4444' : TEXT2, minHeight: 18 }}>
          {statusText}
        </div>
      </div>
    </div>
  );
}
