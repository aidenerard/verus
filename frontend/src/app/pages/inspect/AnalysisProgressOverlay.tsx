import { useEffect, useState } from 'react';
import { ACCENT, TEXT, TEXT2 } from './constants';

const PROGRESS_MSGS = [
  'Uploading files…',
  'Converting data format…',
  'Running AI analysis…',
  'Generating condition map…',
  'Finalizing report…',
];

interface Props {
  fileCount: number;
  estimatedSecs: number;
  jobStatus: 'pending' | 'processing' | 'complete' | 'failed';
  errorMsg: string | null;
}

export default function AnalysisProgressOverlay({ fileCount, estimatedSecs, jobStatus, errorMsg }: Props) {
  const [barPct,       setBarPct]       = useState(0);
  const [msgIdx,       setMsgIdx]       = useState(0);
  const [pastEstimate, setPastEstimate] = useState(false);

  const isDone = jobStatus === 'complete' || jobStatus === 'failed';

  useEffect(() => {
    const t = setTimeout(() => setBarPct(90), 80);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (isDone) setBarPct(100);
  }, [isDone]);

  useEffect(() => {
    const t = setTimeout(() => setPastEstimate(true), estimatedSecs * 1000);
    return () => clearTimeout(t);
  }, [estimatedSecs]);

  useEffect(() => {
    if (isDone) return;
    const iv = setInterval(() => setMsgIdx(i => Math.min(i + 1, PROGRESS_MSGS.length - 1)), 4000);
    return () => clearInterval(iv);
  }, [isDone]);

  const timeLabel = estimatedSecs >= 60
    ? `~${Math.ceil(estimatedSecs / 60)} min`
    : `~${estimatedSecs}s`;

  const currentMsg = isDone
    ? (jobStatus === 'complete' ? 'Analysis complete!' : errorMsg || 'Analysis failed')
    : pastEstimate
    ? 'Almost done…'
    : PROGRESS_MSGS[msgIdx];

  const barTransition = isDone
    ? 'width 0.4s ease'
    : `width ${estimatedSecs}s linear`;

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1001,
      background: 'rgba(0,0,0,0.85)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      <div style={{
        width: 480, background: '#FFFFFF', padding: '48px 52px',
        boxShadow: '0 32px 96px rgba(0,0,0,0.5)', textAlign: 'center',
      }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: TEXT, marginBottom: 8 }}>
          Analyzing Bridge Deck
        </div>
        <div style={{ fontSize: 12, color: TEXT2, marginBottom: 36 }}>
          {fileCount} file{fileCount !== 1 ? 's' : ''} · Estimated time: {timeLabel}
        </div>

        <div style={{ height: 6, background: '#E2DED9', borderRadius: 3, overflow: 'hidden', marginBottom: 18 }}>
          <div style={{
            height: '100%', borderRadius: 3, background: ACCENT,
            width: `${barPct}%`, transition: barTransition,
          }} />
        </div>

        <div style={{ fontSize: 12, color: TEXT2, minHeight: 18 }}>{currentMsg}</div>
      </div>
    </div>
  );
}
