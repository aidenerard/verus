import { useState } from 'react';
import { Download, FileImage, FileSpreadsheet } from 'lucide-react';
import { supabase } from '../../../lib/supabase';
import type { AnalysisResult } from '../inspect/types';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT } from './tokens';

function toSrc(url?: string | null, b64?: string | null): string | undefined {
  if (url) return url;
  if (!b64) return undefined;
  if (b64.startsWith('http') || b64.startsWith('data:')) return b64;
  return `data:image/png;base64,${b64}`;
}

function triggerDownload(href: string, filename: string): void {
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  a.target = '_blank';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

interface Props {
  result:     AnalysisResult;
  projectId?: string;
}

export default function ResultsExportTab({ result, projectId }: Props) {
  const [csvState, setCsvState] = useState<'idle' | 'loading' | 'empty' | 'error'>('idle');

  const baseName = (result.analysis_name ?? 'analysis').trim().replace(/[^\w-]+/g, '_') || 'analysis';
  const depthSrc      = toSrc(result.rebar_depth_map_url, result.rebar_depth_map ?? result.rebar_depth_image);
  const corrosionSrc  = toSrc(result.corrosion_map_url, result.corrosion_map);
  const dielectricSrc = toSrc(result.dielectric_map_url, result.dielectric_map);
  const quantities    = result.quantities ?? result.stats?.quantities ?? null;

  const downloadQuantitiesCsv = () => {
    if (!quantities) return;
    const rows = Object.entries(quantities)
      .map(([k, v]) => `${k},${typeof v === 'string' ? `"${v}"` : v}`);
    const csv = ['field,value', ...rows].join('\n');
    const href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    triggerDownload(href, `${baseName}_quantities.csv`);
    URL.revokeObjectURL(href);
  };

  const downloadPicksCsv = async () => {
    if (!projectId) return;
    setCsvState('loading');
    try {
      const { data, error } = await supabase
        .from('picks')
        .select('swath_idx,trace_idx,sample_idx,depth_in,confidence,is_manual')
        .eq('job_id', projectId)
        .order('swath_idx', { ascending: true })
        .order('trace_idx', { ascending: true });
      if (error) throw error;
      if (!data || data.length === 0) { setCsvState('empty'); return; }
      const header = 'swath_idx,trace_idx,sample_idx,depth_in,confidence,is_manual';
      const rows = data.map(p =>
        [p.swath_idx, p.trace_idx, p.sample_idx, p.depth_in, p.confidence, p.is_manual].join(','));
      const csv = [header, ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const href = URL.createObjectURL(blob);
      triggerDownload(href, `${baseName}_picks.csv`);
      URL.revokeObjectURL(href);
      setCsvState('idle');
    } catch {
      setCsvState('error');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <ExportRow
        Icon={FileImage}
        title="Rebar Depth Map (PNG)"
        subtitle="Unified depth map — fixed 2–9″ color scale"
        action={depthSrc ? () => triggerDownload(depthSrc, `${baseName}_rebar_depth.png`) : undefined}
        disabledNote={depthSrc ? undefined : 'Not available for this analysis'}
      />
      <ExportRow
        Icon={FileImage}
        title="Corrosion Risk Map (PNG)"
        subtitle="ASTM D6087 depth-corrected dB map"
        action={corrosionSrc ? () => triggerDownload(corrosionSrc, `${baseName}_corrosion.png`) : undefined}
        disabledNote={corrosionSrc ? undefined : 'Not available for this analysis'}
      />
      <ExportRow
        Icon={FileImage}
        title="Dielectric / Moisture Map (PNG)"
        subtitle="Per-trace dielectric (εr) — moisture proxy"
        action={dielectricSrc ? () => triggerDownload(dielectricSrc, `${baseName}_dielectric.png`) : undefined}
        disabledNote={dielectricSrc ? undefined : 'Requires metal plate calibration scan'}
      />
      <ExportRow
        Icon={FileSpreadsheet}
        title="Quantities (CSV)"
        subtitle="ASTM D6087 deck condition statistics"
        action={quantities ? downloadQuantitiesCsv : undefined}
        disabledNote={quantities ? undefined : 'Not available for this analysis'}
      />
      <ExportRow
        Icon={FileSpreadsheet}
        title="Rebar Picks (CSV)"
        subtitle={
          csvState === 'loading' ? 'Preparing…'
          : csvState === 'empty' ? 'No picks saved for this job yet'
          : csvState === 'error' ? 'Export failed — try again'
          : 'Per-trace depth picks from the interactive viewer'
        }
        action={projectId ? downloadPicksCsv : undefined}
        disabledNote={projectId ? undefined : 'Save the project first'}
      />
    </div>
  );
}

function ExportRow({ Icon, title, subtitle, action, disabledNote }: {
  Icon: typeof FileImage;
  title: string;
  subtitle: string;
  action?: () => void;
  disabledNote?: string;
}) {
  const enabled = !!action;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 16,
      background: PANEL, border: `1px solid ${BORDER}`, padding: '16px 20px',
    }}>
      <div style={{
        width: 40, height: 40, flexShrink: 0, background: RAISED,
        display: 'flex', alignItems: 'center', justifyContent: 'center', color: TEXT2,
      }}>
        <Icon size={18} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: TEXT }}>{title}</div>
        <div style={{ fontSize: 12, color: enabled ? TEXT2 : TEXT3, marginTop: 2 }}>
          {disabledNote ?? subtitle}
        </div>
      </div>
      <button
        onClick={action}
        disabled={!enabled}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 7, flexShrink: 0,
          padding: '9px 16px', border: 'none', fontFamily: 'inherit',
          fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
          color: enabled ? '#fff' : TEXT3,
          background: enabled ? ACCENT : RAISED,
          cursor: enabled ? 'pointer' : 'not-allowed',
        }}
      >
        <Download size={13} /> Download
      </button>
    </div>
  );
}
