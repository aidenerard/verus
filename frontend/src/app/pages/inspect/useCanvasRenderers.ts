/**
 * useCanvasRenderers.ts
 * Manages all canvas refs and their rendering effects: B-scan, condition map,
 * rebar depth, and amplitude. Re-renders when analysis data or slider values change.
 *
 * Does NOT: handle Mapbox, job orchestration, file uploads, or UI state.
 */

import { useEffect, useRef, useCallback } from 'react';
import {
  decodeF32, renderConditionToCanvas, renderDepthToCanvas,
  renderConditionGrid, renderAmpGrid, renderRebarDepthCanvas, renderDepthColorbar,
} from './colormaps';
import { PANEL, TEXT2 } from './constants';
import type { AnalysisResult, OutputTab } from './types';

interface UseCanvasRenderersProps {
  analysisResult: AnalysisResult | null;
  selectedFileIdx: number;
  outputTab: OutputTab;
  detectionThreshold: number;
  dielectricEr: number;
  ampClampMin: number;
  ampClampMax: number;
}

interface UseCanvasRenderersReturn {
  bscanCanvasRef: React.RefObject<HTMLCanvasElement>;
  condCanvasRef: React.RefObject<HTMLCanvasElement>;
  rebarCanvasRef: React.RefObject<HTMLCanvasElement>;
  ampCanvasRef: React.RefObject<HTMLCanvasElement>;
  rebarColorbarRef: React.RefObject<HTMLCanvasElement>;
}

export function useCanvasRenderers({
  analysisResult,
  selectedFileIdx,
  outputTab,
  detectionThreshold,
  dielectricEr,
  ampClampMin,
  ampClampMax,
}: UseCanvasRenderersProps): UseCanvasRenderersReturn {
  const bscanCanvasRef   = useRef<HTMLCanvasElement>(null);
  const condCanvasRef    = useRef<HTMLCanvasElement>(null);
  const rebarCanvasRef   = useRef<HTMLCanvasElement>(null);
  const ampCanvasRef     = useRef<HTMLCanvasElement>(null);
  const rebarColorbarRef = useRef<HTMLCanvasElement>(null);

  // B-scan render
  const renderBscan = useCallback(() => {
    const canvas = bscanCanvasRef.current;
    if (!canvas || !analysisResult) return;
    const fileResult = analysisResult.per_file_summary[selectedFileIdx];
    if (!fileResult?.bscan) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = 400; canvas.height = 120;
        ctx.fillStyle = PANEL; ctx.fillRect(0, 0, 400, 120);
        ctx.fillStyle = TEXT2; ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('B-scan data not available for this file', 200, 65);
      }
      return;
    }
    const { data, n_traces, n_samples } = fileResult.bscan;
    const binary = atob(data);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const floats = new Float32Array(bytes.buffer);
    let minVal = Infinity, maxVal = -Infinity;
    for (const v of floats) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
    const range = maxVal - minVal || 1;
    canvas.width = n_traces; canvas.height = n_samples;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imageData = ctx.createImageData(n_traces, n_samples);
    for (let t = 0; t < n_traces; t++) {
      for (let s = 0; s < n_samples; s++) {
        const v   = Math.round(((floats[t * n_samples + s] - minVal) / range) * 255);
        const idx = (s * n_traces + t) * 4;
        imageData.data[idx] = v; imageData.data[idx+1] = v; imageData.data[idx+2] = v; imageData.data[idx+3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
    ctx.strokeStyle = '#ff3cff'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let t = 0; t < n_traces; t++) {
      let peakS = 0, peakA = -Infinity;
      for (let s = 0; s < n_samples; s++) {
        const a = Math.abs(floats[t * n_samples + s]);
        if (a > peakA) { peakA = a; peakS = s; }
      }
      t === 0 ? ctx.moveTo(t, peakS) : ctx.lineTo(t, peakS);
    }
    ctx.stroke();
  }, [analysisResult, selectedFileIdx]);

  useEffect(() => { renderBscan(); }, [renderBscan]);

  // Condition canvas
  useEffect(() => {
    const canvas = condCanvasRef.current;
    if (!canvas || !analysisResult) return;
    console.log('[canvas:cond] ref OK, canvas px:', canvas.width, 'x', canvas.height);
    const pgd = analysisResult.prob_grid_data;
    try {
      if (pgd && pgd.length > 0) {
        console.log('[canvas:cond] rendering grid', pgd.length, 'x', pgd[0]?.length);
        renderConditionGrid(canvas, pgd, detectionThreshold);
      } else if (analysisResult.prob_grid) {
        console.log('[canvas:cond] falling back to b64, rows/cols:', analysisResult.prob_grid_rows, analysisResult.prob_grid_cols);
        renderConditionToCanvas(canvas, decodeF32(analysisResult.prob_grid),
          analysisResult.prob_grid_rows!, analysisResult.prob_grid_cols!, detectionThreshold);
      } else {
        console.warn('[canvas:cond] no data available');
      }
    } catch (e) { console.error('[canvas:cond] render failed:', e); }
  }, [detectionThreshold, analysisResult, outputTab]);

  // Rebar canvas
  useEffect(() => {
    const canvas = rebarCanvasRef.current;
    if (!canvas || !analysisResult) return;
    const rdg = analysisResult.rebar_depth_grid;
    const pkg = analysisResult.rebar_peak_grid ?? null;

    const draw = () => {
      try {
        if (rdg && rdg.length > 0) {
          renderRebarDepthCanvas(canvas, rdg, pkg);
        } else if (analysisResult.twt_grid) {
          renderDepthToCanvas(canvas, decodeF32(analysisResult.twt_grid),
            analysisResult.twt_grid_rows!, analysisResult.twt_grid_cols!, dielectricEr);
        }
        const colorbar = rebarColorbarRef.current;
        if (colorbar) renderDepthColorbar(colorbar);
      } catch (e) { console.error('[canvas:rebar] render failed:', e); }
    };

    const rafId = requestAnimationFrame(draw);
    const obs   = new ResizeObserver(() => requestAnimationFrame(draw));
    obs.observe(canvas);
    return () => { cancelAnimationFrame(rafId); obs.disconnect(); };
  }, [dielectricEr, analysisResult, outputTab]);

  // Amplitude canvas
  useEffect(() => {
    const canvas = ampCanvasRef.current;
    const agd = analysisResult?.amplitude_grid_data;
    if (!canvas || !agd || agd.length === 0) return;
    try { renderAmpGrid(canvas, agd, ampClampMin, ampClampMax); }
    catch (e) { console.error('[canvas:amp] render failed:', e); }
  }, [ampClampMin, ampClampMax, analysisResult, outputTab]);

  return { bscanCanvasRef, condCanvasRef, rebarCanvasRef, ampCanvasRef, rebarColorbarRef };
}
