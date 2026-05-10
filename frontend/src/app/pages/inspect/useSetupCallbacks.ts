/**
 * useSetupCallbacks.ts — completeSetup and newProject callbacks.
 * Handles Supabase project create/update and resets workspace for a new project.
 * Does NOT manage analysis job state, canvas rendering, or routing.
 */

import { useCallback } from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from '../../../lib/supabase';
import { DEFAULT_ER } from './constants';
import type { ManufacturerKey } from './constants';

interface UseSetupCallbacksOptions {
  session:       Session | null;
  projectId:     string | null;
  manufacturer:  ManufacturerKey | '';
  frequencyMhz:  number;
  useCustomFreq: boolean;
  customFreq:    string;
  structureName: string;
  projectName:   string;
  bridgeId:      string;
  inspDate:      string;
  notes:         string;
  setProjectId:     (v: string | null) => void;
  setDielectricEr:  (v: number) => void;
  setSetupDone:     (v: boolean) => void;
  setManufacturer:  (v: ManufacturerKey | '') => void;
  setFrequencyMhz:  (v: number) => void;
  setSetupStep:     (v: 1|2|3) => void;
  setFiles:         (v: []) => void;
  setJobStatus:     (v: 'idle') => void;
  setAnalysisResult:(v: null) => void;
  setErrorMsg:      (v: null) => void;
}

export function useSetupCallbacks({
  session, projectId, manufacturer, frequencyMhz, useCustomFreq, customFreq,
  structureName, projectName, bridgeId, inspDate, notes,
  setProjectId, setDielectricEr, setSetupDone, setManufacturer, setFrequencyMhz,
  setSetupStep, setFiles, setJobStatus, setAnalysisResult, setErrorMsg,
}: UseSetupCallbacksOptions) {
  const completeSetup = useCallback(async () => {
    const effectiveFreq = useCustomFreq ? (parseInt(customFreq) || 1600) : frequencyMhz;
    if (session?.user?.id) {
      const payload = {
        name: projectName, structure_name: structureName, bridge_id: bridgeId || null,
        inspection_date: inspDate, notes: notes || null, manufacturer: manufacturer || null,
        frequency_mhz: effectiveFreq, inspection_method: 'GPR',
        updated_at: new Date().toISOString(),
      };
      try {
        if (projectId) {
          await supabase.from('projects').update(payload).eq('id', projectId).select().single();
        } else {
          const { data } = await supabase.from('projects')
            .insert({ user_id: session.user.id, ...payload }).select('id').single();
          if (data?.id) { setProjectId(data.id); localStorage.setItem('verus_project_id', data.id); }
        }
      } catch (err) { console.error('[setup] unexpected error:', err); }
    }
    setDielectricEr(DEFAULT_ER[effectiveFreq] ?? 6);
    setSetupDone(true);
  }, [manufacturer, frequencyMhz, useCustomFreq, customFreq, structureName, projectName,
      bridgeId, inspDate, notes, session, projectId, setProjectId, setDielectricEr, setSetupDone]);

  const newProject = useCallback(() => {
    localStorage.removeItem('verus_project_id');
    setManufacturer(''); setFrequencyMhz(1600); setProjectId(null);
    setSetupDone(false); setSetupStep(1);
    setFiles([]); setJobStatus('idle'); setAnalysisResult(null); setErrorMsg(null);
  }, [setManufacturer, setFrequencyMhz, setProjectId, setSetupDone, setSetupStep,
      setFiles, setJobStatus, setAnalysisResult, setErrorMsg]);

  return { completeSetup, newProject };
}
