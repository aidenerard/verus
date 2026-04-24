import { createClient } from '@supabase/supabase-js';

const supabaseUrl  = import.meta.env.VITE_SUPABASE_URL  as string;
const supabaseKey  = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || !supabaseKey) {
  console.warn(
    '[supabase] VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY is not set. ' +
    'Auth and database features will not work until you add them to .env.local. ' +
    'See SETUP.md for instructions.',
  );
}

export const supabase = createClient(supabaseUrl ?? '', supabaseKey ?? '');
