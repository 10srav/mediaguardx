import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// Demo mode when Supabase is not configured
export const isDemoMode =
  !supabaseUrl ||
  !supabaseAnonKey ||
  supabaseUrl.includes('your-project') ||
  supabaseAnonKey === 'your-anon-key-here';

if (isDemoMode) {
  console.info(
    '%c[MediaGuardX] Running in DEMO mode — Supabase not configured. Auth uses local storage.',
    'color: #6366f1; font-weight: bold',
  );
}

// In demo mode, create a proxy that throws descriptive errors if supabase
// methods are accidentally called without an isDemoMode guard.
const demoProxy = new Proxy({} as ReturnType<typeof createClient>, {
  get(_target, prop) {
    if (prop === 'auth') {
      return new Proxy(
        {},
        {
          get(_t, authProp) {
            return () => {
              throw new Error(
                `Supabase not configured: cannot call auth.${String(authProp)}() in demo mode. ` +
                  'Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env to enable live auth.',
              );
            };
          },
        },
      );
    }
    return () => {
      throw new Error(
        `Supabase not configured: cannot call ${String(prop)}() in demo mode. ` +
          'Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env to enable live auth.',
      );
    };
  },
});

export const supabase = isDemoMode
  ? demoProxy
  : createClient(supabaseUrl, supabaseAnonKey);
