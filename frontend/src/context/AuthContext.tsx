import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { User, Session } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

// ── Types ─────────────────────────────────────────────────────────────────────

interface AuthContextType {
  user:            User | null;
  session:         Session | null;
  isAuthenticated: boolean;
  isLoading:       boolean;

  login:  (email: string, password: string) => Promise<{ error: string | null }>;
  signup: (email: string, password: string, name: string, company?: string) => Promise<{ error: string | null }>;
  logout: () => Promise<void>;

  // Compatibility shim: components that read auth.user.name / auth.user.email
  auth: {
    isAuthenticated: boolean;
    user: { name: string; email: string } | null;
  };
}

const AuthContext = createContext<AuthContextType | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,      setUser]      = useState<User | null>(null);
  const [session,   setSession]   = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Restore existing session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setIsLoading(false);
    });

    // Keep session in sync across tabs / refreshes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setIsLoading(false);
      },
    );

    return () => subscription.unsubscribe();
  }, []);

  // ── Auth actions ────────────────────────────────────────────────────────────

  const login = async (
    email: string,
    password: string,
  ): Promise<{ error: string | null }> => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    return { error: error?.message ?? null };
  };

  const signup = async (
    email: string,
    password: string,
    name: string,
    company?: string,
  ): Promise<{ error: string | null }> => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { full_name: name, company: company ?? '' },
      },
    });
    if (error) return { error: error.message };

    // Create profile row (the SQL trigger handles this too, but belt-and-suspenders)
    if (data.user) {
      await supabase.from('profiles').upsert({
        id:        data.user.id,
        full_name: name,
        company:   company ?? '',
      });
    }

    return { error: null };
  };

  const logout = async () => {
    await supabase.auth.signOut();
  };

  // ── Compatibility shim (DashboardPage reads auth.user.name / auth.user.email) ─

  const isAuthenticated = !!user;

  const auth = isAuthenticated
    ? {
        isAuthenticated: true,
        user: {
          name:  (user?.user_metadata?.full_name as string | undefined)
                   ?? user?.email?.split('@')[0]
                   ?? 'User',
          email: user?.email ?? '',
        },
      }
    : { isAuthenticated: false, user: null };

  return (
    <AuthContext.Provider
      value={{ user, session, isAuthenticated, isLoading, login, signup, logout, auth }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
