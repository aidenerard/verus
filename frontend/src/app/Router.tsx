import { ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router';
import { useAuth } from '../context/AuthContext';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import DashboardPage from './pages/DashboardPage';
import TeamPage from './pages/TeamPage';
import GPRWorkspace from './pages/inspect/GPRWorkspace';
import ComingSoonWorkspace from './pages/inspect/ComingSoonWorkspace';

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  if (isLoading) return null;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function Router() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"         element={<HomePage />} />
        <Route path="/login"    element={<LoginPage />} />
        <Route path="/signup"   element={<SignupPage />} />
        <Route path="/team"     element={<TeamPage />} />

        <Route path="/dashboard" element={
          <ProtectedRoute><DashboardPage /></ProtectedRoute>
        } />

        {/* Inspect workspaces */}
        <Route path="/inspect/gpr" element={
          <ProtectedRoute><GPRWorkspace /></ProtectedRoute>
        } />
        <Route path="/inspect/impact-echo" element={
          <ProtectedRoute><ComingSoonWorkspace method="impact-echo" /></ProtectedRoute>
        } />
        <Route path="/inspect/ir" element={
          <ProtectedRoute><ComingSoonWorkspace method="ir" /></ProtectedRoute>
        } />
        <Route path="/inspect/ras" element={
          <ProtectedRoute><ComingSoonWorkspace method="ras" /></ProtectedRoute>
        } />

        {/* Legacy redirect */}
        <Route path="/analyze" element={<Navigate to="/inspect/gpr" replace />} />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
