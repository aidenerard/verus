import { ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router';
import { useAuth } from '../context/AuthContext';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import DashboardPage from './pages/DashboardPage';
import TeamPage from './pages/TeamPage';
import ModuleSelectPage from './pages/workspace/ModuleSelectPage';
import MethodSelectPage from './pages/workspace/MethodSelectPage';
import WorkspaceLayout from './pages/workspace/WorkspaceLayout';
import GPRWorkspace from './pages/workspace/GPRWorkspace';
import FDEMWorkspace from './pages/workspace/FDEMWorkspace';
import MagWorkspace from './pages/workspace/MagWorkspace';
import MASWWorkspace from './pages/workspace/MASWWorkspace';
import ImpactEchoWorkspace from './pages/workspace/ImpactEchoWorkspace';
import PrivacyPage from './pages/legal/PrivacyPage';
import TermsPage from './pages/legal/TermsPage';

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  if (isLoading) return null;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function RedirectKeepQuery({ to }: { to: string }) {
  const { search, hash } = useLocation();
  return <Navigate to={`${to}${search}${hash}`} replace />;
}

export default function Router() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"         element={<HomePage />} />
        <Route path="/login"    element={<LoginPage />} />
        <Route path="/signup"   element={<SignupPage />} />
        <Route path="/team"     element={<TeamPage />} />
        <Route path="/privacy"  element={<PrivacyPage />} />
        <Route path="/terms"    element={<TermsPage />} />

        <Route path="/dashboard" element={
          <ProtectedRoute><DashboardPage /></ProtectedRoute>
        } />

        {/* Pre-workspace selection flow (no sidebar) */}
        <Route path="/workspace" element={
          <ProtectedRoute><ModuleSelectPage /></ProtectedRoute>
        } />
        <Route path="/workspace/em" element={
          <ProtectedRoute><MethodSelectPage moduleId="em" /></ProtectedRoute>
        } />
        <Route path="/workspace/seismic" element={
          <ProtectedRoute><MethodSelectPage moduleId="seismic" /></ProtectedRoute>
        } />

        {/* Analysis workspaces — wrapped in WorkspaceLayout (top bar + breadcrumb only) */}
        <Route element={
          <ProtectedRoute><WorkspaceLayout /></ProtectedRoute>
        }>
          <Route path="/workspace/em/gpr"              element={<GPRWorkspace />} />
          <Route path="/workspace/em/fdem"             element={<FDEMWorkspace />} />
          <Route path="/workspace/em/magnetometer"     element={<MagWorkspace />} />
          <Route path="/workspace/seismic/masw"        element={<MASWWorkspace />} />
          <Route path="/workspace/seismic/impact-echo" element={<ImpactEchoWorkspace />} />
        </Route>

        {/* Legacy redirects — preserve query string for ?project_id=... links */}
        <Route path="/analyze"      element={<RedirectKeepQuery to="/workspace/em/gpr" />} />
        <Route path="/inspect/gpr"  element={<RedirectKeepQuery to="/workspace/em/gpr" />} />
        <Route path="/inspect/masw" element={<RedirectKeepQuery to="/workspace/seismic/masw" />} />
        <Route path="/inspect/ir"   element={<RedirectKeepQuery to="/workspace/seismic/impact-echo" />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
