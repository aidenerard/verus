import { ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router';
import { useAuth } from '../context/AuthContext';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import DashboardPage from './pages/DashboardPage';
import TeamPage from './pages/TeamPage';
import WorkspaceLayout from './pages/workspace/WorkspaceLayout';
import EMModule from './pages/workspace/EMModule';
import SeismicModule from './pages/workspace/SeismicModule';
import GPRWorkspace from './pages/workspace/GPRWorkspace';
import FDEMWorkspace from './pages/workspace/FDEMWorkspace';
import MagWorkspace from './pages/workspace/MagWorkspace';
import MASWWorkspace from './pages/workspace/MASWWorkspace';
import ImpactEchoWorkspace from './pages/workspace/ImpactEchoWorkspace';

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

        <Route path="/dashboard" element={
          <ProtectedRoute><DashboardPage /></ProtectedRoute>
        } />

        <Route path="/workspace" element={
          <ProtectedRoute><WorkspaceLayout /></ProtectedRoute>
        }>
          <Route index                  element={<Navigate to="em/gpr" replace />} />
          <Route path="em"              element={<EMModule />} />
          <Route path="em/gpr"          element={<GPRWorkspace />} />
          <Route path="em/fdem"         element={<FDEMWorkspace />} />
          <Route path="em/magnetometer" element={<MagWorkspace />} />
          <Route path="seismic"         element={<SeismicModule />} />
          <Route path="seismic/masw"    element={<MASWWorkspace />} />
          <Route path="seismic/impact-echo" element={<ImpactEchoWorkspace />} />
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
