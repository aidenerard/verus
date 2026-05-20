import React from 'react';
import ReactDOM from 'react-dom/client';
import { AuthProvider } from './context/AuthContext';
import Router from './app/Router';
import './styles/index.css';

async function bootstrap() {
  if (import.meta.env.VITE_USE_MOCKS) {
    const { startMocks } = await import('./mocks/browser');
    await startMocks();
  }
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <AuthProvider>
        <Router />
      </AuthProvider>
    </React.StrictMode>,
  );
}

bootstrap();
