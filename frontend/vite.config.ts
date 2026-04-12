import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // In dev mode, proxy API calls to the local Python server so you don't
  // need to change the API URL in App.tsx.  The production build uses the
  // hardcoded PYTHON_SERVER_URL in App.tsx.
  server: {
    proxy: {
      '/health':  'http://localhost:10000',
      '/analyze': 'http://localhost:10000',
      '/memory':  'http://localhost:10000',
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
