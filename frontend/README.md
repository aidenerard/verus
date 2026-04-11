# Verus Frontend

React + TypeScript + Tailwind CSS frontend for the Verus bridge inspection platform.

## Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS v4** for styling
- **Vite** for build tooling
- **Lucide React** for icons
- **Recharts** for data visualization

## Development

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev
```

## Features

- **Multi-method inspection platform** with sidebar navigation
- **GPR Analysis Module** (available)
  - CSV file upload (drag & drop or file picker)
  - Real-time analysis with PyTorch CNN model
  - C-scan visualization
  - Detailed per-file reports
  - PDF export (planned)
  
- **Additional modules** (in development):
  - Impact-Echo Acoustic Testing
  - Infrared Thermography
  - Automated Sounding

## API Integration

Frontend connects to FastAPI backend at: `https://verus-0j5j.onrender.com`

Endpoints:
- `GET /health` - Server health check
- `POST /analyze` - Upload CSV files for GPR analysis

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── App.tsx           # Main application component
│   │   └── components/       # React components
│   │       ├── ui/           # Reusable UI components
│   │       └── figma/        # Figma-imported components
│   ├── styles/
│   │   ├── theme.css         # Design tokens
│   │   └── fonts.css         # Font imports
│   └── imports/              # Static assets
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Design System

Professional engineering aesthetic with:
- Navy accents (#1a2f5a)
- Clean white backgrounds
- ASTM standard compliance indicators
- Responsive layout
