import { useState, useRef, useEffect } from 'react';
import { Upload, Check, X, Loader2, Download, Circle, Radio, Waves, Thermometer, Speaker } from 'lucide-react';
import PlanView from './components/PlanView';
import ThreeDView from './components/ThreeDView';

// In dev mode (npm run dev), VITE_API_URL is set to '' in .env.local so all
// fetch calls are relative (e.g. /health) and the Vite proxy forwards them to
// http://localhost:10000.  In production the env var is not set, so it falls
// back to the Render URL.
const STATUS_MESSAGES = [
  'Waking up server...',
  'Loading AI model...',
  'Processing GPR signals...',
  'Generating C-scan map...',
];

const PYTHON_SERVER_URL =
  import.meta.env.VITE_API_URL !== undefined
    ? import.meta.env.VITE_API_URL          // '' in dev → relative URLs
    : 'https://verus-server.onrender.com';  // production fallback

type InspectionMethod = 'gpr' | 'impact-echo' | 'infrared' | 'sounding';

interface InspectionModule {
  id: InspectionMethod;
  name: string;
  fullName: string;
  description: string;
  status: 'available' | 'in-development';
  icon: any;
  input: string;
  output: string;
  standard: string;
  detectionDescription: string;
}

const INSPECTION_MODULES: InspectionModule[] = [
  {
    id: 'gpr',
    name: 'GPR',
    fullName: 'Ground-Penetrating Radar',
    description: '',
    status: 'available',
    icon: Radio,
    input: 'A-scan waveforms',
    output: 'C-scan map + report',
    standard: 'ASTM D6087',
    detectionDescription: 'Detects subsurface delamination from electromagnetic reflection patterns in GPR A-scan waveforms.',
  },
  {
    id: 'impact-echo',
    name: 'Impact-Echo',
    fullName: 'Impact-Echo Acoustic Testing',
    description: 'Dedicated AI model for time-domain acoustic waveform classification.',
    status: 'in-development',
    icon: Waves,
    input: 'Time-domain waveforms',
    output: 'Condition map + report',
    standard: 'ASTM C1383',
    detectionDescription: 'Detects voids, delamination, and thickness anomalies from stress wave reflections in time-domain acoustic waveforms.',
  },
  {
    id: 'infrared',
    name: 'Infrared Thermography',
    fullName: 'Infrared Thermography (IR)',
    description: 'Dedicated AI model for thermal anomaly detection and delamination screening.',
    status: 'in-development',
    icon: Thermometer,
    input: 'Thermal imagery',
    output: 'Thermal map + report',
    standard: 'ASTM D4788',
    detectionDescription: 'Detects subsurface delamination and moisture intrusion from thermal gradient patterns in infrared imagery.',
  },
  {
    id: 'sounding',
    name: 'Automated Sounding',
    fullName: 'Rapid Automated Sounding (RAS)',
    description: 'Dedicated AI model for acoustic sounding signal classification and deck condition mapping.',
    status: 'in-development',
    icon: Speaker,
    input: 'Acoustic signals',
    output: 'Condition map + report',
    standard: 'ASTM D4580',
    detectionDescription: 'Detects hollow or delaminated areas from acoustic response patterns in automated chain drag or hammer sounding signals.',
  },
];

interface UploadedFile {
  file: File;
  name: string;
}

interface AnalysisResult {
  signals_analyzed: number;
  delamination_pct: number;
  sound_pct: number;
  analysis_time_sec: number;
  cscan_image: string; // base64 encoded PNG
  per_file_summary: FileResult[];
}

interface GpsData {
  lat_start: number;
  lon_start: number;
  lat_end: number;
  lon_end: number;
  coordinates: [number, number][];
}

interface FileResult {
  filename: string;
  signals: number;
  delam_pct: number;
  gps?: GpsData | null;
}

export default function App() {
  const [activeMethod, setActiveMethod] = useState<InspectionMethod>('gpr');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'warming' | 'offline'>('offline');
  const [slowRequest, setSlowRequest] = useState(false);
  const [statusMsgIndex, setStatusMsgIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [notifyEmail, setNotifyEmail] = useState('');
  const [activeResultTab, setActiveResultTab] = useState<'cscan' | 'plan' | '3d'>('cscan');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${PYTHON_SERVER_URL}/health`, {
          signal: AbortSignal.timeout(90000), // 90 second timeout for Render cold start
        });

        if (response.ok) {
          const data = await response.json();
          if (data.status === 'ok') {
            setServerStatus('online');
          } else {
            setServerStatus('warming');
          }
        } else {
          setServerStatus('warming');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setServerStatus('warming');
      }
    };

    checkHealth();
  }, []);

  // Cycle status messages every 4 seconds while analyzing
  useEffect(() => {
    if (!isAnalyzing) {
      setStatusMsgIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setStatusMsgIndex(i => (i + 1) % STATUS_MESSAGES.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      addFiles(selectedFiles);
    }
  };

  const GPR_EXTENSIONS = new Set([
    '.csv', '.dzt', '.dt1', '.rd3', '.rd7', '.segy', '.sgy',
    // companion / metadata
    '.dzg', '.hd', '.rad',
  ]);

  const addFiles = (newFiles: File[]) => {
    const accepted = newFiles.filter(file => {
      const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
      return GPR_EXTENSIONS.has(ext);
    });

    if (accepted.length === 0 && newFiles.length > 0) {
      alert('No supported GPR files found. Supported formats: CSV, DZT, DT1, RD3, RD7, SEG-Y (and companion .dzg / .hd / .rad files).');
      return;
    }

    if (accepted.length !== newFiles.length) {
      alert(`Added ${accepted.length} file(s). ${newFiles.length - accepted.length} unsupported file(s) were skipped.`);
    }

    const uploadedFiles: UploadedFile[] = accepted.map(file => ({
      file,
      name: file.name,
    }));

    setFiles(prev => [...prev, ...uploadedFiles]);
    setErrorMsg(null);
  };

  const removeFile = (fileName: string) => {
    setFiles(prev => prev.filter(f => f.name !== fileName));
    setErrorMsg(null);
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setSlowRequest(false);
    setErrorMsg(null);

    // Set slow request warning after 5 seconds
    const slowTimeout = setTimeout(() => {
      setSlowRequest(true);
    }, 5000);

    try {
      // Create FormData with all files
      const formData = new FormData();
      files.forEach(f => {
        formData.append('files', f.file);
      });

      console.log('[ANALYZE] Sending request to:', `${PYTHON_SERVER_URL}/analyze`);
      console.log('[ANALYZE] Number of files:', files.length);

      const response = await fetch(`${PYTHON_SERVER_URL}/analyze`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(300000), // 5 minute timeout (increased from 60s)
      });

      clearTimeout(slowTimeout);

      console.log('[ANALYZE] Response status:', response.status);
      console.log('[ANALYZE] Response OK:', response.ok);

      if (!response.ok) {
        // Handle 503 hibernate-wake-error from Render
        if (response.status === 503) {
          setErrorMsg('Server is waking up from sleep (Render free tier). This can take up to 2 minutes — wait, then try again.');
          setIsAnalyzing(false);
          setSlowRequest(false);
          setServerStatus('warming');
          return;
        }

        let httpErrorMsg = 'Unknown error';
        try {
          const errJson = await response.json();
          console.error('[ANALYZE] Error response:', errJson);
          httpErrorMsg = errJson.detail || errJson.error || errJson.stderr || JSON.stringify(errJson);
        } catch {
          const errorText = await response.text();
          console.error('[ANALYZE] Error text:', errorText);
          httpErrorMsg = errorText || `HTTP ${response.status} error`;
        }
        console.error('Analysis failed:', httpErrorMsg);
        setErrorMsg(`Analysis failed: ${httpErrorMsg}`);
        setIsAnalyzing(false);
        setSlowRequest(false);
        return;
      }

      const result: AnalysisResult = await response.json();
      console.log('Analysis result:', result);

      setAnalysisResult(result);
      setIsAnalyzing(false);
      setSlowRequest(false);
      setServerStatus('online'); // Server is clearly online after successful response

    } catch (error) {
      clearTimeout(slowTimeout);
      console.error('Error during analysis:', error);

      if (error instanceof Error && error.name === 'TimeoutError') {
        setErrorMsg('Analysis timed out after 5 minutes. The server may still be starting up — wait 90 seconds and try again, or upload fewer files.');
      } else if (error instanceof TypeError && error.message.includes('fetch')) {
        setErrorMsg('Cannot connect to server. The server may be asleep (Render free tier) — wait 2 minutes and try again.');
        setServerStatus('warming');
      } else if (error instanceof Error) {
        setErrorMsg(`Analysis failed: ${error.message}`);
      } else {
        setErrorMsg('Analysis failed. Please check your connection and try again.');
      }

      setIsAnalyzing(false);
      setSlowRequest(false);
    }
  };

  const activeModule = INSPECTION_MODULES.find(m => m.id === activeMethod)!;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#FFFFFF' }}>
      {/* Professional Header */}
      <header className="border-b-2" style={{ borderColor: '#E8601C', background: '#FFFFFF' }}>
        <div className="px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src="/verus-logo.png"
              alt="Verus"
              style={{ width: 40, height: 40, objectFit: 'contain', borderRadius: 6, display: 'block' }}
            />
            <div>
              <h1 className="text-2xl font-bold" style={{ color: '#0A0A0A', letterSpacing: '-0.01em' }}>
                VERUS
              </h1>
              <p className="text-xs" style={{ color: '#7A7470' }}>
                Bridge Deck Inspection Intelligence
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs font-semibold" style={{ color: '#7A7470' }}>Multi-Standard Compliant</p>
          </div>
        </div>
      </header>

      {/* Main Content with Sidebar */}
      <div className="flex flex-1">
        {/* Left Sidebar - Inspection Method Selector */}
        <aside className="w-80 border-r-2 p-6" style={{ background: '#F5F3EF', borderColor: '#E2DED9' }}>
          <h2 className="text-xs font-bold uppercase tracking-wide mb-4" style={{ color: '#7A7470' }}>
            Inspection Method
          </h2>
          <div className="space-y-3">
            {INSPECTION_MODULES.map((module) => {
              const Icon = module.icon;
              const isActive = activeMethod === module.id;

              return (
                <button
                  key={module.id}
                  onClick={() => setActiveMethod(module.id)}
                  className="w-full text-left p-4 border-2 transition-all"
                  style={{
                    background: '#FFFFFF',
                    borderColor: isActive ? '#0A0A0A' : '#E2DED9',
                    borderLeftWidth: isActive ? '3px' : '2px',
                  }}
                >
                  <div className="flex items-start gap-3">
                    <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: '#0A0A0A' }} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <h3 className="text-sm font-bold" style={{ color: '#0A0A0A' }}>
                          {module.name}
                        </h3>
                        <span
                          className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                          style={{
                            background: module.status === 'available' ? '#2E7D32' : '#E8EDF5',
                            color: module.status === 'available' ? '#FFFFFF' : '#0A0A0A',
                          }}
                        >
                          {module.status === 'available' ? 'Available' : 'In Development'}
                        </span>
                      </div>
                      <p className="text-[10px] mb-1" style={{ color: '#7A7470' }}>
                        {module.fullName}
                      </p>
                      {module.description && (
                        <p className="text-[11px] mt-2" style={{ color: '#6B7280' }}>
                          {module.description}
                        </p>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main Panel */}
        <div className="flex-1 p-8">
          {activeMethod === 'gpr' ? (
            <div className="w-full">
              {!analysisResult ? (
                /* ── Pre-analysis: centered compact stack ── */
                <div className="flex justify-center">
                  <div className="w-full space-y-6" style={{ maxWidth: 600 }}>

                    {/* Upload Section */}
                    <div className="border-2" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>
                      <div className="px-6 py-4 border-b-2" style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                        <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#0A0A0A' }}>
                          Data Upload
                        </h2>
                      </div>
                      <div className="p-6">
                        <div
                          className={`border-2 border-dashed cursor-pointer transition-all ${
                            isDragging ? 'border-[#E8601C] bg-[#FFF4EE]' : 'border-[#E2DED9] bg-[#F5F3EF]'
                          }`}
                          style={{ padding: '32px' }}
                          onDragOver={handleDragOver}
                          onDragLeave={handleDragLeave}
                          onDrop={handleDrop}
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <div className="flex flex-col items-center gap-3">
                            <Upload className="w-10 h-10" style={{ color: '#7A7470' }} />
                            <div className="text-center">
                              <p className="text-sm font-semibold" style={{ color: '#0A0A0A' }}>
                                Upload GPR Data Files
                              </p>
                              <p className="text-xs mt-1" style={{ color: '#7A7470' }}>
                                CSV, DZT, DT1, RD3, SEG-Y supported • Batch upload supported
                              </p>
                              <p className="text-xs mt-1" style={{ color: '#9CA3AF', fontStyle: 'italic' }}>
                                Include companion files (.dzg, .hd, .rad) for GPS and header data
                              </p>
                            </div>
                          </div>
                        </div>
                        <input
                          ref={fileInputRef}
                          type="file"
                          multiple
                          accept=".csv,.dzt,.DZT,.dt1,.DT1,.rd3,.RD3,.rd7,.RD7,.segy,.sgy,.SEG-Y,.SGY,.dzg,.DZG,.hd,.HD,.rad,.RAD"
                          onChange={handleFileSelect}
                          className="hidden"
                        />

                        {/* File List */}
                        {files.length > 0 && (
                          <div className="mt-4 space-y-2">
                            {files.map((file) => (
                              <div
                                key={file.name}
                                className="flex items-center justify-between p-3 border"
                                style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}
                              >
                                <div className="flex items-center gap-3 flex-1 min-w-0">
                                  <Check className="w-4 h-4 flex-shrink-0" style={{ color: '#2ECC71' }} />
                                  <div className="min-w-0 flex-1">
                                    <p className="text-sm font-medium truncate" style={{ color: '#0A0A0A' }}>
                                      {file.name}
                                    </p>
                                    <p className="text-xs" style={{ color: '#7A7470' }}>
                                      {(file.file.size / 1024).toFixed(1)} KB
                                    </p>
                                  </div>
                                </div>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    removeFile(file.name);
                                  }}
                                  className="p-1 hover:bg-gray-100"
                                >
                                  <X className="w-4 h-4" style={{ color: '#7A7470' }} />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Analysis Configuration */}
                    <div className="border-2" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>
                      <div className="px-6 py-4 border-b-2" style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                        <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#0A0A0A' }}>
                          Analysis Configuration
                        </h2>
                      </div>
                      <div className="p-6 space-y-4">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="text-xs font-bold uppercase tracking-wide" style={{ color: '#7A7470' }}>
                              AI Model Version
                            </label>
                            <div className="flex items-center gap-2">
                              <Circle
                                className="w-2 h-2"
                                fill={serverStatus === 'online' ? '#2ECC71' : serverStatus === 'warming' ? '#F39C12' : '#E74C3C'}
                                stroke="none"
                              />
                              <span className="text-xs" style={{ color: '#7A7470' }}>
                                {serverStatus === 'online' ? 'Online' : serverStatus === 'warming' ? 'Server warming up' : 'Offline'}
                              </span>
                            </div>
                          </div>
                          <div className="px-4 py-3 border-2" style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                            <p className="text-sm font-medium" style={{ color: '#0A0A0A' }}>
                              model_v13.pth
                            </p>
                          </div>
                          <p className="text-xs mt-2" style={{ color: '#7A7470' }}>
                            Latest model automatically selected
                          </p>
                        </div>

                        <div className="pt-4 border-t" style={{ borderColor: '#E2DED9' }}>
                          <div className="flex justify-between text-xs mb-1">
                            <span style={{ color: '#7A7470' }}>Detection Threshold:</span>
                            <span className="font-bold" style={{ color: '#0A0A0A' }}>0.65 (Optimal)</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span style={{ color: '#7A7470' }}>Analysis Standard:</span>
                            <span className="font-bold" style={{ color: '#0A0A0A' }}>ASTM D6087</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Run Analysis Button */}
                    <button
                      onClick={handleAnalyze}
                      disabled={files.length === 0 || isAnalyzing}
                      className="w-full py-5 text-base font-bold uppercase tracking-widest transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        background: files.length === 0 || isAnalyzing ? '#E2DED9' : '#E8601C',
                        color: files.length === 0 || isAnalyzing ? '#7A7470' : '#FFFFFF',
                        border: files.length === 0 || isAnalyzing ? '2px solid #E2DED9' : '2px solid #E8601C',
                      }}
                    >
                      {isAnalyzing ? (
                        <div className="flex items-center justify-center gap-2">
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Analyzing {files.length} {files.length === 1 ? 'file' : 'files'}...</span>
                        </div>
                      ) : (
                        'Run Analysis'
                      )}
                    </button>

                    {/* Inline error card — replaces browser alert() */}
                    {errorMsg && !isAnalyzing && (
                      <div className="flex items-start gap-2 p-3"
                           style={{ background: '#FFF5F5', border: '1.5px solid #E53E3E', borderRadius: '4px' }}>
                        <X className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: '#E53E3E' }} />
                        <p className="text-xs leading-relaxed" style={{ color: '#742A2A' }}>
                          {errorMsg}
                        </p>
                        <button onClick={() => setErrorMsg(null)} className="ml-auto flex-shrink-0"
                                style={{ color: '#E53E3E', lineHeight: 1 }}>
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    )}

                    {/* Loading state */}
                    {isAnalyzing && (
                      <div className="space-y-2">
                        <style>{`
                          @keyframes verus-bar {
                            0%   { transform: translateX(-100%); }
                            100% { transform: translateX(350%); }
                          }
                        `}</style>
                        <p className="text-xs text-center font-medium" style={{ color: '#7A7470' }}>
                          {STATUS_MESSAGES[statusMsgIndex]}
                        </p>
                        <div style={{
                          width: '100%',
                          height: '6px',
                          borderRadius: '3px',
                          background: '#E2DED9',
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            height: '100%',
                            width: '40%',
                            borderRadius: '3px',
                            background: '#0A0A0A',
                            animation: 'verus-bar 1.8s ease-in-out infinite',
                          }} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* ── Post-analysis: full-width results ── */
                <div className="space-y-6">

                  {/* Stats Bar — always visible across all tabs */}
                  <div className="grid grid-cols-4 gap-4">
                    <StatCard
                      label="Signals Analyzed"
                      value={analysisResult.signals_analyzed.toLocaleString()}
                    />
                    <StatCard
                      label="Delamination"
                      value={`${analysisResult.delamination_pct.toFixed(1)}%`}
                      alert
                    />
                    <StatCard
                      label="Sound"
                      value={`${analysisResult.sound_pct.toFixed(1)}%`}
                    />
                    <StatCard
                      label="Analysis Time"
                      value={`${analysisResult.analysis_time_sec.toFixed(1)}s`}
                    />
                  </div>

                  {/* Tabbed visualization card */}
                  <div className="border-2" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>

                    {/* Tab bar */}
                    <div className="flex items-center justify-between border-b-2 px-2"
                         style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                      <div className="flex">
                        {([
                          { key: 'cscan', label: 'C-Scan Map'  },
                          { key: 'plan',  label: 'Plan View'   },
                          { key: '3d',    label: '3D View'     },
                        ] as const).map(({ key, label }) => (
                          <button
                            key={key}
                            onClick={() => setActiveResultTab(key)}
                            style={{
                              padding: '12px 20px',
                              fontSize: 11,
                              fontWeight: 700,
                              textTransform: 'uppercase',
                              letterSpacing: '0.06em',
                              background: 'transparent',
                              border: 'none',
                              borderBottom: activeResultTab === key
                                ? '2px solid #E8601C'
                                : '2px solid transparent',
                              color: activeResultTab === key ? '#E8601C' : '#7A7470',
                              cursor: 'pointer',
                            }}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <div className="flex items-center gap-3 pr-2">
                        {activeResultTab === 'cscan' && analysisResult.cscan_image && (
                          <a
                            href={`data:image/png;base64,${analysisResult.cscan_image}`}
                            download="cscan_map.png"
                            className="px-4 py-2 text-xs font-bold uppercase tracking-wide flex items-center gap-2"
                            style={{ background: '#E8601C', color: '#FFFFFF' }}
                          >
                            <Download className="w-4 h-4" />
                            Export PNG
                          </a>
                        )}
                        <button
                          onClick={() => {
                            setAnalysisResult(null);
                            setFiles([]);
                            setActiveResultTab('cscan');
                          }}
                          className="px-4 py-2 text-xs font-bold uppercase tracking-wide"
                          style={{ border: '2px solid #E2DED9', color: '#7A7470', background: '#FFFFFF' }}
                        >
                          New Analysis
                        </button>
                      </div>
                    </div>

                    {/* C-Scan tab */}
                    {activeResultTab === 'cscan' && (
                      <div className="p-4">
                        {analysisResult.cscan_image ? (
                          <img
                            src={`data:image/png;base64,${analysisResult.cscan_image}`}
                            alt="C-Scan Condition Map"
                            className="w-full"
                            style={{ minWidth: 900, display: 'block' }}
                          />
                        ) : (
                          <div className="w-full h-64 flex items-center justify-center border-2 border-dashed"
                               style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                            <p className="text-sm" style={{ color: '#7A7470' }}>
                              C-scan visualization not available
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Plan View tab */}
                    {activeResultTab === 'plan' && (
                      <PlanView perFileSummary={analysisResult.per_file_summary} />
                    )}

                    {/* 3D View tab */}
                    {activeResultTab === '3d' && (
                      <ThreeDView perFileSummary={analysisResult.per_file_summary} />
                    )}
                  </div>

                  {/* Per-File Breakdown Table — always visible */}
                  <div className="border-2" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>
                    <div className="px-6 py-4 border-b-2" style={{ borderColor: '#E2DED9', background: '#F5F3EF' }}>
                      <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#0A0A0A' }}>
                        Per-File Breakdown
                      </h3>
                    </div>
                    <table className="w-full text-sm">
                      <thead>
                        <tr style={{ background: '#F5F3EF', borderBottom: '2px solid #E2DED9' }}>
                          <th className="text-left px-6 py-3 font-bold text-xs uppercase" style={{ color: '#7A7470' }}>File</th>
                          <th className="text-right px-6 py-3 font-bold text-xs uppercase" style={{ color: '#7A7470' }}>Signals</th>
                          <th className="text-right px-6 py-3 font-bold text-xs uppercase" style={{ color: '#7A7470' }}>Delamination %</th>
                          <th className="text-right px-6 py-3 font-bold text-xs uppercase" style={{ color: '#7A7470' }}>Sound %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisResult.per_file_summary.map((file, idx) => (
                          <tr key={idx} style={{ borderTop: '1px solid #E2DED9' }}>
                            <td className="px-6 py-3 font-mono text-xs" style={{ color: '#0A0A0A' }}>{file.filename}</td>
                            <td className="text-right px-6 py-3" style={{ color: '#7A7470' }}>{file.signals.toLocaleString()}</td>
                            <td className="text-right px-6 py-3">
                              <span className="font-bold" style={{ color: file.delam_pct > 10 ? '#E74C3C' : '#0A0A0A' }}>
                                {file.delam_pct.toFixed(1)}%
                              </span>
                            </td>
                            <td className="text-right px-6 py-3">
                              <span className="font-bold" style={{ color: '#0A0A0A' }}>
                                {(100 - file.delam_pct).toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                </div>
              )}
            </div>
          ) : (
            /* In Development View */
            <div className="flex items-center justify-center min-h-[calc(100vh-120px)]">
              <div className="max-w-2xl w-full border-2 p-12" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>
                <h2 className="text-3xl font-bold mb-4" style={{ color: '#0A0A0A' }}>
                  {activeModule.fullName}
                </h2>

                <p className="text-base mb-6" style={{ color: '#7A7470' }}>
                  {activeModule.detectionDescription}
                </p>

                {/* Specs Row */}
                <div className="flex gap-3 mb-8">
                  <div className="px-3 py-2" style={{ background: '#F3F4F6' }}>
                    <p className="text-[11px] font-bold uppercase tracking-wide mb-1" style={{ color: '#9CA3AF' }}>
                      Input
                    </p>
                    <p className="text-[11px] font-medium" style={{ color: '#374151' }}>
                      {activeModule.input}
                    </p>
                  </div>
                  <div className="px-3 py-2" style={{ background: '#F3F4F6' }}>
                    <p className="text-[11px] font-bold uppercase tracking-wide mb-1" style={{ color: '#9CA3AF' }}>
                      Output
                    </p>
                    <p className="text-[11px] font-medium" style={{ color: '#374151' }}>
                      {activeModule.output}
                    </p>
                  </div>
                  <div className="px-3 py-2" style={{ background: '#F3F4F6' }}>
                    <p className="text-[11px] font-bold uppercase tracking-wide mb-1" style={{ color: '#9CA3AF' }}>
                      Standard
                    </p>
                    <p className="text-[11px] font-medium" style={{ color: '#374151' }}>
                      {activeModule.standard}
                    </p>
                  </div>
                </div>

                {/* Body Text */}
                <p className="text-sm mb-8" style={{ color: '#7A7470', lineHeight: '1.6' }}>
                  Verus is training a dedicated AI model for {activeModule.fullName} analysis. Each inspection method gets its own purpose-built model — same output format, same one-day turnaround.
                </p>

                {/* Email Notification */}
                <div>
                  <label className="block text-xs font-bold uppercase tracking-wide mb-2" style={{ color: '#7A7470' }}>
                    Get notified when this launches
                  </label>
                  <div className="flex gap-3">
                    <input
                      type="email"
                      value={notifyEmail}
                      onChange={(e) => setNotifyEmail(e.target.value)}
                      placeholder="your@email.com"
                      className="flex-1 px-4 py-3 border-2 text-sm"
                      style={{
                        borderColor: '#E2DED9',
                        background: '#FFFFFF',
                        color: '#0A0A0A',
                      }}
                    />
                    <button
                      onClick={() => {
                        if (notifyEmail) {
                          alert(`We'll notify you at ${notifyEmail} when ${activeModule.name} launches!`);
                          setNotifyEmail('');
                        }
                      }}
                      className="px-6 py-3 text-sm font-bold uppercase tracking-wide"
                      style={{ background: '#E8601C', color: '#FFFFFF' }}
                    >
                      Notify Me
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, alert = false }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="border-2 p-4" style={{ borderColor: '#E2DED9', background: '#FFFFFF' }}>
      <p className="text-xs font-bold uppercase tracking-wide mb-2" style={{ color: '#7A7470' }}>
        {label}
      </p>
      <p className="text-2xl font-bold" style={{ color: alert ? '#E74C3C' : '#0A0A0A' }}>
        {value}
      </p>
    </div>
  );
}
