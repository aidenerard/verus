import { useState, useRef, useEffect } from 'react';
import { Upload, Check, X, ChevronDown, Loader2, Download, FileText, Circle, Radio, Waves, Thermometer, Speaker } from 'lucide-react';
import { Button } from './components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './components/ui/collapsible';
import { Progress } from './components/ui/progress';

const PYTHON_SERVER_URL = 'https://verus-0j5j.onrender.com';

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

interface FileResult {
  filename: string;
  signals: number;
  delam_pct: number;
}

export default function App() {
  const [activeMethod, setActiveMethod] = useState<InspectionMethod>('gpr');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'warming' | 'offline'>('offline');
  const [slowRequest, setSlowRequest] = useState(false);
  const [notifyEmail, setNotifyEmail] = useState('');
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

  const addFiles = (newFiles: File[]) => {
    const csvFiles = newFiles.filter(file => file.name.toLowerCase().endsWith('.csv'));

    if (csvFiles.length === 0 && newFiles.length > 0) {
      alert('Please select CSV files. Open your folder and select the CSV files inside, not the folder itself.');
      return;
    }

    if (csvFiles.length !== newFiles.length) {
      alert(`Found ${csvFiles.length} CSV file(s). ${newFiles.length - csvFiles.length} non-CSV file(s) were skipped.`);
    }

    const uploadedFiles: UploadedFile[] = csvFiles.map(file => ({
      file,
      name: file.name,
    }));

    setFiles(prev => [...prev, ...uploadedFiles]);
  };

  const removeFile = (fileName: string) => {
    setFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setSlowRequest(false);

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
          alert('Server is waking up from sleep (Render free tier). This can take up to 2 minutes.\n\nPlease:\n1. Wait 2 minutes\n2. Try uploading again\n\nOR go to Render dashboard and click "Restart Service" to wake it up immediately.');
          setIsAnalyzing(false);
          setSlowRequest(false);
          setServerStatus('warming');
          return;
        }

        let errorMsg = 'Unknown error';
        try {
          const error = await response.json();
          console.error('[ANALYZE] Error response:', error);
          errorMsg = error.detail || error.error || error.stderr || JSON.stringify(error);
        } catch {
          const errorText = await response.text();
          console.error('[ANALYZE] Error text:', errorText);
          errorMsg = errorText || `HTTP ${response.status} error`;
        }
        console.error('Analysis failed:', errorMsg);
        alert(`Analysis failed:\n\n${errorMsg}\n\nThis might mean your CSV files aren't in the expected GPR format. Check browser console for details.`);
        setIsAnalyzing(false);
        setSlowRequest(false);
        return;
      }

      const result: AnalysisResult = await response.json();
      console.log('Analysis result:', result);

      setAnalysisResult(result);
      setReportOpen(true);
      setIsAnalyzing(false);
      setSlowRequest(false);
      setServerStatus('online'); // Server is clearly online after successful response

    } catch (error) {
      clearTimeout(slowTimeout);
      console.error('Error during analysis:', error);

      if (error instanceof Error && error.name === 'TimeoutError') {
        alert('Analysis timed out after 5 minutes. This may happen if:\n\n1. Server is cold-starting (wait 90 seconds and try again)\n2. Too many files uploaded (try with fewer files)\n3. Server needs more resources\n\nCheck the Render dashboard logs for details.');
      } else if (error instanceof TypeError && error.message.includes('fetch')) {
        // Network error or CORS issue
        alert('Cannot connect to server. This usually means:\n\n1. Server is asleep (Render free tier) - wait 2 minutes and retry\n2. Network connection issue\n3. CORS error\n\nTry:\n• Wait 2 minutes, then upload again\n• Go to Render dashboard → Click "Restart Service"\n• Check browser console for CORS errors');
        setServerStatus('warming');
      } else if (error instanceof Error) {
        alert(`Analysis failed: ${error.message}\n\nCheck browser console for details.`);
      } else {
        alert('Analysis failed. Please check your connection and try again.');
      }

      setIsAnalyzing(false);
      setSlowRequest(false);
    }
  };

  const activeModule = INSPECTION_MODULES.find(m => m.id === activeMethod)!;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#FFFFFF' }}>
      {/* Professional Header */}
      <header className="border-b-2" style={{ borderColor: '#0D47A1', background: '#FFFFFF' }}>
        <div className="px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 flex items-center justify-center" style={{ background: '#0D47A1' }}>
              <span className="text-white font-bold text-lg">V</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold" style={{ color: '#2C3E50', letterSpacing: '-0.01em' }}>
                VERUS
              </h1>
              <p className="text-xs" style={{ color: '#6C757D' }}>
                Bridge Deck Inspection Intelligence
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs font-semibold" style={{ color: '#6C757D' }}>Multi-Standard Compliant</p>
          </div>
        </div>
      </header>

      {/* Main Content with Sidebar */}
      <div className="flex flex-1">
        {/* Left Sidebar - Inspection Method Selector */}
        <aside className="w-80 border-r-2 p-6" style={{ background: '#F4F5F7', borderColor: '#DEE2E6' }}>
          <h2 className="text-xs font-bold uppercase tracking-wide mb-4" style={{ color: '#6C757D' }}>
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
                    borderColor: isActive ? '#1a2f5a' : '#DEE2E6',
                    borderLeftWidth: isActive ? '3px' : '2px',
                  }}
                >
                  <div className="flex items-start gap-3">
                    <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: '#1a2f5a' }} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <h3 className="text-sm font-bold" style={{ color: '#2C3E50' }}>
                          {module.name}
                        </h3>
                        <span
                          className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                          style={{
                            background: module.status === 'available' ? '#2E7D32' : '#E8EDF5',
                            color: module.status === 'available' ? '#FFFFFF' : '#1a2f5a',
                          }}
                        >
                          {module.status === 'available' ? 'Available' : 'In Development'}
                        </span>
                      </div>
                      <p className="text-[10px] mb-1" style={{ color: '#6C757D' }}>
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
            <div className="grid grid-cols-12 gap-8">
              {/* Left Panel - Upload & Settings */}
              <div className="col-span-4 space-y-6">
            {/* Upload Section */}
            <div className="border-2" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
              <div className="px-6 py-4 border-b-2" style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#2C3E50' }}>
                  Data Upload
                </h2>
              </div>
              <div className="p-6">
                <div
                  className={`border-2 border-dashed cursor-pointer transition-all ${
                    isDragging ? 'border-[#0D47A1] bg-[#E3F2FD]' : 'border-[#DEE2E6] bg-[#F8F9FA]'
                  }`}
                  style={{ padding: '32px' }}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="flex flex-col items-center gap-3">
                    <Upload className="w-10 h-10" style={{ color: '#6C757D' }} />
                    <div className="text-center">
                      <p className="text-sm font-semibold" style={{ color: '#2C3E50' }}>
                        Upload GPR Signal Files
                      </p>
                      <p className="text-xs mt-1" style={{ color: '#6C757D' }}>
                        Select multiple CSV files • Drag & drop supported
                      </p>
                      <p className="text-xs mt-1" style={{ color: '#9CA3AF', fontStyle: 'italic' }}>
                        (Open folder and select all CSV files, not the folder itself)
                      </p>
                    </div>
                  </div>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".csv"
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
                        style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}
                      >
                        <div className="flex items-center gap-3 flex-1 min-w-0">
                          <Check className="w-4 h-4 flex-shrink-0" style={{ color: '#2ECC71' }} />
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-medium truncate" style={{ color: '#2C3E50' }}>
                              {file.name}
                            </p>
                            <p className="text-xs" style={{ color: '#6C757D' }}>
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
                          <X className="w-4 h-4" style={{ color: '#6C757D' }} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Analysis Configuration */}
            <div className="border-2" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
              <div className="px-6 py-4 border-b-2" style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#2C3E50' }}>
                  Analysis Configuration
                </h2>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-bold uppercase tracking-wide" style={{ color: '#6C757D' }}>
                      AI Model Version
                    </label>
                    <div className="flex items-center gap-2">
                      <Circle
                        className="w-2 h-2"
                        fill={serverStatus === 'online' ? '#2ECC71' : serverStatus === 'warming' ? '#F39C12' : '#E74C3C'}
                        stroke="none"
                      />
                      <span className="text-xs" style={{ color: '#6C757D' }}>
                        {serverStatus === 'online' ? 'Online' : serverStatus === 'warming' ? 'Server warming up' : 'Offline'}
                      </span>
                    </div>
                  </div>
                  <div className="px-4 py-3 border-2" style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                    <p className="text-sm font-medium" style={{ color: '#2C3E50' }}>
                      model_v13.pth
                    </p>
                  </div>
                  <p className="text-xs mt-2" style={{ color: '#6C757D' }}>
                    Latest model automatically selected
                  </p>
                </div>

                <div className="pt-4 border-t" style={{ borderColor: '#DEE2E6' }}>
                  <div className="flex justify-between text-xs mb-1">
                    <span style={{ color: '#6C757D' }}>Detection Threshold:</span>
                    <span className="font-bold" style={{ color: '#2C3E50' }}>0.65 (Optimal)</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span style={{ color: '#6C757D' }}>Analysis Standard:</span>
                    <span className="font-bold" style={{ color: '#2C3E50' }}>ASTM D6087</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Run Analysis Button */}
            <button
              onClick={handleAnalyze}
              disabled={files.length === 0 || isAnalyzing}
              className="w-full py-4 text-sm font-bold uppercase tracking-wide transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: files.length === 0 || isAnalyzing ? '#DEE2E6' : '#0D47A1',
                color: files.length === 0 || isAnalyzing ? '#6C757D' : '#FFFFFF',
                border: files.length === 0 || isAnalyzing ? '2px solid #DEE2E6' : '2px solid #0D47A1',
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

            {/* Loading state with slow request warning */}
            {isAnalyzing && (
              <div className="space-y-2">
                <p className="text-xs text-center font-medium" style={{ color: '#6C757D' }}>
                  Processing GPR signals...
                </p>
                {slowRequest && (
                  <div className="p-3 border-2" style={{ borderColor: '#F39C12', background: '#FFF3CD' }}>
                    <p className="text-xs text-center font-medium" style={{ color: '#856404' }}>
                      Waking up server, this may take up to 90 seconds on first request...
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Panel - Results */}
          <div className="col-span-8">
            {!analysisResult ? (
              <div className="h-full min-h-[600px] flex items-center justify-center border-2"
                   style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                <div className="text-center p-12">
                  <FileText className="w-16 h-16 mx-auto mb-4" style={{ color: '#ADB5BD' }} />
                  <h3 className="text-lg font-bold mb-2" style={{ color: '#2C3E50' }}>
                    Ready for Analysis
                  </h3>
                  <p className="text-sm" style={{ color: '#6C757D' }}>
                    Upload GPR signal files and run analysis to generate C-scan report
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* C-Scan Map Display */}
                <div className="border-2" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
                  <div className="px-6 py-4 border-b-2 flex items-center justify-between"
                       style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                    <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#2C3E50' }}>
                      C-Scan Analysis Map
                    </h2>
                    <button
                      className="px-4 py-2 text-xs font-bold uppercase tracking-wide flex items-center gap-2"
                      style={{ background: '#0D47A1', color: '#FFFFFF' }}
                    >
                      <Download className="w-4 h-4" />
                      Export PNG
                    </button>
                  </div>
                  <div className="p-6">
                    {analysisResult.cscan_image ? (
                      <img
                        src={`data:image/png;base64,${analysisResult.cscan_image}`}
                        alt="C-Scan Analysis Map"
                        className="w-full border"
                        style={{ borderColor: '#DEE2E6' }}
                      />
                    ) : (
                      <div className="w-full h-96 flex items-center justify-center border-2 border-dashed"
                           style={{ borderColor: '#DEE2E6', background: '#F8F9FA' }}>
                        <p className="text-sm" style={{ color: '#6C757D' }}>
                          C-scan visualization not available
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Summary Statistics */}
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

                {/* ASTM Report */}
                <Collapsible open={reportOpen} onOpenChange={setReportOpen}>
                  <div className="border-2" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
                    <CollapsibleTrigger className="w-full">
                      <div className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors">
                        <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#2C3E50' }}>
                          ASTM D6087 Condition Assessment Report
                        </h3>
                        <ChevronDown
                          className={`w-5 h-5 transition-transform ${reportOpen ? 'rotate-180' : ''}`}
                          style={{ color: '#6C757D' }}
                        />
                      </div>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="px-6 py-4 border-t-2 space-y-6" style={{ borderColor: '#DEE2E6' }}>
                        {/* Report Header */}
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="font-bold" style={{ color: '#6C757D' }}>Bridge ID:</span>
                            <span style={{ color: '#2C3E50' }}>BR-2024-001</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="font-bold" style={{ color: '#6C757D' }}>Analysis Date:</span>
                            <span style={{ color: '#2C3E50' }}>
                              {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="font-bold" style={{ color: '#6C757D' }}>Model Version:</span>
                            <span style={{ color: '#2C3E50' }}>model_v13.pth</span>
                          </div>
                        </div>

                        {/* Per-File Summary Table */}
                        <div>
                          <h4 className="text-xs font-bold uppercase tracking-wide mb-3" style={{ color: '#2C3E50' }}>
                            File Analysis Summary
                          </h4>
                          <div className="border-2" style={{ borderColor: '#DEE2E6' }}>
                            <table className="w-full text-sm">
                              <thead>
                                <tr style={{ background: '#F8F9FA' }}>
                                  <th className="text-left p-3 font-bold text-xs uppercase" style={{ color: '#6C757D' }}>File</th>
                                  <th className="text-right p-3 font-bold text-xs uppercase" style={{ color: '#6C757D' }}>Signals</th>
                                  <th className="text-right p-3 font-bold text-xs uppercase" style={{ color: '#6C757D' }}>Delamination %</th>
                                </tr>
                              </thead>
                              <tbody>
                                {analysisResult.per_file_summary.map((file, idx) => (
                                  <tr key={idx} style={{ borderTop: '1px solid #DEE2E6' }}>
                                    <td className="p-3" style={{ color: '#2C3E50' }}>{file.filename}</td>
                                    <td className="text-right p-3" style={{ color: '#6C757D' }}>{file.signals.toLocaleString()}</td>
                                    <td className="text-right p-3">
                                      <span className="font-bold" style={{ color: file.delam_pct > 10 ? '#E74C3C' : '#2C3E50' }}>
                                        {file.delam_pct.toFixed(1)}%
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Export Button */}
                        <button
                          className="w-full py-3 text-sm font-bold uppercase tracking-wide flex items-center justify-center gap-2"
                          style={{ background: '#0D47A1', color: '#FFFFFF' }}
                        >
                          <Download className="w-5 h-5" />
                          Export PDF Report
                        </button>
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
                </div>
              )}
            </div>
          </div>
          ) : (
            /* In Development View */
            <div className="flex items-center justify-center min-h-[calc(100vh-120px)]">
              <div className="max-w-2xl w-full border-2 p-12" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
                <h2 className="text-3xl font-bold mb-4" style={{ color: '#2C3E50' }}>
                  {activeModule.fullName}
                </h2>

                <p className="text-base mb-6" style={{ color: '#6C757D' }}>
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
                <p className="text-sm mb-8" style={{ color: '#6C757D', lineHeight: '1.6' }}>
                  Verus is training a dedicated AI model for {activeModule.fullName} analysis. Each inspection method gets its own purpose-built model — same output format, same one-day turnaround.
                </p>

                {/* Email Notification */}
                <div>
                  <label className="block text-xs font-bold uppercase tracking-wide mb-2" style={{ color: '#6C757D' }}>
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
                        borderColor: '#DEE2E6',
                        background: '#FFFFFF',
                        color: '#2C3E50',
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
                      style={{ background: '#0D47A1', color: '#FFFFFF' }}
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
    <div className="border-2 p-4" style={{ borderColor: '#DEE2E6', background: '#FFFFFF' }}>
      <p className="text-xs font-bold uppercase tracking-wide mb-2" style={{ color: '#6C757D' }}>
        {label}
      </p>
      <p className="text-2xl font-bold" style={{ color: alert ? '#E74C3C' : '#2C3E50' }}>
        {value}
      </p>
    </div>
  );
}
