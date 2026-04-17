import React, { useState, useCallback, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  TextField,
  Button,
  Slider,
  CircularProgress,
  Card,
  CardMedia,
  CardContent,
  Chip,
  AppBar,
  Toolbar,
  Divider,
  Alert,
  ThemeProvider,
  createTheme,
  CssBaseline,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tabs,
  Tab,
  LinearProgress,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Settings as SettingsIcon,
  Image as ImageIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Movie as MovieIcon,
  Folder as FolderIcon,
  CreateNewFolder as CreateNewFolderIcon,
  Refresh as RefreshIcon,
  BarChart as BarChartIcon,
} from '@mui/icons-material';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, Cell } from 'recharts';

const API_BASE_URL = 'http://localhost:8080';
const WS_BASE_URL = 'ws://localhost:8080';

const sandTheme = createTheme({
  palette: {
    mode: 'light',
    background: { default: '#F7F5F0', paper: '#FFFFFF' },
    primary: { main: '#4A4238' },
    secondary: { main: '#D4CBB3' },
    success: { main: '#6B7A5F' },
    error: { main: '#A65D57' },
    text: { primary: '#2C2723', secondary: '#665E57' },
  },
  typography: {
    fontFamily: '"JetBrains Mono", "Roboto Mono", monospace',
    h6: { fontWeight: 700, letterSpacing: '-0.5px' },
  },
  shape: { borderRadius: 4 },
});

interface VideoMetadata {
  duration: number;
  width: number;
  height: number;
  avg_frame_rate: string;
}

interface ProcessResult {
  filename: string;
  url: string;
  score: number;
  is_blurry: boolean;
}

interface DirectoryNode {
  name: string;
  path: string;
  children: DirectoryNode[];
}

const PRESETS = {
  '3DGS': { fps: 5, qscale: 2, threshold: 100, label: '3D Gaussian Splatting (High Density)' },
  '2DGS': { fps: 2, qscale: 3, threshold: 80, label: '2D Gaussian Splatting (Standard)' },
  'COLMAP': { fps: 3, qscale: 2, threshold: 120, label: 'COLMAP/SfM (Sharp Only)' },
  'FAST': { fps: 1, qscale: 5, threshold: 50, label: 'Fast Preview (Low Quality)' },
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<ProcessResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [preset, setPreset] = useState('');
  const [fps, setFps] = useState(1);
  const [scale, setScale] = useState('-1:-1');
  const [scaleOption, setScaleOption] = useState('original');
  const [qscale, setQscale] = useState(2);
  const [namingRule, setNamingRule] = useState('frame_%04d.jpg');
  const [threshold, setThreshold] = useState(100);
  const [outputPath, setOutputPath] = useState('');

  const [progressMsg, setProgressMsg] = useState('');
  const [currentProgress, setCurrentProgress] = useState(0);
  const [totalEstimated, setTotalEstimated] = useState(0);

  const [directories, setDirectories] = useState<DirectoryNode | null>(null);
  const [newFolderName, setNewFolderName] = useState('');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [targetParentPath, setTargetParentPath] = useState('');
  const [tabValue, setTabValue] = useState(0);

  const fetchDirectories = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/directories`);
      setDirectories(res.data);
      if (!outputPath) setOutputPath(res.data.path);
    } catch (err) { console.error(err); }
  };

  useEffect(() => { fetchDirectories(); }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
      uploadVideo(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'video/*': [] }, multiple: false,
  });

  const uploadVideo = async (selectedFile: File) => {
    setLoading(true);
    setMetadata(null);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const uploadRes = await axios.post(`${API_BASE_URL}/upload`, formData);
      const id = uploadRes.data.file_id;
      setFileId(id);
      const metaRes = await axios.get(`${API_BASE_URL}/metadata/${id}`);
      setMetadata(metaRes.data);
      setScale('-1:-1');
      if (metaRes.data) setTotalEstimated(Math.ceil(metaRes.data.duration * fps));
    } catch (err: any) { setError(`CONNECTION_ERROR: ${err.message}`); } finally { setLoading(false); }
  };

  useEffect(() => { if (metadata) setTotalEstimated(Math.ceil(metadata.duration * fps)); }, [fps, metadata]);

  const applyPreset = (key: keyof typeof PRESETS) => {
    const p = PRESETS[key];
    setPreset(key); setFps(p.fps); setQscale(p.qscale); setThreshold(p.threshold);
  };

  const handleScaleChange = (option: string) => {
    setScaleOption(option);
    if (!metadata) return;
    if (option === 'original') setScale('-1:-1');
    else if (option === '1080p') setScale('1920:1080');
    else if (option === '720p') setScale('1280:720');
    else if (option === 'half') setScale(`${Math.round(metadata.width/2)}:${Math.round(metadata.height/2)}`);
  };

  const handleCreateFolder = async () => {
    if (!newFolderName) return;
    try {
      await axios.post(`${API_BASE_URL}/create-directory`, { parent_path: targetParentPath, new_name: newFolderName });
      setNewFolderName(''); setIsDialogOpen(false); fetchDirectories();
    } catch (err: any) { alert(`Folder creation failed: ${err.message}`); }
  };

  const handleProcess = () => {
    if (!fileId) return;
    setProcessing(true); setResults([]); setError(null); setCurrentProgress(0); setProgressMsg('CONNECTING...');
    const ws = new WebSocket(`${WS_BASE_URL}/ws/process/${fileId}`);
    ws.onopen = () => ws.send(JSON.stringify({ fps, scale, qscale, naming_rule: namingRule, threshold, output_path: outputPath }));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status' || data.type === 'progress') {
        setProgressMsg(data.message);
        if (data.current) setCurrentProgress(data.current);
      } else if (data.type === 'complete') {
        setResults(data.results); setProcessing(false); fetchDirectories(); ws.close();
      } else if (data.type === 'error') {
        setError(`PROCESS_ERROR: ${data.message}`); setProcessing(false); ws.close();
      }
    };
    ws.onerror = () => { setError('WebSocket Connection Failed.'); setProcessing(false); };
  };

  const filteredResults = tabValue === 0 ? results : results.filter(r => !r.is_blurry);
  const sharpCount = results.filter(r => !r.is_blurry).length;
  const blurCount = results.length - sharpCount;

  const chartData = [
    { name: 'SHARP', count: sharpCount, color: '#6B7A5F' },
    { name: 'BLUR', count: blurCount, color: '#A65D57' }
  ];

  const renderTree = (node: DirectoryNode) => (
    <TreeItem key={node.path} itemId={node.path} label={
      <Box sx={{ display: 'flex', alignItems: 'center', py: 0.5 }}>
        <FolderIcon sx={{ mr: 1, fontSize: 18, color: '#D4CBB3' }} />
        <Typography variant="caption" sx={{ flexGrow: 1 }}>{node.name}</Typography>
        <IconButton size="small" onClick={(e) => { e.stopPropagation(); setTargetParentPath(node.path); setIsDialogOpen(true); }}><CreateNewFolderIcon sx={{ fontSize: 16 }} /></IconButton>
      </Box>
    }>{Array.isArray(node.children) ? node.children.map((child) => renderTree(child)) : null}</TreeItem>
  );

  return (
    <ThemeProvider theme={sandTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', pb: 8 }}>
        <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid #E6E1D6', mb: 4, bgcolor: '#FFFFFF' }}>
          <Toolbar>
            <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              VIDEO_TO_IMAGE_PIPELINE <Chip label="V1.6.0" size="small" sx={{ ml: 1, height: 20, fontSize: '10px' }} />
            </Typography>
            {metadata && <Typography variant="caption" color="success.main">● BACKEND_CONNECTED</Typography>}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl">
          <Grid container spacing={3}>
            {/* Sidebar */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, mb: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[01] INPUT_SOURCE</Typography>
                <Box {...getRootProps()} sx={{ border: '2px dashed #D4CBB3', borderRadius: 1, p: 4, textAlign: 'center', cursor: 'pointer', bgcolor: isDragActive ? '#F4F1EA' : 'transparent', '&:hover': { borderColor: '#4A4238', bgcolor: '#F4F1EA' } }}>
                  <input {...getInputProps()} /><CloudUploadIcon sx={{ fontSize: 40, color: '#D4CBB3', mb: 1 }} />
                  <Typography variant="body2">{isDragActive ? "Drop video here..." : "Drag & drop video, or click to select"}</Typography>
                </Box>
                {file && (
                  <Box sx={{ mt: 3, p: 2, bgcolor: '#FFFFFF', border: '1px solid #E6E1D6' }}>
                    <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1.5, fontWeight: 'bold' }}><MovieIcon sx={{ mr: 1, fontSize: 18 }} /> [DATA_SUMMARY]</Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {[{ label: 'FILE_NAME', value: file.name }, { label: 'FILE_SIZE', value: `${(file.size / (1024 * 1024)).toFixed(2)} MB` }, ...(metadata ? [{ label: 'RESOLUTION', value: `${metadata.width} x ${metadata.height}` }, { label: 'VIDEO_FPS', value: metadata.avg_frame_rate }, { label: 'LENGTH', value: `${metadata.duration.toFixed(2)}s` }] : [])].map((item, idx) => (
                        <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #F0EDE5', pb: 0.5 }}>
                          <Typography variant="caption" color="textSecondary" sx={{ fontWeight: 'bold' }}>{item.label}:</Typography>
                          <Typography variant="caption" sx={{ color: 'primary.main' }}>{item.value}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>

              <Paper sx={{ p: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[02] CONFIGURATION</Typography>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1, color: '#8C8273' }}>--- OUTPUT_BROWSER --- <IconButton size="small" onClick={fetchDirectories}><RefreshIcon sx={{ fontSize: 14 }} /></IconButton></Typography>
                  <Box sx={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #E6E1D6', p: 1, mb: 1, bgcolor: '#FFFFFF' }}>
                    {directories ? <SimpleTreeView onSelectedItemsChange={(_, id) => setOutputPath(id as string)} selectedItems={outputPath}>{renderTree(directories)}</SimpleTreeView> : <CircularProgress size={20} />}
                  </Box>
                  <TextField fullWidth label="Selected Path" value={outputPath} size="small" variant="filled" slotProps={{ input: { readOnly: true } }} />
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1, color: '#8C8273' }}>--- EXTRACTION & BLUR ---</Typography>
                  <Grid container spacing={1}>
                    <Grid item xs={4}><TextField fullWidth label="FPS" type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} size="small" /></Grid>
                    <Grid item xs={4}><TextField fullWidth label="Quality" type="number" value={qscale} onChange={(e) => setQscale(Number(e.target.value))} size="small" /></Grid>
                    <Grid item xs={4}><TextField fullWidth label="Thres" type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} size="small" /></Grid>
                  </Grid>
                  <Slider value={threshold} onChange={(_, v) => setThreshold(v as number)} min={0} max={500} step={5} sx={{ mt: 1 }} />
                </Box>
                <Button fullWidth variant="contained" size="large" sx={{ py: 1.5, bgcolor: '#4A4238' }} onClick={handleProcess} disabled={!fileId || processing}>
                  {processing ? <CircularProgress size={24} color="inherit" /> : 'EXEC_PROCESSING'}
                </Button>
              </Paper>
            </Grid>

            {/* Main Area */}
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, minHeight: '85vh', borderLeft: '4px solid #4A4238' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="subtitle2" color="textSecondary">[03] OUTPUT_BUFFER & VISUAL_ANALYTICS</Typography>
                  {results.length > 0 && (
                    <Box sx={{ width: 250, height: 100, border: '1px solid #E6E1D6', p: 1, bgcolor: '#FDFCFB' }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>CLASSIFICATION_RATIO</Typography>
                      <ResponsiveContainer width="100%" height="80%">
                        <BarChart data={chartData} layout="vertical" margin={{ left: -30 }}>
                          <XAxis type="number" hide />
                          <YAxis type="category" dataKey="name" hide />
                          <ChartTooltip />
                          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                            {chartData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  )}
                </Box>

                {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
                {processing && (
                  <Box sx={{ mb: 3, p: 2, bgcolor: '#FDFCFB', border: '1px solid #E6E1D6' }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>{progressMsg}</Typography>
                    <LinearProgress variant="determinate" value={Math.min((currentProgress / (totalEstimated || 1)) * 100, 100)} sx={{ height: 10, borderRadius: 1 }} />
                  </Box>
                )}

                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                  <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} textColor="primary" indicatorColor="primary">
                    <Tab label={`ALL (${results.length})`} sx={{ fontSize: '11px' }} />
                    <Tab label={`SHARP (${sharpCount})`} sx={{ fontSize: '11px' }} />
                    <Tab label={`BLUR (${blurCount})`} sx={{ fontSize: '11px' }} />
                  </Tabs>
                </Box>

                {results.length > 0 ? (
                  <Grid container spacing={1.5}>
                    {filteredResults.map((res, index) => (
                      <Grid item xs={6} sm={4} md={3} lg={2.4} key={index}>
                        <Card variant="outlined" sx={{ borderRadius: 0, opacity: res.is_blurry ? 0.6 : 1 }}>
                          <CardMedia component="img" height="100" image={`${API_BASE_URL}${res.url}`} sx={{ filter: res.is_blurry ? 'grayscale(80%)' : 'none' }} />
                          <CardContent sx={{ p: 0.8 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="caption" sx={{ fontSize: '9px', fontWeight: 'bold' }}>S:{res.score}</Typography>
                              <Chip label={res.is_blurry ? 'B' : 'S'} size="small" color={res.is_blurry ? 'error' : 'success'} sx={{ height: 12, fontSize: '7px', minWidth: 16 }} />
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                ) : !processing && <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', opacity: 0.2 }}><ImageIcon sx={{ fontSize: 64 }} /></Box>}
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <DialogTitle sx={{ fontSize: '14px', fontWeight: 'bold' }}>CREATE_NEW_FOLDER</DialogTitle>
        <DialogContent><TextField autoFocus margin="dense" label="Folder Name" fullWidth size="small" value={newFolderName} onChange={(e) => setNewFolderName(e.target.value)} /></DialogContent>
        <DialogActions><Button onClick={() => setIsDialogOpen(false)}>CANCEL</Button><Button onClick={handleCreateFolder} variant="contained">CREATE</Button></DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}

export default App;
