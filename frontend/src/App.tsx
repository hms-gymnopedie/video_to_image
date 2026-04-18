import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
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
  Sync as SyncIcon,
  AutoFixHigh as MagicIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';

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

interface ProcessResult {
  filename: string;
  url: string | null;
  score: number;
  is_blurry: boolean;
  full_path?: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [results, setResults] = useState<ProcessResult[]>([]);
  const [serverOutputDir, setServerOutputDir] = useState('');
  const [error, setError] = useState<string | null>(null);

  const [preset, setPreset] = useState('');
  const [fps, setFps] = useState(1);
  const [scale, setScale] = useState('-1:-1');
  const [scaleOption, setScaleOption] = useState('original');
  const [qscale, setQscale] = useState(2);
  const [threshold, setThreshold] = useState(100);
  const [outputPath, setOutputPath] = useState('');

  const [progressMsg, setProgressMsg] = useState('');
  const [currentProgress, setCurrentProgress] = useState(0);
  const [totalEstimated, setTotalEstimated] = useState(0);

  const [directories, setDirectories] = useState<any>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [targetParentPath, setTargetParentPath] = useState('');
  const [tabValue, setTabValue] = useState(0);

  // Masking Editor State
  const [isMaskEditorOpen, setIsMaskEditorOpen] = useState(false);
  const [editingImage, setEditingImage] = useState<ProcessResult | null>(null);
  const [points, setPoints] = useState<number[][]>([]);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const fetchDirectories = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/directories`);
      setDirectories(res.data);
      if (!outputPath && res.data.path) setOutputPath(res.data.path);
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
    setLoading(true); setMetadata(null); setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const uploadRes = await axios.post(`${API_BASE_URL}/upload`, formData);
      const id = uploadRes.data.file_id;
      setFileId(id);
      const metaRes = await axios.get(`${API_BASE_URL}/metadata/${id}`);
      setMetadata(metaRes.data);
      if (metaRes.data) setTotalEstimated(Math.ceil(metaRes.data.duration * fps));
    } catch (err: any) { setError(`CONNECTION_ERROR: ${err.message}`); } finally { setLoading(false); }
  };

  const handleProcess = () => {
    if (!fileId) return;
    setProcessing(true); setResults([]); setError(null); setCurrentProgress(0); setProgressMsg('CONNECTING...');
    const ws = new WebSocket(`${WS_BASE_URL}/ws/process/${fileId}`);
    ws.onopen = () => ws.send(JSON.stringify({ fps, scale, qscale, naming_rule: 'frame_%04d.jpg', threshold, output_path: outputPath }));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status' || data.type === 'progress') {
        setProgressMsg(data.message);
        if (data.current) setCurrentProgress(data.current);
      } else if (data.type === 'complete') {
        setResults(data.results);
        setServerOutputDir(data.output_dir);
        setProcessing(false);
        fetchDirectories();
        ws.close();
      } else if (data.type === 'error') {
        setError(`PROCESS_ERROR: ${data.message}`); setProcessing(false); ws.close();
      }
    };
  };

  const handleImageClick = (res: ProcessResult) => {
    setEditingImage(res);
    setPoints([]);
    setMaskUrl(null);
    setIsMaskEditorOpen(true);
  };

  const handleCanvasClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current || !editingImage) return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * (metadata?.width || 1920));
    const y = Math.round(((e.clientY - rect.top) / rect.height) * (metadata?.height || 1080));
    
    const newPoints = [...points, [x, y]];
    setPoints(newPoints);
    
    setIsSegmenting(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/segment`, {
        image_path: editingImage.full_path,
        points: newPoints,
        labels: new Array(newPoints.length).fill(1)
      });
      setMaskUrl(`${API_BASE_URL}${res.data.mask_url}?t=${Date.now()}`);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSegmenting(false);
    }
  };

  const analyticsData = useMemo(() => {
    if (results.length === 0) return null;
    const sharp = results.filter(r => r.score >= threshold);
    const blur = results.filter(r => r.score < threshold);
    const avgAll = results.reduce((acc, curr) => acc + curr.score, 0) / results.length;
    const avgSharp = sharp.length > 0 ? sharp.reduce((acc, curr) => acc + curr.score, 0) / sharp.length : 0;
    return { sharpCount: sharp.length, blurCount: blur.length, avgAll: parseFloat(avgAll.toFixed(2)), avgSharp: parseFloat(avgSharp.toFixed(2)) };
  }, [results, threshold]);

  const renderTree = (node: any) => (
    <TreeItem key={node.path} itemId={node.path} label={
      <Box sx={{ display: 'flex', alignItems: 'center', py: 0.5 }}>
        <FolderIcon sx={{ mr: 1, fontSize: 18, color: '#D4CBB3' }} />
        <Typography variant="caption" sx={{ flexGrow: 1 }}>{node.name}</Typography>
        <IconButton size="small" onClick={(e) => { e.stopPropagation(); setTargetParentPath(node.path); setIsDialogOpen(true); }}><CreateNewFolderIcon sx={{ fontSize: 16 }} /></IconButton>
      </Box>
    }>{Array.isArray(node.children) ? node.children.map((child: any) => renderTree(child)) : null}</TreeItem>
  );

  return (
    <ThemeProvider theme={sandTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', pb: 8 }}>
        <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid #E6E1D6', mb: 4, bgcolor: '#FFFFFF' }}>
          <Toolbar>
            <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              VIDEO_TO_IMAGE_AI_PIPELINE <Chip label="V3.0.0" size="small" sx={{ ml: 1, height: 20, fontSize: '10px' }} />
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl">
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, mb: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[01] INPUT_SOURCE</Typography>
                <Box {...getRootProps()} sx={{ border: '2px dashed #D4CBB3', borderRadius: 1, p: 4, textAlign: 'center', cursor: 'pointer', bgcolor: isDragActive ? '#F4F1EA' : 'transparent', '&:hover': { borderColor: '#4A4238', bgcolor: '#F4F1EA' } }}>
                  <input {...getInputProps()} /><CloudUploadIcon sx={{ fontSize: 40, color: '#D4CBB3', mb: 1 }} />
                  <Typography variant="body2">{isDragActive ? "Drop video here..." : "Drag & drop video, or click to select"}</Typography>
                </Box>
                {file && (
                  <Box sx={{ mt: 3, p: 2, bgcolor: '#FFFFFF', border: '1px solid #E6E1D6' }}>
                    <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1.5, fontWeight: 'bold' }}><MovieIcon sx={{ mr: 1, fontSize: 18 }} /> SUMMARY</Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {[{ label: 'FILE', value: file.name }, { label: 'SIZE', value: `${(file.size / (1024 * 1024)).toFixed(2)} MB` }].map((item, idx) => (
                        <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #F0EDE5', pb: 0.5 }}>
                          <Typography variant="caption" color="textSecondary">{item.label}:</Typography>
                          <Typography variant="caption">{item.value}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>

              <Paper sx={{ p: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[02] AI_CONFIG</Typography>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>--- OUTPUT_DIR ---</Typography>
                  <Box sx={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #E6E1D6', p: 1, mb: 1, bgcolor: '#FFFFFF' }}>
                    {directories ? <SimpleTreeView onSelectedItemsChange={(_, id) => setOutputPath(id as string)} selectedItems={outputPath}>{renderTree(directories)}</SimpleTreeView> : <CircularProgress size={20} />}
                  </Box>
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>--- PARAMS ---</Typography>
                  <Grid container spacing={1}>
                    <Grid item xs={6}><TextField fullWidth label="FPS" type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} size="small" /></Grid>
                    <Grid item xs={6}><TextField fullWidth label="Thres" type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} size="small" /></Grid>
                  </Grid>
                  <Slider value={threshold} onChange={(_, v) => setThreshold(v as number)} min={0} max={500} step={5} sx={{ mt: 1 }} />
                </Box>
                <Button fullWidth variant="contained" size="large" sx={{ py: 1.5, bgcolor: '#4A4238' }} onClick={handleProcess} disabled={!fileId || processing}>
                  {processing ? <CircularProgress size={24} color="inherit" /> : 'START_PIPELINE'}
                </Button>
              </Paper>
            </Grid>

            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, minHeight: '85vh', borderLeft: '4px solid #4A4238' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="subtitle2" color="textSecondary">[03] AI_BUFFER</Typography>
                  {analyticsData && (
                    <Box sx={{ display: 'flex', gap: 2 }}>
                      <Box sx={{ width: 180, height: 100, border: '1px solid #E6E1D6', p: 1 }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', fontSize: '8px' }}>COUNT</Typography>
                        <ResponsiveContainer width="100%" height="80%">
                          <BarChart data={[{n:'S',v:analyticsData.sharpCount},{n:'B',v:analyticsData.blurCount}]} layout="vertical">
                            <YAxis dataKey="n" type="category" hide />
                            <Bar dataKey="v" radius={[0, 4, 4, 0]}>
                              <Cell fill="#6B7A5F"/><Cell fill="#A65D57"/>
                              <LabelList dataKey="v" position="right" fontSize={9} />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </Box>
                  )}
                </Box>

                <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} sx={{ mb: 3 }}>
                  <Tab label="ALL" />
                  <Tab label="SHARP" />
                  <Tab label="BLUR" />
                </Tabs>

                <Grid container spacing={1}>
                  {results.length > 0 ? (
                    filteredResults.map((res, index) => (
                      <Grid item xs={4} sm={3} md={2} key={index}>
                        <Card variant="outlined" onClick={() => handleImageClick(res)} sx={{ cursor: 'pointer', '&:hover': { borderColor: 'primary.main' } }}>
                          <CardMedia component="img" height="80" image={`${API_BASE_URL}${res.url}`} />
                          <Box sx={{ p: 0.5, textAlign: 'center' }}>
                            <Typography variant="caption" sx={{ fontSize: '8px' }}>{res.score}</Typography>
                          </Box>
                        </Card>
                      </Grid>
                    ))
                  ) : (
                    <Box sx={{ width: '100%', textAlign: 'center', py: 10, opacity: 0.2 }}>
                      <MagicIcon sx={{ fontSize: 64 }} />
                      <Typography>Waiting for results...</Typography>
                    </Box>
                  )}
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Masking Editor Dialog */}
      <Dialog open={isMaskEditorOpen} onClose={() => setIsMaskEditorOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          AI MASKING EDITOR (SAM)
          <Box>
            <Button size="small" startIcon={<ClearIcon />} onClick={() => setPoints([])}>Reset</Button>
            <IconButton onClick={() => setIsMaskEditorOpen(false)}><ClearIcon /></IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ position: 'relative', textAlign: 'center', bgcolor: '#000', borderRadius: 1, overflow: 'hidden' }}>
            {editingImage && (
              <>
                <img
                  ref={imgRef}
                  src={`${API_BASE_URL}${editingImage.url}`}
                  style={{ maxWidth: '100%', display: 'block', margin: '0 auto', cursor: 'crosshair' }}
                  onClick={handleCanvasClick}
                />
                {maskUrl && (
                  <img
                    src={maskUrl}
                    style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)', width: imgRef.current?.clientWidth, height: imgRef.current?.clientHeight, pointerEvents: 'none', opacity: 0.5 }}
                  />
                )}
                {points.map((p, i) => (
                  <Box key={i} sx={{ position: 'absolute', left: `${(p[0] / (metadata?.width || 1)) * (imgRef.current?.clientWidth || 1)}px`, top: `${(p[1] / (metadata?.height || 1)) * (imgRef.current?.clientHeight || 1)}px`, width: 8, height: 8, bgcolor: 'blue', borderRadius: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none', border: '2px solid white' }} />
                ))}
              </>
            )}
            {isSegmenting && (
              <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, bgcolor: 'rgba(0,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress color="inherit" />
              </Box>
            )}
          </Box>
          <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
            Click on the object you want to mask. AI will automatically segment the area.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsMaskEditorOpen(false)}>Close</Button>
          <Button variant="contained" color="success">Save Mask</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}

export default App;
