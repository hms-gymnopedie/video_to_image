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
} from '@mui/icons-material';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080';

// Claude CLI inspired "Sand" Theme
const sandTheme = createTheme({
  palette: {
    mode: 'light',
    background: {
      default: '#F7F5F0', 
      paper: '#FFFFFF',
    },
    primary: {
      main: '#4A4238', 
    },
    secondary: {
      main: '#D4CBB3', 
    },
    success: {
      main: '#6B7A5F', 
    },
    error: {
      main: '#A65D57', 
    },
    text: {
      primary: '#2C2723',
      secondary: '#665E57',
    },
  },
  typography: {
    fontFamily: '"JetBrains Mono", "Roboto Mono", monospace',
    h6: { fontWeight: 700, letterSpacing: '-0.5px' },
  },
  shape: {
    borderRadius: 4,
  },
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

  // Settings
  const [preset, setPreset] = useState('');
  const [fps, setFps] = useState(1);
  const [scale, setScale] = useState('-1:-1');
  const [scaleOption, setScaleOption] = useState('original');
  const [qscale, setQscale] = useState(2);
  const [namingRule, setNamingRule] = useState('frame_%04d.jpg');
  const [threshold, setThreshold] = useState(100);
  const [outputPath, setOutputPath] = useState('');

  // Directory Explorer
  const [directories, setDirectories] = useState<DirectoryNode | null>(null);
  const [newFolderName, setNewFolderName] = useState('');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [targetParentPath, setTargetParentPath] = useState('');

  const fetchDirectories = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/directories`);
      setDirectories(res.data);
      if (!outputPath && res.data.path) {
        setOutputPath(res.data.path);
      }
    } catch (err) {
      console.error("Failed to fetch directories", err);
    }
  };

  useEffect(() => {
    fetchDirectories();
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setError(null);
      uploadVideo(selectedFile);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': [] },
    multiple: false,
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
    } catch (err: any) {
      setError(`CONNECTION_ERROR: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (key: keyof typeof PRESETS) => {
    const p = PRESETS[key];
    setPreset(key);
    setFps(p.fps);
    setQscale(p.qscale);
    setThreshold(p.threshold);
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
      await axios.post(`${API_BASE_URL}/create-directory`, {
        parent_path: targetParentPath,
        new_name: newFolderName
      });
      setNewFolderName('');
      setIsDialogOpen(false);
      fetchDirectories();
    } catch (err: any) {
      alert(`Folder creation failed: ${err.message}`);
    }
  };

  const handleProcess = async () => {
    if (!fileId) return;
    setProcessing(true);
    setResults([]);
    setError(null);
    const formData = new FormData();
    formData.append('fps', fps.toString());
    formData.append('scale', scale);
    formData.append('qscale', qscale.toString());
    formData.append('naming_rule', namingRule);
    formData.append('threshold', threshold.toString());
    formData.append('output_path', outputPath);
    try {
      const res = await axios.post(`${API_BASE_URL}/process/${fileId}`, formData);
      setResults(res.data.results);
      fetchDirectories(); // Refresh tree to show newly created subfolder if any
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Processing failed.';
      setError(`PROCESS_ERROR: ${msg}`);
    } finally {
      setProcessing(false);
    }
  };

  const getThresholdGuide = (val: number) => {
    if (val < 50) return "Very lenient (includes motion blur)";
    if (val < 100) return "Moderate (standard filtering)";
    if (val < 200) return "Strict (sharp images only)";
    return "Ultra strict (high-res static scenes)";
  };

  const renderTree = (node: DirectoryNode) => (
    <TreeItem key={node.path} itemId={node.path} label={
      <Box sx={{ display: 'flex', alignItems: 'center', py: 0.5 }}>
        <FolderIcon sx={{ mr: 1, fontSize: 18, color: '#D4CBB3' }} />
        <Typography variant="caption" sx={{ flexGrow: 1 }}>{node.name}</Typography>
        <IconButton size="small" onClick={(e) => {
          e.stopPropagation();
          setTargetParentPath(node.path);
          setIsDialogOpen(true);
        }}>
          <CreateNewFolderIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>
    }>
      {Array.isArray(node.children) ? node.children.map((child) => renderTree(child)) : null}
    </TreeItem>
  );

  return (
    <ThemeProvider theme={sandTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', pb: 8 }}>
        <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid #E6E1D6', mb: 4, bgcolor: '#FFFFFF' }}>
          <Toolbar>
            <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              VIDEO_TO_IMAGE_PIPELINE <Chip label="V1.3.7" size="small" sx={{ ml: 1, height: 20, fontSize: '10px' }} />
            </Typography>
            {metadata && <Typography variant="caption" color="success.main">● BACKEND_CONNECTED</Typography>}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl">
          <Grid container spacing={3}>
            {/* Sidebar */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, mb: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary" sx={{ mb: 2 }}>
                  [01] INPUT_SOURCE
                </Typography>
                <Box {...getRootProps()} sx={{ border: '2px dashed #D4CBB3', borderRadius: 1, p: 4, textAlign: 'center', cursor: 'pointer', bgcolor: isDragActive ? '#F4F1EA' : 'transparent', '&:hover': { borderColor: '#4A4238', bgcolor: '#F4F1EA' } }}>
                  <input {...getInputProps()} />
                  <CloudUploadIcon sx={{ fontSize: 40, color: '#D4CBB3', mb: 1 }} />
                  <Typography variant="body2">{isDragActive ? "Drop video here..." : "Drag & drop video, or click to select"}</Typography>
                </Box>
                {file && (
                  <Box sx={{ mt: 3, p: 2, bgcolor: '#FFFFFF', border: '1px solid #E6E1D6' }}>
                    <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', mb: 1.5, fontWeight: 'bold' }}>
                      <MovieIcon sx={{ mr: 1, fontSize: 18 }} /> [DATA_SUMMARY]
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {[
                        { label: 'FILE_NAME', value: file.name },
                        { label: 'FILE_SIZE', value: `${(file.size / (1024 * 1024)).toFixed(2)} MB` },
                        ...(metadata ? [
                          { label: 'RESOLUTION', value: `${metadata.width} x ${metadata.height}` },
                          { label: 'VIDEO_FPS', value: metadata.avg_frame_rate },
                          { label: 'LENGTH', value: `${metadata.duration.toFixed(2)}s` },
                          { label: 'ASPECT', value: `${(metadata.width / metadata.height).toFixed(2)}:1` }
                        ] : [])
                      ].map((item, idx) => (
                        <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #F0EDE5', pb: 0.5 }}>
                          <Typography variant="caption" color="textSecondary" sx={{ fontWeight: 'bold' }}>{item.label}:</Typography>
                          <Typography variant="caption" sx={{ color: 'primary.main', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.value}</Typography>
                        </Box>
                      ))}
                      {loading && <Box sx={{ textAlign: 'center', py: 1 }}><CircularProgress size={14} sx={{ mr: 1 }} /><Typography variant="caption" color="textSecondary">ANALYZING...</Typography></Box>}
                    </Box>
                  </Box>
                )}
              </Paper>

              <Paper sx={{ p: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary" sx={{ mb: 2.5 }}>
                  [02] CONFIGURATION
                </Typography>
                
                <Box sx={{ mb: 4 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1.5, color: '#8C8273' }}>
                    --- OUTPUT_DIRECTORY_BROWSER ---
                    <IconButton size="small" onClick={fetchDirectories} sx={{ ml: 1 }}><RefreshIcon sx={{ fontSize: 14 }} /></IconButton>
                  </Typography>
                  <Box sx={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #E6E1D6', p: 1, mb: 1, bgcolor: '#FFFFFF' }}>
                    {directories && (
                      <SimpleTreeView
                        onSelectedItemsChange={(_, itemId) => setOutputPath(itemId as string)}
                        selectedItems={outputPath}
                      >
                        {renderTree(directories)}
                      </SimpleTreeView>
                    )}
                  </Box>
                  <TextField fullWidth label="Selected Path" value={outputPath} size="small" variant="filled" InputProps={{ readOnly: true }} />
                </Box>

                <Box sx={{ mb: 4 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1.5, color: '#8C8273' }}>
                    --- PRESET_SELECTION ---
                  </Typography>
                  <FormControl fullWidth size="small"><InputLabel>Application Presets</InputLabel>
                    <Select value={preset} label="Application Presets" onChange={(e) => applyPreset(e.target.value as keyof typeof PRESETS)}>
                      {Object.entries(PRESETS).map(([key, p]) => (
                        <MenuItem key={key} value={key}><Box><Typography variant="body2" fontWeight="bold">{key}</Typography><Typography variant="caption" color="textSecondary">{p.label}</Typography></Box></MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                <Box sx={{ mb: 4 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1.5, color: '#8C8273' }}>
                    --- EXTRACTION_PARAMS ---
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}><TextField fullWidth label="Target FPS" type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} size="small" helperText="Frames per sec" /></Grid>
                    <Grid item xs={6}><TextField fullWidth label="JPEG Quality" type="number" value={qscale} onChange={(e) => setQscale(Number(e.target.value))} size="small" helperText="1(best)-31" /></Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth size="small"><InputLabel>Scale Guide</InputLabel>
                        <Select value={scaleOption} label="Scale Guide" onChange={(e) => handleScaleChange(e.target.value)}>
                          <MenuItem value="original">Original Ratio (-1:-1)</MenuItem>
                          <MenuItem value="1080p">Force 1080p (1920:1080)</MenuItem>
                          <MenuItem value="720p">Force 720p (1280:720)</MenuItem>
                          <MenuItem value="half">Half Size (Auto Scale)</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}><TextField fullWidth label="Custom Scale (w:h)" value={scale} onChange={(e) => setScale(e.target.value)} size="small" /></Grid>
                  </Grid>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1.5, color: '#8C8273' }}>
                    --- BLUR_ANALYSIS_PARAMS ---
                  </Typography>
                  <Box sx={{ px: 1 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      Threshold: {threshold}
                      <Tooltip title="Laplacian Variance. Higher = Stricter."><InfoIcon sx={{ ml: 1, fontSize: 16, color: 'text.secondary' }} /></Tooltip>
                    </Typography>
                    <Slider value={threshold} onChange={(_, v) => setThreshold(v as number)} min={0} max={500} step={5} color="primary" />
                    <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold', bgcolor: '#F4F1EA', px: 1, py: 0.5, borderRadius: 0.5 }}>GUIDE: {getThresholdGuide(threshold)}</Typography>
                  </Box>
                </Box>

                <Button fullWidth variant="contained" size="large" sx={{ mt: 4, py: 1.5, bgcolor: '#4A4238', '&:hover': { bgcolor: '#2C2723' } }} onClick={handleProcess} disabled={!fileId || processing}>
                  {processing ? <CircularProgress size={24} color="inherit" /> : 'EXEC_PROCESSING'}
                </Button>
              </Paper>
            </Grid>

            {/* Main Area */}
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, minHeight: '80vh', borderLeft: '4px solid #4A4238' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary" sx={{ mb: 3 }}>
                  [03] OUTPUT_BUFFER
                </Typography>
                {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
                {processing && (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 15 }}>
                    <CircularProgress size={40} thickness={2} sx={{ color: '#4A4238' }} />
                    <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>RUNNING_FFMPEG_EXTRACTION...</Typography>
                  </Box>
                )}
                {!processing && results.length > 0 && (
                  <>
                    <Box sx={{ mb: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      <Chip label={`TOTAL: ${results.length}`} variant="outlined" size="small" />
                      <Chip label={`ACCEPTED: ${results.filter(r => !r.is_blurry).length}`} color="success" size="small" />
                      <Chip label={`REJECTED: ${results.filter(r => r.is_blurry).length}`} color="error" size="small" variant="outlined" />
                    </Box>
                    <Grid container spacing={2}>
                      {results.map((res, index) => (
                        <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
                          <Card sx={{ borderRadius: 0, border: '1px solid #E6E1D6', '&:hover': { borderColor: '#4A4238' } }}>
                            {res.url ? (
                              <CardMedia component="img" height="120" image={`${API_BASE_URL}${res.url}`} sx={{ filter: res.is_blurry ? 'grayscale(100%) opacity(0.4)' : 'none' }} />
                            ) : (
                              <Box sx={{ height: 120, bgcolor: '#F0EDE5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><ImageIcon color="disabled" /></Box>
                            )}
                            <CardContent sx={{ p: 1.5 }}>
                              <Typography variant="caption" sx={{ display: 'block', mb: 1, color: 'text.secondary' }} noWrap>{res.filename}</Typography>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" fontWeight="bold">VAR: {res.score}</Typography>
                                <Chip label={res.is_blurry ? 'BLUR' : 'SHARP'} size="small" color={res.is_blurry ? 'error' : 'success'} sx={{ height: 16, fontSize: '9px', fontWeight: 800 }} />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </>
                )}
                {!processing && results.length === 0 && !error && (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '500px' }}>
                    <Box sx={{ textAlign: 'center', opacity: 0.3 }}><ImageIcon sx={{ fontSize: 64, mb: 1 }} /><Typography variant="body2">NO_DATA_READY</Typography></Box>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* New Folder Dialog */}
      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <DialogTitle sx={{ fontSize: '14px', fontWeight: 'bold' }}>CREATE_NEW_FOLDER</DialogTitle>
        <DialogContent>
          <Typography variant="caption" color="textSecondary" sx={{ mb: 2, display: 'block' }}>Parent: {targetParentPath}</Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Folder Name"
            fullWidth
            variant="outlined"
            size="small"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsDialogOpen(false)} sx={{ fontSize: '12px' }}>CANCEL</Button>
          <Button onClick={handleCreateFolder} variant="contained" sx={{ fontSize: '12px' }}>CREATE</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}

export default App;
