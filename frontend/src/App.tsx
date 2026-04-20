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
  Alert,
  ThemeProvider,
  createTheme,
  CssBaseline,
  MenuItem,
  Select,
  FormControl,
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
  Image as ImageIcon,
  Movie as MovieIcon,
  Folder as FolderIcon,
  CreateNewFolder as CreateNewFolderIcon,
  Refresh as RefreshIcon,
  Sync as SyncIcon,
  Clear as ClearIcon,
  Psychology as BrainIcon,
  PlayArrow as PropagateIcon,
  TextFields as TextIcon,
  Visibility as PreviewIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { FormControlLabel, Checkbox, ButtonGroup } from '@mui/material';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';

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
  manual_class?: 'sharp' | 'blur';
}

interface BatchStatus {
  current: number;
  total: number;
  message: string;
}

interface MaskInfo {
  scene_dir: string;
  mask_dir: string;
  convention: string;
  exists: boolean;
  count: number;
}

const PRESETS = {
  '3DGS': { fps: 5, threshold: 100, label: '3D Gaussian Splatting' },
  '2DGS': { fps: 2, threshold: 80, label: '2D Gaussian Splatting' },
  'COLMAP': { fps: 3, threshold: 120, label: 'COLMAP/SfM' },
};

const SAM2_MODELS = [
  { value: 'tiny',      label: 'Hiera-Tiny (Fastest)' },
  { value: 'small',     label: 'Hiera-Small (Light)' },
  { value: 'base_plus', label: 'Hiera-Base+ (Recommended)' },
  { value: 'large',     label: 'Hiera-Large (Best Quality)' },
];

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

  const [samModel, setSamModel] = useState('base_plus');
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [preset, setPreset] = useState('');
  const [fps, setFps] = useState(1);
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

  const [isMaskEditorOpen, setIsMaskEditorOpen] = useState(false);
  const [editingImage, setEditingImage] = useState<ProcessResult | null>(null);
  const [points, setPoints] = useState<number[][]>([]);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // ---- Phase 2: Video propagation ----
  const [propagating, setPropagating] = useState(false);
  const [propagateStatus, setPropagateStatus] = useState<BatchStatus | null>(null);

  // ---- Phase 4: Text-prompt masking ----
  const [textQueries, setTextQueries] = useState('person, car');
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [textBatchRunning, setTextBatchRunning] = useState(false);
  const [textBatchStatus, setTextBatchStatus] = useState<BatchStatus | null>(null);

  // ---- Phase 3: Reconstruction mask info ----
  const [maskInfo, setMaskInfo] = useState<MaskInfo | null>(null);

  // ---- COMBINE mode + mask preview viewer ----
  const [combineMode, setCombineMode] = useState(true);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewItems, setPreviewItems] = useState<Array<{ filename: string; overlay_url: string | null; coverage: number }>>([]);
  const [previewPage, setPreviewPage] = useState(0);
  const PREVIEW_PAGE_SIZE = 12;

  const fetchDirectories = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/directories`);
      setDirectories(res.data);
      if (!outputPath && res.data.path) setOutputPath(res.data.path);
    } catch (err) { console.error(err); }
  };

  useEffect(() => { fetchDirectories(); }, []);

  const handleSamModelChange = async (newModel: string) => {
    setIsModelLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/change-model/${newModel}`);
      setSamModel(newModel);
    } catch (err: any) { setError(`MODEL_LOAD_ERROR: ${err.message}`); } finally { setIsModelLoading(false); }
  };

  const applyPreset = (key: keyof typeof PRESETS) => {
    const p = PRESETS[key];
    setPreset(key); setFps(p.fps); setThreshold(p.threshold);
  };

  const handleCreateFolder = async () => {
    if (!newFolderName) return;
    try {
      await axios.post(`${API_BASE_URL}/create-directory`, { parent_path: targetParentPath, new_name: newFolderName });
      setNewFolderName(''); setIsDialogOpen(false); fetchDirectories();
    } catch (err: any) { alert(`Folder creation failed: ${err.message}`); }
  };

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
    setProcessing(true); setResults([]); setServerOutputDir(''); setError(null); setCurrentProgress(0); setProgressMsg('CONNECTING...');
    setMaskInfo(null); setPropagateStatus(null); setTextBatchStatus(null);
    const ws = new WebSocket(`${WS_BASE_URL}/ws/process/${fileId}`);
    ws.onopen = () => ws.send(JSON.stringify({ fps, threshold, output_path: outputPath }));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status' || data.type === 'progress') {
        setProgressMsg(data.message);
        if (data.current) setCurrentProgress(data.current);
      } else if (data.type === 'complete') {
        setResults(data.results); setServerOutputDir(data.output_dir); setProcessing(false); fetchDirectories(); ws.close();
      } else if (data.type === 'error') {
        setError(`PROCESS_ERROR: ${data.message}`); setProcessing(false); ws.close();
      }
    };
  };

  const handleSyncFolders = async () => {
    if (!serverOutputDir || results.length === 0) return;
    setSyncing(true);
    try {
      await axios.post(`${API_BASE_URL}/sync-folders`, { output_dir: serverOutputDir, threshold, results });
      fetchDirectories();
      alert('Folders synchronized!');
    } catch (err: any) { setError(`SYNC_ERROR: ${err.message}`); } finally { setSyncing(false); }
  };

  const handleImageClick = (res: ProcessResult) => {
    setEditingImage(res); setPoints([]); setMaskUrl(null); setIsMaskEditorOpen(true);
  };

  const runSegment = async (pts: number[][]) => {
    if (!editingImage?.full_path) return;
    if (pts.length === 0) { setMaskUrl(null); return; }
    setIsSegmenting(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/segment`, {
        image_path: editingImage.full_path,
        points: pts,
        labels: new Array(pts.length).fill(1),
      });
      setMaskUrl(`${API_BASE_URL}${res.data.mask_url}?t=${Date.now()}`);
    } catch (err) { console.error(err); } finally { setIsSegmenting(false); }
  };

  const handleCanvasClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current || !editingImage) return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * (metadata?.width || 1920));
    const y = Math.round(((e.clientY - rect.top) / rect.height) * (metadata?.height || 1080));
    const newPoints = [...points, [x, y]];
    setPoints(newPoints);
    await runSegment(newPoints);
  };

  const handleRemovePoint = async (idx: number) => {
    const newPoints = points.filter((_, i) => i !== idx);
    setPoints(newPoints);
    await runSegment(newPoints);
  };

  // --------------------------------------------------- Phase 2: propagate
  const handlePropagate = () => {
    if (!editingImage?.full_path || points.length === 0 || !serverOutputDir) return;

    // frames_dir = parent directory of the clicked image
    const parts = editingImage.full_path.split('/');
    const fname = parts.pop()!;
    const framesDir = parts.join('/');

    // init_frame_idx = index of this frame among sorted frames in same dir
    const siblings = results
      .filter(r => r.full_path && r.full_path.startsWith(framesDir + '/'))
      .map(r => r.filename)
      .sort();
    const initFrameIdx = Math.max(0, siblings.indexOf(fname));

    setPropagating(true);
    setPropagateStatus({ current: 0, total: siblings.length, message: 'CONNECTING' });

    const ws = new WebSocket(`${WS_BASE_URL}/ws/segment-video`);
    ws.onopen = () => ws.send(JSON.stringify({
      frames_dir: framesDir,
      init_frame_idx: initFrameIdx,
      points: points,
      labels: new Array(points.length).fill(1),
      scene_dir: serverOutputDir,
      invert_mask: true,
      write_overlay: false,
      combine: combineMode,
      skip_empty: true,
    }));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status') {
        setPropagateStatus(s => ({ current: s?.current || 0, total: data.total ?? s?.total ?? 0, message: data.message }));
      } else if (data.type === 'progress') {
        setPropagateStatus({ current: data.current, total: data.total, message: 'PROPAGATING' });
      } else if (data.type === 'complete') {
        setPropagateStatus({ current: data.frame_count, total: data.frame_count, message: `DONE: ${data.frame_count} masks written` });
        setPropagating(false);
        setIsMaskEditorOpen(false);
        ws.close();
        refreshMaskInfo();
        handleOpenPreview();
      } else if (data.type === 'error') {
        setError(`PROPAGATE_ERROR: ${data.message}`);
        setPropagating(false);
        ws.close();
      }
    };
    ws.onerror = () => { setError('PROPAGATE_WS_ERROR'); setPropagating(false); };
  };

  // --------------------------------------------------- Phase 4: text batch
  const handleTextBatch = () => {
    if (!serverOutputDir || !textQueries.trim()) return;
    const queries = textQueries.split(',').map(q => q.trim()).filter(Boolean);
    if (queries.length === 0) return;

    const framesDir = `${serverOutputDir}/sharp`;

    setTextBatchRunning(true);
    setTextBatchStatus({ current: 0, total: 0, message: 'CONNECTING' });

    const ws = new WebSocket(`${WS_BASE_URL}/ws/segment-text-batch`);
    ws.onopen = () => ws.send(JSON.stringify({
      frames_dir: framesDir,
      queries,
      scene_dir: serverOutputDir,
      box_threshold: boxThreshold,
      text_threshold: textThreshold,
      invert_mask: true,
      combine: combineMode,
      skip_empty: true,
    }));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status') {
        setTextBatchStatus(s => ({ current: s?.current || 0, total: data.total ?? s?.total ?? 0, message: data.message }));
      } else if (data.type === 'progress') {
        setTextBatchStatus({ current: data.current, total: data.total, message: `PROCESSING ${data.filename || ''}` });
      } else if (data.type === 'complete') {
        setTextBatchStatus({ current: data.frame_count, total: data.frame_count, message: `DONE: ${data.frame_count} masks` });
        setTextBatchRunning(false);
        ws.close();
        refreshMaskInfo();
        handleOpenPreview();
      } else if (data.type === 'error') {
        setError(`TEXT_BATCH_ERROR: ${data.message}`);
        setTextBatchRunning(false);
        ws.close();
      }
    };
    ws.onerror = () => { setError('TEXT_BATCH_WS_ERROR'); setTextBatchRunning(false); };
  };

  // --------------------------------------------------- Phase 3: mask info
  const refreshMaskInfo = async () => {
    if (!serverOutputDir) return;
    try {
      const res = await axios.post(`${API_BASE_URL}/reconstruction-mask-info`, {
        scene_dir: serverOutputDir,
        preset: (preset || '3DGS').toLowerCase(),
      });
      setMaskInfo(res.data);
    } catch (err: any) { setError(`MASK_INFO_ERROR: ${err.message}`); }
  };

  useEffect(() => { if (serverOutputDir) refreshMaskInfo(); }, [serverOutputDir]);

  const handleOpenPreview = async () => {
    if (!serverOutputDir) return;
    setPreviewOpen(true);
    setPreviewLoading(true);
    setPreviewPage(0);
    try {
      const res = await axios.post(`${API_BASE_URL}/generate-mask-previews`, {
        scene_dir: serverOutputDir,
        frames_subdir: 'sharp',
      });
      setPreviewItems(res.data.items || []);
    } catch (err: any) {
      setError(`PREVIEW_ERROR: ${err.message}`);
    } finally {
      setPreviewLoading(false);
    }
  };

  const isBlurryView = useCallback(
    (r: ProcessResult) => (r.manual_class ? r.manual_class === 'blur' : r.score < threshold),
    [threshold]
  );

  const analyticsData = useMemo(() => {
    if (results.length === 0) return null;
    const sharp = results.filter(r => !isBlurryView(r));
    const blur = results.filter(r => isBlurryView(r));
    const avgAll = results.reduce((acc, curr) => acc + curr.score, 0) / results.length;
    const avgSharp = sharp.length > 0 ? sharp.reduce((acc, curr) => acc + curr.score, 0) / sharp.length : 0;
    return { sharpCount: sharp.length, blurCount: blur.length, avgAll: parseFloat(avgAll.toFixed(2)), avgSharp: parseFloat(avgSharp.toFixed(2)) };
  }, [results, isBlurryView]);

  const filteredResults = useMemo(() => {
    if (tabValue === 0) return results;
    if (tabValue === 1) return results.filter(r => !isBlurryView(r));
    return results.filter(r => isBlurryView(r));
  }, [results, isBlurryView, tabValue]);

  const navigateImage = useCallback((direction: 1 | -1) => {
    if (!editingImage || filteredResults.length === 0) return;
    const idx = filteredResults.findIndex(r => r.full_path === editingImage.full_path);
    if (idx < 0) return;
    const nextIdx = (idx + direction + filteredResults.length) % filteredResults.length;
    const next = filteredResults[nextIdx];
    setEditingImage(next);
    setPoints([]);
    setMaskUrl(null);
  }, [editingImage, filteredResults]);

  const handleReclassify = async (target: 'sharp' | 'blur') => {
    if (!editingImage?.full_path || !serverOutputDir) return;
    try {
      const res = await axios.post(`${API_BASE_URL}/reclassify`, {
        full_path: editingImage.full_path,
        target,
        output_dir: serverOutputDir,
      });
      const updated: ProcessResult = {
        ...editingImage,
        full_path: res.data.full_path,
        url: res.data.url ?? editingImage.url,
        manual_class: target,
      };
      setResults(prev => prev.map(r => (r.full_path === editingImage.full_path ? updated : r)));
      setEditingImage(updated);
    } catch (err: any) {
      setError(`RECLASSIFY_ERROR: ${err.message}`);
    }
  };

  useEffect(() => {
    if (!isMaskEditorOpen) return;
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
      if (propagating || isSegmenting) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateImage(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); navigateImage(-1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isMaskEditorOpen, navigateImage, propagating, isSegmenting]);

  const renderTree = (node: any) => (
    <TreeItem key={node.path} itemId={node.path} label={
      <Box sx={{ display: 'flex', alignItems: 'center', py: 0.5 }}>
        <FolderIcon sx={{ mr: 1, fontSize: 18, color: '#D4CBB3' }} />
        <Typography variant="caption" sx={{ flexGrow: 1 }}>{node.name}</Typography>
        <IconButton size="small" onClick={(e) => { e.stopPropagation(); setTargetParentPath(node.path); setIsDialogOpen(true); }}><CreateNewFolderIcon sx={{ fontSize: 16 }} /></IconButton>
      </Box>
    }>{Array.isArray(node.children) ? node.children.map((child: any) => renderTree(child)) : null}</TreeItem>
  );

  const propagatePct = propagateStatus && propagateStatus.total > 0
    ? (propagateStatus.current / propagateStatus.total) * 100 : 0;
  const textBatchPct = textBatchStatus && textBatchStatus.total > 0
    ? (textBatchStatus.current / textBatchStatus.total) * 100 : 0;

  return (
    <ThemeProvider theme={sandTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', pb: 8 }}>
        <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid #E6E1D6', mb: 4, bgcolor: '#FFFFFF' }}>
          <Toolbar>
            <Typography variant="h6" color="primary" sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
              VIDEO_TO_IMAGE_AI_PIPELINE <Chip label="V4.0.0" size="small" sx={{ ml: 1, height: 20, fontSize: '10px' }} />
            </Typography>
            {metadata && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="caption" color={metadata.ai_ready ? "success.main" : "error.main"} sx={{ fontWeight: 'bold' }}>
                  {metadata.ai_ready ? "● SAM2_ONLINE" : "○ SAM2_OFFLINE"}
                </Typography>
                <Typography variant="caption" color="success.main">● PIPELINE_READY</Typography>
              </Box>
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl">
          <Grid container spacing={3}>
            {/* Left Column: Input & Config */}
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
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.8 }}>
                      {[
                        { label: 'FILE_NAME', value: file.name },
                        { label: 'FILE_SIZE', value: `${(file.size / (1024 * 1024)).toFixed(2)} MB` },
                        ...(metadata ? [
                          { label: 'RESOLUTION', value: `${metadata.width} x ${metadata.height}` },
                          { label: 'VIDEO_FPS', value: metadata.avg_frame_rate },
                          { label: 'LENGTH', value: `${metadata.duration.toFixed(2)}s` },
                          { label: 'ASPECT_RATIO', value: `${(metadata.width / metadata.height).toFixed(2)}:1` }
                        ] : [])
                      ].map((item, idx) => (
                        <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #F0EDE5', pb: 0.5 }}>
                          <Typography variant="caption" color="textSecondary" sx={{ fontWeight: 'bold' }}>{item.label}:</Typography>
                          <Typography variant="caption" sx={{ color: 'primary.main', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.value}</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>

              <Paper sx={{ p: 3, bgcolor: '#FDFCFB' }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[02] PIPELINE_CONFIGURATION</Typography>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', mb: 1 }}><BrainIcon sx={{ fontSize: 16, mr: 0.5 }} /> ENGINE: SAM_2.1_MODEL</Typography>
                  <FormControl fullWidth size="small">
                    <Select value={samModel} onChange={(e) => handleSamModelChange(e.target.value)} disabled={isModelLoading}>
                      {SAM2_MODELS.map(m => (
                        <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  {isModelLoading && <LinearProgress sx={{ mt: 1 }} />}
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>--- PRESETS ---</Typography>
                  <FormControl fullWidth size="small">
                    <Select value={preset} onChange={(e) => applyPreset(e.target.value as keyof typeof PRESETS)} displayEmpty>
                      <MenuItem value="" disabled>Select Pipeline Preset</MenuItem>
                      {Object.entries(PRESETS).map(([key, p]) => (
                        <MenuItem key={key} value={key}><Box><Typography variant="body2" fontWeight="bold">{key}</Typography><Typography variant="caption" color="textSecondary">{p.label}</Typography></Box></MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', mb: 1 }}>--- OUTPUT_DIR --- <IconButton size="small" onClick={fetchDirectories} sx={{ ml: 1 }}><RefreshIcon sx={{ fontSize: 14 }} /></IconButton></Typography>
                  <Box sx={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid #E6E1D6', p: 1, mb: 1, bgcolor: '#FFFFFF' }}>
                    {directories ? <SimpleTreeView onSelectedItemsChange={(_, id) => setOutputPath(id as string)} selectedItems={outputPath}>{renderTree(directories)}</SimpleTreeView> : <CircularProgress size={20} />}
                  </Box>
                  <TextField fullWidth label="Path" value={outputPath} size="small" variant="filled" slotProps={{ input: { readOnly: true } }} />
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>--- EXTRACTION_PARAMS ---</Typography>
                  <Grid container spacing={1}>
                    <Grid item xs={6}><TextField fullWidth label="FPS" type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} size="small" /></Grid>
                    <Grid item xs={6}><TextField fullWidth label="Threshold" type="number" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} size="small" /></Grid>
                  </Grid>
                  <Slider value={threshold} onChange={(_, v) => setThreshold(v as number)} min={0} max={500} step={5} sx={{ mt: 1 }} />
                  <Typography variant="caption" sx={{ bgcolor: '#F4F1EA', p: 0.5, display: 'block', fontWeight: 'bold', borderRadius: 0.5 }}>
                    GUIDE: {threshold < 50 ? "Lenient (Blurry ok)" : threshold < 150 ? "Moderate (Standard)" : "Strict (Sharp only)"}
                  </Typography>
                </Box>
                <Button fullWidth variant="contained" size="large" sx={{ py: 1.5, bgcolor: '#4A4238', mb: 2 }} onClick={handleProcess} disabled={!fileId || processing || isModelLoading}>
                  {processing ? <CircularProgress size={24} color="inherit" /> : 'START_PIPELINE'}
                </Button>
                {results.length > 0 && (
                  <Button fullWidth variant="outlined" startIcon={syncing ? <CircularProgress size={16}/> : <SyncIcon />} onClick={handleSyncFolders} disabled={syncing}>
                    {syncing ? 'SYNCING...' : 'APPLY & SYNC FOLDERS'}
                  </Button>
                )}
              </Paper>
            </Grid>

            {/* Main Content Area */}
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, minHeight: '85vh', borderLeft: '4px solid #4A4238' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="subtitle2" color="textSecondary">[03] BUFFER & ANALYTICS</Typography>
                  {analyticsData && (
                    <Box sx={{ display: 'flex', gap: 2 }}>
                      <Box sx={{ width: 220, height: 130, border: '1px solid #E6E1D6', p: 1, bgcolor: '#FDFCFB' }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5, fontSize: '9px' }}>COUNT_RATIO</Typography>
                        <ResponsiveContainer width="100%" height="85%">
                          <BarChart data={[{label:'Sharp',val:analyticsData.sharpCount},{label:'Blur',val:analyticsData.blurCount}]} layout="vertical" margin={{ left: 5, right: 40 }}>
                            <XAxis type="number" hide />
                            <YAxis type="category" dataKey="label" fontSize={9} tickLine={false} axisLine={false} width={45} />
                            <Bar dataKey="val" radius={[0, 4, 4, 0]} barSize={15}>
                              <Cell fill="#6B7A5F"/><Cell fill="#A65D57"/>
                              <LabelList dataKey="val" position="right" fontSize={9} />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                      <Box sx={{ width: 220, height: 130, border: '1px solid #E6E1D6', p: 1, bgcolor: '#FDFCFB' }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5, fontSize: '9px' }}>QUALITY_PREVIEW (AVG)</Typography>
                        <ResponsiveContainer width="100%" height="85%">
                          <BarChart data={[{label:'AllAvg',val:analyticsData.avgAll},{label:'SharpAvg',val:analyticsData.avgSharp}]} layout="vertical" margin={{ left: 5, right: 40 }}>
                            <XAxis type="number" hide />
                            <YAxis type="category" dataKey="label" fontSize={9} tickLine={false} axisLine={false} width={45} />
                            <Bar dataKey="val" radius={[0, 4, 4, 0]} barSize={15}>
                              <Cell fill="#D4CBB3"/><Cell fill="#6B7A5F"/>
                              <LabelList dataKey="val" position="right" fontSize={9} />
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </Box>
                  )}
                </Box>

                {error && <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>{error}</Alert>}
                {processing && (
                  <Box sx={{ mb: 3, p: 2, bgcolor: '#FDFCFB', border: '1px solid #E6E1D6' }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>{progressMsg}</Typography>
                    <LinearProgress variant="determinate" value={Math.min((currentProgress / (totalEstimated || 1)) * 100, 100)} sx={{ height: 10, borderRadius: 1 }} />
                  </Box>
                )}

                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                  <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} textColor="primary" indicatorColor="primary">
                    <Tab label={`ALL (${results.length})`} sx={{ fontSize: '11px' }} />
                    <Tab label={`SHARP (${analyticsData?.sharpCount || 0})`} sx={{ fontSize: '11px' }} />
                    <Tab label={`BLUR (${analyticsData?.blurCount || 0})`} sx={{ fontSize: '11px' }} />
                  </Tabs>
                </Box>

                <Grid container spacing={1.5}>
                  {results.length > 0 ? (
                    filteredResults.map((res, index) => {
                      const blurry = isBlurryView(res);
                      return (
                        <Grid item xs={6} sm={4} md={3} lg={2.4} key={index}>
                          <Card variant="outlined" onClick={() => handleImageClick(res)} sx={{ cursor: 'pointer', opacity: blurry ? 0.6 : 1, '&:hover': { borderColor: 'primary.main' } }}>
                            <CardMedia component="img" height="100" image={`${API_BASE_URL}${res.url}`} sx={{ filter: blurry ? 'grayscale(80%)' : 'none' }} />
                            <CardContent sx={{ p: 0.8 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" sx={{ fontSize: '9px', fontWeight: 'bold' }}>S:{res.score}</Typography>
                                <Chip label={res.manual_class ? (blurry ? 'B*' : 'S*') : (blurry ? 'B' : 'S')} size="small" color={blurry ? 'error' : 'success'} sx={{ height: 12, fontSize: '7px', minWidth: 16 }} />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })
                  ) : !processing && <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', opacity: 0.2 }}><ImageIcon sx={{ fontSize: 64 }} /></Box>}
                </Grid>
              </Paper>

              {/* ======================================================== */}
              {/* [04] RECONSTRUCTION MASKING (SAM 2.1 + Grounded-SAM 2)    */}
              {/* ======================================================== */}
              {serverOutputDir && (
                <Paper sx={{ p: 3, mt: 3, bgcolor: '#FDFCFB' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2" color="textSecondary">
                      [04] RECONSTRUCTION_MASKING
                    </Typography>
                    <FormControlLabel
                      control={
                        <Checkbox
                          size="small"
                          checked={combineMode}
                          onChange={(e) => setCombineMode(e.target.checked)}
                          disabled={propagating || textBatchRunning}
                        />
                      }
                      label={
                        <Typography variant="caption" sx={{ fontSize: '10px' }}>
                          COMBINE with existing masks (AND)
                        </Typography>
                      }
                    />
                  </Box>

                  <Grid container spacing={2}>
                    {/* ---- Phase 2: Video Propagation ---- */}
                    <Grid item xs={12} md={6}>
                      <Box sx={{ border: '1px solid #E6E1D6', p: 2, height: '100%' }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', mb: 1 }}>
                          <PropagateIcon sx={{ fontSize: 14, mr: 0.5 }} /> VIDEO_PROPAGATION
                        </Typography>
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 1.5, fontSize: '10px' }}>
                          1. Click any frame above → open editor<br/>
                          2. Click object(s) → points captured<br/>
                          3. Press <b>PROPAGATE</b> → masks written across sequence
                        </Typography>
                        {propagateStatus && (
                          <Box sx={{ mt: 1, p: 1, bgcolor: '#FFFFFF', border: '1px solid #E6E1D6' }}>
                            <Typography variant="caption" sx={{ fontSize: '10px', fontWeight: 'bold' }}>
                              {propagateStatus.message} ({propagateStatus.current}/{propagateStatus.total})
                            </Typography>
                            <LinearProgress variant="determinate" value={propagatePct} sx={{ mt: 0.5 }} />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* ---- Phase 4: Text-prompt Batch ---- */}
                    <Grid item xs={12} md={6}>
                      <Box sx={{ border: '1px solid #E6E1D6', p: 2, height: '100%' }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', mb: 1 }}>
                          <TextIcon sx={{ fontSize: 14, mr: 0.5 }} /> TEXT_PROMPT_MASK (Grounded-SAM 2)
                        </Typography>
                        <TextField
                          size="small" fullWidth
                          label="Queries (comma-separated)"
                          value={textQueries}
                          onChange={e => setTextQueries(e.target.value)}
                          placeholder="person, car, sky"
                          sx={{ mb: 1 }}
                        />
                        <Grid container spacing={1} sx={{ mb: 1 }}>
                          <Grid item xs={6}>
                            <TextField
                              size="small" fullWidth type="number"
                              label="Box Thr"
                              value={boxThreshold}
                              onChange={e => setBoxThreshold(Number(e.target.value))}
                              inputProps={{ step: 0.05, min: 0, max: 1 }}
                            />
                          </Grid>
                          <Grid item xs={6}>
                            <TextField
                              size="small" fullWidth type="number"
                              label="Text Thr"
                              value={textThreshold}
                              onChange={e => setTextThreshold(Number(e.target.value))}
                              inputProps={{ step: 0.05, min: 0, max: 1 }}
                            />
                          </Grid>
                        </Grid>
                        <Button
                          fullWidth variant="contained" size="small"
                          onClick={handleTextBatch}
                          disabled={textBatchRunning || !textQueries.trim()}
                          sx={{ bgcolor: '#4A4238' }}
                        >
                          {textBatchRunning ? <CircularProgress size={16} color="inherit" /> : 'RUN ON SHARP/'}
                        </Button>
                        {textBatchStatus && (
                          <Box sx={{ mt: 1, p: 1, bgcolor: '#FFFFFF', border: '1px solid #E6E1D6' }}>
                            <Typography variant="caption" sx={{ fontSize: '10px', fontWeight: 'bold' }}>
                              {textBatchStatus.message} ({textBatchStatus.current}/{textBatchStatus.total})
                            </Typography>
                            <LinearProgress variant="determinate" value={textBatchPct} sx={{ mt: 0.5 }} />
                          </Box>
                        )}
                      </Box>
                    </Grid>

                    {/* ---- Phase 3: Mask Status ---- */}
                    <Grid item xs={12}>
                      <Box sx={{ border: '1px solid #E6E1D6', p: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                            MASK_STATUS (preset: {preset || '3DGS'})
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Button
                              size="small"
                              startIcon={<PreviewIcon sx={{ fontSize: 14 }} />}
                              onClick={handleOpenPreview}
                              disabled={!maskInfo || maskInfo.count === 0}
                              variant="outlined"
                            >
                              PREVIEW MASKS
                            </Button>
                            <Button size="small" startIcon={<RefreshIcon sx={{ fontSize: 14 }} />} onClick={refreshMaskInfo}>
                              REFRESH
                            </Button>
                          </Box>
                        </Box>
                        {maskInfo ? (
                          <Box>
                            <Typography variant="caption" sx={{ display: 'block', fontSize: '10px' }}>
                              <b>PATH:</b> {maskInfo.mask_dir}
                            </Typography>
                            <Typography variant="caption" sx={{ display: 'block', fontSize: '10px' }}>
                              <b>COUNT:</b> {maskInfo.count} masks {maskInfo.exists ? '(ready)' : '(not generated yet)'}
                            </Typography>
                            <Typography variant="caption" sx={{ display: 'block', fontSize: '10px', color: 'text.secondary' }}>
                              <b>CONVENTION:</b> {maskInfo.convention}
                            </Typography>
                          </Box>
                        ) : (
                          <Typography variant="caption" color="textSecondary">No mask info yet — run propagation or text masking.</Typography>
                        )}
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              )}
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* New Folder Dialog */}
      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <DialogTitle sx={{ fontSize: '14px', fontWeight: 'bold' }}>CREATE_NEW_FOLDER</DialogTitle>
        <DialogContent><TextField autoFocus margin="dense" label="Folder Name" fullWidth size="small" value={newFolderName} onChange={(e) => setNewFolderName(e.target.value)} /></DialogContent>
        <DialogActions><Button onClick={() => setIsDialogOpen(false)}>CANCEL</Button><Button onClick={handleCreateFolder} variant="contained">CREATE</Button></DialogActions>
      </Dialog>

      {/* Masking Editor Dialog */}
      <Dialog open={isMaskEditorOpen} onClose={() => !propagating && setIsMaskEditorOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          AI MASKING EDITOR (SAM 2.1)
          <Box>
            <Button size="small" startIcon={<ClearIcon />} onClick={() => { setPoints([]); setMaskUrl(null); }} disabled={propagating}>Reset</Button>
            <IconButton onClick={() => setIsMaskEditorOpen(false)} disabled={propagating}><ClearIcon /></IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ position: 'relative', textAlign: 'center', bgcolor: '#000', borderRadius: 1, overflow: 'hidden' }}>
            {editingImage && (
              <>
                <img ref={imgRef} src={`${API_BASE_URL}${editingImage.url}`} style={{ maxWidth: '100%', display: 'block', margin: '0 auto', cursor: 'crosshair' }} onClick={handleCanvasClick} />
                {maskUrl && <img src={maskUrl} style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)', width: imgRef.current?.clientWidth, height: imgRef.current?.clientHeight, pointerEvents: 'none', opacity: 0.6 }} />}
                {points.map((p, i) => {
                  const left = (p[0] / (metadata?.width || 1)) * (imgRef.current?.clientWidth || 1);
                  const top = (p[1] / (metadata?.height || 1)) * (imgRef.current?.clientHeight || 1);
                  return (
                    <React.Fragment key={i}>
                      {/* exact-click dot (small, so tiny objects remain visible) */}
                      <Box sx={{
                        position: 'absolute',
                        left: `${left}px`,
                        top: `${top}px`,
                        width: 7, height: 7,
                        bgcolor: 'rgba(255, 235, 59, 1)',
                        border: '1.5px solid #1565c0',
                        borderRadius: '50%',
                        transform: 'translate(-50%, -50%)',
                        pointerEvents: 'none',
                        boxShadow: '0 0 2px rgba(0,0,0,0.9)',
                      }} />
                      {/* offset numbered badge (doesn't cover the target) */}
                      <Box sx={{
                        position: 'absolute',
                        left: `${left + 8}px`,
                        top: `${top - 8}px`,
                        minWidth: 14, height: 14, px: '2px',
                        bgcolor: 'rgba(25, 118, 210, 0.95)',
                        color: 'white',
                        fontSize: '9px',
                        fontWeight: 'bold',
                        borderRadius: '7px',
                        border: '1px solid white',
                        pointerEvents: 'none',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        boxShadow: '0 0 2px rgba(0,0,0,0.6)',
                        lineHeight: 1,
                      }}>
                        {i + 1}
                      </Box>
                    </React.Fragment>
                  );
                })}
              </>
            )}
            {(isSegmenting || propagating) && <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, bgcolor: 'rgba(0,0,0,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><CircularProgress color="inherit" /></Box>}
          </Box>
          {points.length > 0 && (
            <Box sx={{ mt: 2, p: 1.5, bgcolor: '#F4F1EA', border: '1px solid #E6E1D6' }}>
              <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
                POINTS ({points.length}) — click × to remove
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {points.map((p, i) => (
                  <Chip
                    key={i}
                    size="small"
                    label={`#${i + 1}  (${p[0]}, ${p[1]})`}
                    onDelete={() => handleRemovePoint(i)}
                    disabled={propagating || isSegmenting}
                    sx={{ fontSize: '10px', bgcolor: '#FFFFFF' }}
                  />
                ))}
              </Box>
            </Box>
          )}
          {propagateStatus && propagating && (
            <Box sx={{ mt: 2, p: 1, bgcolor: '#F4F1EA' }}>
              <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                {propagateStatus.message} — {propagateStatus.current}/{propagateStatus.total}
              </Typography>
              <LinearProgress variant="determinate" value={propagatePct} sx={{ mt: 0.5 }} />
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ flexWrap: 'wrap', gap: 1, justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton
              size="small"
              onClick={() => navigateImage(-1)}
              disabled={propagating || isSegmenting || filteredResults.length < 2}
            >
              <ChevronLeftIcon />
            </IconButton>
            <Typography variant="caption" sx={{ minWidth: 70, textAlign: 'center', fontSize: '11px' }}>
              {editingImage
                ? `${Math.max(0, filteredResults.findIndex(r => r.full_path === editingImage.full_path)) + 1} / ${filteredResults.length}`
                : '— / —'}
            </Typography>
            <IconButton
              size="small"
              onClick={() => navigateImage(1)}
              disabled={propagating || isSegmenting || filteredResults.length < 2}
            >
              <ChevronRightIcon />
            </IconButton>
            {editingImage && serverOutputDir && (
              <ButtonGroup size="small" variant="outlined" sx={{ ml: 2 }}>
                <Button
                  variant={!isBlurryView(editingImage) ? 'contained' : 'outlined'}
                  color="success"
                  onClick={() => handleReclassify('sharp')}
                  disabled={propagating || isSegmenting}
                >
                  SHARP
                </Button>
                <Button
                  variant={isBlurryView(editingImage) ? 'contained' : 'outlined'}
                  color="error"
                  onClick={() => handleReclassify('blur')}
                  disabled={propagating || isSegmenting}
                >
                  BLUR
                </Button>
              </ButtonGroup>
            )}
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button onClick={() => setIsMaskEditorOpen(false)} disabled={propagating}>Close</Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={<PropagateIcon />}
              onClick={handlePropagate}
              disabled={propagating || points.length === 0 || !serverOutputDir}
            >
              {propagating ? 'PROPAGATING...' : 'PROPAGATE TO SEQUENCE'}
            </Button>
          </Box>
        </DialogActions>
      </Dialog>

      {/* Mask Preview Dialog */}
      <Dialog open={previewOpen} onClose={() => setPreviewOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            MASK PREVIEW
            <Typography variant="caption" color="textSecondary" sx={{ ml: 2, fontSize: '11px' }}>
              red tint = excluded region ({previewItems.length} masked frames)
            </Typography>
          </Box>
          <IconButton onClick={() => setPreviewOpen(false)}><ClearIcon /></IconButton>
        </DialogTitle>
        <DialogContent>
          {previewLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}>
              <CircularProgress />
            </Box>
          ) : previewItems.length === 0 ? (
            <Alert severity="info">No masks found. Run PROPAGATE or TEXT_PROMPT_MASK first.</Alert>
          ) : (
            <>
              <Grid container spacing={1.5}>
                {previewItems
                  .slice(previewPage * PREVIEW_PAGE_SIZE, (previewPage + 1) * PREVIEW_PAGE_SIZE)
                  .map((it, i) => (
                    <Grid item xs={6} sm={4} md={3} key={i}>
                      <Card variant="outlined">
                        {it.overlay_url ? (
                          <CardMedia
                            component="img"
                            image={`${API_BASE_URL}${it.overlay_url}?t=${Date.now()}`}
                            sx={{ height: 140, objectFit: 'cover' }}
                          />
                        ) : (
                          <Box sx={{ height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#F4F1EA' }}>
                            <Typography variant="caption" color="textSecondary">
                              scene is outside OUTPUT_DIR
                            </Typography>
                          </Box>
                        )}
                        <CardContent sx={{ p: 0.8 }}>
                          <Typography variant="caption" sx={{ fontSize: '9px', display: 'block', fontWeight: 'bold' }}>
                            {it.filename}
                          </Typography>
                          <Typography variant="caption" sx={{ fontSize: '9px', color: 'text.secondary' }}>
                            EXCL: {(it.coverage * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
              </Grid>
              {previewItems.length > PREVIEW_PAGE_SIZE && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2, mt: 2 }}>
                  <Button
                    size="small"
                    disabled={previewPage === 0}
                    onClick={() => setPreviewPage(p => Math.max(0, p - 1))}
                  >
                    PREV
                  </Button>
                  <Typography variant="caption">
                    PAGE {previewPage + 1} / {Math.ceil(previewItems.length / PREVIEW_PAGE_SIZE)}
                  </Typography>
                  <Button
                    size="small"
                    disabled={(previewPage + 1) * PREVIEW_PAGE_SIZE >= previewItems.length}
                    onClick={() => setPreviewPage(p => p + 1)}
                  >
                    NEXT
                  </Button>
                </Box>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleOpenPreview} startIcon={<RefreshIcon />}>REGENERATE</Button>
          <Button onClick={() => setPreviewOpen(false)}>CLOSE</Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}

export default App;
