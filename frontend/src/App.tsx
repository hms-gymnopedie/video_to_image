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

type Classification = 'sharp' | 'blur' | 'drop' | 'dup';

interface ProcessResult {
  filename: string;
  url: string | null;
  score: number;
  is_blurry: boolean;
  full_path?: string;
  manual_class?: Classification;
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

// Two backends coexist behind /change-model. Names match resolve_model_size()
// in backend/ai_engine.py — `tiny/small/base_plus/large` route to SAM 2.1,
// `sam3/sam3.1` route to SAM 3 / SAM 3.1 (gated on Hugging Face).
const MODELS = [
  { value: 'tiny',      label: 'SAM 2.1 — Hiera-Tiny (Fastest)',     backend: 'sam2' },
  { value: 'small',     label: 'SAM 2.1 — Hiera-Small (Light)',      backend: 'sam2' },
  { value: 'base_plus', label: 'SAM 2.1 — Hiera-Base+ (Recommended)', backend: 'sam2' },
  { value: 'large',     label: 'SAM 2.1 — Hiera-Large (Best Quality)', backend: 'sam2' },
  { value: 'sam3',      label: 'SAM 3 (Text-native, gated)',          backend: 'sam3' },
  { value: 'sam3.1',    label: 'SAM 3.1 (Latest, gated)',             backend: 'sam3' },
];
const backendOf = (v: string) => MODELS.find(m => m.value === v)?.backend ?? 'sam2';

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

  // ---- Page stage tracker (left sticky stepper) ------------------------------
  const STAGES = [
    { key: 'input',   label: 'INPUT',     hint: 'Upload video' },
    { key: 'config',  label: 'PIPELINE',  hint: 'FPS / threshold / model' },
    { key: 'frames',  label: 'FRAMES',    hint: 'Sharp / Blur / Drop / Dup' },
    { key: 'masking', label: 'MASKING',   hint: 'Reconstruction masks' },
  ] as const;
  const stageRefs = useRef<Array<HTMLElement | null>>([null, null, null, null]);
  const [activeStage, setActiveStage] = useState(0);

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

  // ---- Multi-select bulk classification ----
  const [selectMode, setSelectMode] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState(false);

  // ---- Duplicate detection ----
  type DupCandidate = {
    filename: string;
    full_path: string;
    anchor_filename: string;
    distance: number;
    url?: string | null;
  };
  const [dupOpen, setDupOpen] = useState(false);
  const [dupThreshold, setDupThreshold] = useState(5);
  const [dupCandidates, setDupCandidates] = useState<DupCandidate[]>([]);
  const [dupScanned, setDupScanned] = useState(0);
  const [dupBusy, setDupBusy] = useState(false);
  const [dupCount, setDupCount] = useState(0);   // current files in dup/

  // ---- Source video crop ----
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [cropMode, setCropMode] = useState(false);
  const [cropRect, setCropRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [cropDrag, setCropDrag] = useState<{ x: number; y: number } | null>(null);
  const previewImgRef = useRef<HTMLImageElement>(null);

  // Aspect-ratio lock for the crop selection. "source" matches the input video
  // exactly so SfM intrinsics stay consistent. "free" lets the user draw any
  // rectangle. Numeric ratios are width / height.
  type CropAspect = 'source' | 'free' | '1:1' | '4:3' | '16:9' | '9:16';
  const [cropAspect, setCropAspect] = useState<CropAspect>('source');
  const cropAspectValue = useMemo<number | null>(() => {
    if (cropAspect === 'free') return null;
    if (cropAspect === 'source') return metadata?.width && metadata?.height
      ? metadata.width / metadata.height
      : null;
    if (cropAspect === '1:1')  return 1;
    if (cropAspect === '4:3')  return 4 / 3;
    if (cropAspect === '16:9') return 16 / 9;
    if (cropAspect === '9:16') return 9 / 16;
    return null;
  }, [cropAspect, metadata]);

  // ---- COMBINE mode + mask preview viewer ----
  const [combineMode, setCombineMode] = useState(true);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewItems, setPreviewItems] = useState<Array<{ filename: string; overlay_url: string | null; coverage: number }>>([]);
  const [previewPage, setPreviewPage] = useState(0);
  const PREVIEW_PAGE_SIZE = 12;

  // ---- Server info (backend /health) — drives device-aware model gating ----
  // SAM 3 / 3.1 hard-depend on triton, which has no macOS / MPS / CPU build.
  // We only enable those options when the backend reports device === 'cuda'.
  const [serverInfo, setServerInfo] = useState<{
    device?: string; backend?: string; ai_ready?: boolean; model?: string;
  } | null>(null);
  const isCudaHost = serverInfo?.device === 'cuda';

  const fetchDirectories = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/directories`);
      setDirectories(res.data);
      if (!outputPath && res.data.path) setOutputPath(res.data.path);
    } catch (err) { console.error(err); }
  };

  const fetchServerInfo = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      setServerInfo(res.data);
    } catch (err) { console.error(err); }
  };

  useEffect(() => {
    fetchDirectories();
    fetchServerInfo();
    // Re-poll every 5s so the indicator updates after model swaps / DINO load.
    const t = setInterval(fetchServerInfo, 5000);
    return () => clearInterval(t);
  }, []);

  // Re-fit an existing crop rectangle when the aspect lock changes. Anchored
  // at the rect's top-left, taking the larger of (w, h*aspect) so the user
  // doesn't lose area unnecessarily, then clamped to the source bounds.
  useEffect(() => {
    if (!cropRect || !metadata) return;
    const ax = cropAspectValue;
    if (ax === null || ax <= 0) return;
    const cur = cropRect.w / Math.max(1, cropRect.h);
    if (Math.abs(cur - ax) < 1e-3) return;
    let w = Math.max(cropRect.w, cropRect.h * ax);
    let h = w / ax;
    // Clamp into image starting at (x, y).
    const maxW = metadata.width  - cropRect.x;
    const maxH = metadata.height - cropRect.y;
    if (w > maxW) { w = maxW; h = w / ax; }
    if (h > maxH) { h = maxH; w = h * ax; }
    setCropRect({ x: cropRect.x, y: cropRect.y, w: Math.round(w), h: Math.round(h) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cropAspect]);

  // Track which section the user is currently looking at. Uses
  // IntersectionObserver against a horizontal band centered on the viewport
  // (rootMargin = -35% top, -55% bottom) so the active stage updates only
  // when a section actually crosses the middle of the screen.
  useEffect(() => {
    const els = stageRefs.current.filter((el): el is HTMLElement => !!el);
    if (els.length === 0) return;
    const obs = new IntersectionObserver(
      (entries) => {
        // Pick the entry that is most visible inside the focus band.
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          const idx = stageRefs.current.indexOf(visible[0].target as HTMLElement);
          if (idx >= 0) setActiveStage(idx);
        }
      },
      { rootMargin: '-35% 0px -55% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] }
    );
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
    // Re-bind when [04] mounts (serverOutputDir flips).
  }, [serverOutputDir, results.length]);

  const scrollToStage = (idx: number) => {
    const el = stageRefs.current[idx];
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleSamModelChange = async (newModel: string) => {
    if (backendOf(newModel) === 'sam3' && !isCudaHost) {
      setError(
        `SAM 3 / SAM 3.1 require a CUDA GPU. The backend is running on ` +
        `${serverInfo?.device ?? 'unknown'} — please use SAM 2.1 here, or ` +
        `deploy this server on a CUDA host to use SAM 3.`
      );
      return;
    }
    setIsModelLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/change-model/${newModel}`);
      setSamModel(newModel);
      fetchServerInfo();
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
      // Source preview for the crop picker. Cache-bust per upload.
      setPreviewSrc(`${API_BASE_URL}/preview-frame/${id}?t=${Date.now()}`);
      setCropRect(null);
      setCropMode(false);
    } catch (err: any) { setError(`CONNECTION_ERROR: ${err.message}`); } finally { setLoading(false); }
  };

  const handleProcess = () => {
    if (!fileId) return;
    setProcessing(true); setResults([]); setServerOutputDir(''); setError(null); setCurrentProgress(0); setProgressMsg('CONNECTING...');
    setMaskInfo(null); setPropagateStatus(null); setTextBatchStatus(null);
    setSelectedPaths(new Set()); setDupCandidates([]); setDupCount(0);
    const ws = new WebSocket(`${WS_BASE_URL}/ws/process/${fileId}`);
    ws.onopen = () => ws.send(JSON.stringify({
      fps,
      threshold,
      output_path: outputPath,
      crop: cropRect ? { x: cropRect.x, y: cropRect.y, width: cropRect.w, height: cropRect.h } : null,
    }));
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

    // Masking always targets the sharp/ bucket — blur and drop frames are
    // excluded from reconstruction. The clicked frame may currently be in
    // sharp/ (typical) or it may have just been re-classified; the points
    // were drawn on its pixels regardless, so we still want to anchor at
    // the corresponding sharp/ frame if it's there.
    const framesDir = `${serverOutputDir}/sharp`;
    const fname = editingImage.filename;
    const sharpSiblings = results
      .filter(r => classOf(r) === 'sharp')
      .map(r => r.filename)
      .sort();
    const initFrameIdx = Math.max(0, sharpSiblings.indexOf(fname));
    if (sharpSiblings.length === 0) {
      setError('No SHARP frames to propagate over. Reclassify some frames as SHARP first.');
      return;
    }
    if (!sharpSiblings.includes(fname)) {
      setError(
        `The selected frame ${fname} is not in SHARP. Reclassify it as SHARP (press S) ` +
        `or pick a sharp frame to anchor propagation, then try again.`
      );
      return;
    }

    setPropagating(true);
    setPropagateStatus({ current: 0, total: sharpSiblings.length, message: 'CONNECTING' });

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

  // ------------------------------------------------- Bulk reclassification
  const toggleSelected = (full_path?: string) => {
    if (!full_path) return;
    setSelectedPaths(prev => {
      const next = new Set(prev);
      if (next.has(full_path)) next.delete(full_path); else next.add(full_path);
      return next;
    });
  };

  const handleBulkReclassify = async (target: Classification) => {
    if (selectedPaths.size === 0 || !serverOutputDir) return;
    if (target === 'dup') return;   // dup goes through the duplicate-detection workflow

    setBulkBusy(true);
    const paths = Array.from(selectedPaths);
    // Snapshot originals for revert.
    const originalsByPath = new Map<string, ProcessResult>();
    setResults(prev => {
      for (const r of prev) {
        if (r.full_path && selectedPaths.has(r.full_path)) {
          originalsByPath.set(r.full_path, r);
        }
      }
      return prev;
    });

    // Optimistic predictions.
    const predictedByOldPath = new Map<string, { full_path: string; url: string | null }>();
    setResults(prev => prev.map(r => {
      if (!r.full_path || !selectedPaths.has(r.full_path)) return r;
      const newFull = `${serverOutputDir}/${target}/${r.filename}`;
      const newUrl = r.url ? r.url.replace(/\/(sharp|blur|drop|dup)\//, `/${target}/`) : r.url;
      predictedByOldPath.set(r.full_path, { full_path: newFull, url: newUrl });
      return { ...r, full_path: newFull, url: newUrl, manual_class: target };
    }));
    setSelectedPaths(new Set());

    try {
      const res = await axios.post(`${API_BASE_URL}/reclassify-bulk`, {
        full_paths: paths,
        target,
        output_dir: serverOutputDir,
      });
      const failedSet = new Set((res.data.failed as Array<{ full_path: string }>).map(f => f.full_path));
      // Reconcile predicted entries with server-confirmed values.
      const serverByOldPath = new Map<string, { full_path: string; url: string | null }>();
      let mi = 0;
      for (const p of paths) {
        if (failedSet.has(p)) continue;
        const m = res.data.moved[mi++];
        if (m) serverByOldPath.set(p, { full_path: m.full_path, url: m.url });
      }
      setResults(prev => prev.map(r => {
        // Find by predicted full_path. We tagged manual_class=target during
        // the optimistic pass, so undo for failed ones.
        const oldPath = [...predictedByOldPath.entries()].find(([, p]) => p.full_path === r.full_path)?.[0];
        if (!oldPath) return r;
        if (failedSet.has(oldPath)) {
          // Revert this one.
          return originalsByPath.get(oldPath) ?? r;
        }
        const srv = serverByOldPath.get(oldPath);
        if (!srv) return r;
        if (srv.full_path === r.full_path && (srv.url ?? null) === (r.url ?? null)) return r;
        return { ...r, full_path: srv.full_path, url: srv.url ?? r.url };
      }));
      if (res.data.failed.length > 0) {
        setError(`BULK_PARTIAL: ${res.data.failed.length} of ${paths.length} failed.`);
      }
    } catch (err: any) {
      // Full revert.
      setResults(prev => prev.map(r => {
        const oldPath = [...predictedByOldPath.entries()].find(([, p]) => p.full_path === r.full_path)?.[0];
        if (!oldPath) return r;
        return originalsByPath.get(oldPath) ?? r;
      }));
      setError(`BULK_RECLASSIFY_ERROR: ${err.message}`);
    } finally {
      setBulkBusy(false);
    }
  };

  // ------------------------------------------------- Duplicate detection
  const handleOpenDupDialog = () => { setDupOpen(true); };

  const handleScanDuplicates = async () => {
    if (!serverOutputDir) return;
    setDupBusy(true);
    setDupCandidates([]);
    try {
      const res = await axios.post(`${API_BASE_URL}/detect-duplicates`, {
        scene_dir: serverOutputDir,
        threshold: dupThreshold,
        frames_subdir: 'sharp',
      });
      setDupCandidates(res.data.candidates || []);
      setDupScanned(res.data.scanned || 0);
    } catch (err: any) {
      setError(`DUP_SCAN_ERROR: ${err.message}`);
    } finally {
      setDupBusy(false);
    }
  };

  const handleApplyDuplicates = async () => {
    if (!serverOutputDir || dupCandidates.length === 0) return;
    setDupBusy(true);
    try {
      const filenames = dupCandidates.map(c => c.filename);
      const res = await axios.post(`${API_BASE_URL}/apply-duplicates`, {
        scene_dir: serverOutputDir,
        filenames,
      });
      const movedFilenames = new Set((res.data.moved as Array<{ filename: string; full_path: string }>).map(m => m.filename));
      const movedFullPathByName = new Map(
        (res.data.moved as Array<{ filename: string; full_path: string }>).map(m => [m.filename, m.full_path])
      );
      setResults(prev => prev.map(r => {
        if (!movedFilenames.has(r.filename)) return r;
        const fp = movedFullPathByName.get(r.filename) ?? r.full_path;
        // Tag with manual_class='dup' via type extension (no UI logic relies on
        // it directly — the file location dictates which tab it shows up in).
        return { ...r, full_path: fp, manual_class: 'dup' as Classification };
      }));
      setDupCount(prev => prev + res.data.moved_count);
      setDupCandidates([]);
    } catch (err: any) {
      setError(`DUP_APPLY_ERROR: ${err.message}`);
    } finally {
      setDupBusy(false);
    }
  };

  const handleRestoreDuplicates = async () => {
    if (!serverOutputDir) return;
    setDupBusy(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/restore-duplicates`, {
        scene_dir: serverOutputDir,
      });
      const restoredFilenames = new Set((res.data.restored as Array<{ filename: string; full_path: string }>).map(m => m.filename));
      const restoredFullPathByName = new Map(
        (res.data.restored as Array<{ filename: string; full_path: string }>).map(m => [m.filename, m.full_path])
      );
      setResults(prev => prev.map(r => {
        if (!restoredFilenames.has(r.filename)) return r;
        const fp = restoredFullPathByName.get(r.filename) ?? r.full_path;
        // Restoring drops the manual_class so the auto sharp/blur logic kicks
        // back in based on the score vs threshold.
        const next = { ...r, full_path: fp, url: fp ? r.url : r.url };
        delete (next as any).manual_class;
        return next;
      }));
      setDupCount(0);
    } catch (err: any) {
      setError(`DUP_RESTORE_ERROR: ${err.message}`);
    } finally {
      setDupBusy(false);
    }
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

  // Three-way classification view. `manual_class` overrides the auto label.
  // Auto-labelled frames are sharp (score >= threshold) or blur (score < threshold).
  // Drop is only ever set manually by the user.
  const classOf = useCallback(
    (r: ProcessResult): Classification => {
      if (r.manual_class) return r.manual_class;
      return r.score < threshold ? 'blur' : 'sharp';
    },
    [threshold]
  );
  const analyticsData = useMemo(() => {
    if (results.length === 0) return null;
    const sharp = results.filter(r => classOf(r) === 'sharp');
    const blur  = results.filter(r => classOf(r) === 'blur');
    const drop  = results.filter(r => classOf(r) === 'drop');
    const dup   = results.filter(r => classOf(r) === 'dup');
    const avgAll = results.reduce((acc, curr) => acc + curr.score, 0) / results.length;
    const avgSharp = sharp.length > 0 ? sharp.reduce((acc, curr) => acc + curr.score, 0) / sharp.length : 0;
    return {
      sharpCount: sharp.length,
      blurCount:  blur.length,
      dropCount:  drop.length,
      dupCount:   dup.length,
      avgAll: parseFloat(avgAll.toFixed(2)),
      avgSharp: parseFloat(avgSharp.toFixed(2)),
    };
  }, [results, classOf]);

  const filteredResults = useMemo(() => {
    if (tabValue === 0) return results;
    if (tabValue === 1) return results.filter(r => classOf(r) === 'sharp');
    if (tabValue === 2) return results.filter(r => classOf(r) === 'blur');
    if (tabValue === 3) return results.filter(r => classOf(r) === 'drop');
    return results.filter(r => classOf(r) === 'dup');
  }, [results, classOf, tabValue]);

  // Position of the current editing image inside `filteredResults`, by filename.
  // If the frame was just reclassified out of the active tab, fall back to the
  // nearest filename — this keeps the counter and arrow-key navigation working
  // instead of resetting to 1/N.
  const positionInFiltered = useMemo(() => {
    if (!editingImage || filteredResults.length === 0) return -1;
    const exact = filteredResults.findIndex(r => r.filename === editingImage.filename);
    if (exact >= 0) return exact;
    const target = editingImage.filename;
    const after = filteredResults.findIndex(r => r.filename >= target);
    return after < 0 ? filteredResults.length - 1 : after;
  }, [editingImage, filteredResults]);

  const navigateImage = useCallback((direction: 1 | -1) => {
    if (filteredResults.length === 0) return;
    const cur = positionInFiltered;
    if (cur < 0) return;
    const nextIdx = (cur + direction + filteredResults.length) % filteredResults.length;
    const next = filteredResults[nextIdx];
    setEditingImage(next);
    setPoints([]);
    setMaskUrl(null);
  }, [positionInFiltered, filteredResults]);

  // Auto-advance: when the editing image is no longer in the active tab
  // (because the user just classified it out of that bucket), jump to the
  // nearest surviving frame so they keep moving through the queue.
  useEffect(() => {
    if (!isMaskEditorOpen || !editingImage || filteredResults.length === 0) return;
    if (filteredResults.some(r => r.filename === editingImage.filename)) return;
    const target = editingImage.filename;
    const next = filteredResults.find(r => r.filename >= target) ?? filteredResults[filteredResults.length - 1];
    if (next && next.filename !== editingImage.filename) {
      setEditingImage(next);
      setPoints([]);
      setMaskUrl(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredResults, editingImage, isMaskEditorOpen]);

  const handleReclassify = useCallback(async (target: Classification) => {
    if (!editingImage?.full_path || !serverOutputDir) return;
    if (target === 'dup') return;   // dup is set via duplicate-detection workflow

    // Optimistic update: predict the new path/URL and update local state
    // immediately so keyboard shortcuts feel instant. Reconcile with the
    // backend response afterwards; revert on failure.
    const original = editingImage;
    const originalFullPath = editingImage.full_path;
    const predictedFullPath = `${serverOutputDir}/${target}/${editingImage.filename}`;
    const predictedUrl = editingImage.url
      ? editingImage.url.replace(/\/(sharp|blur|drop|dup)\//, `/${target}/`)
      : editingImage.url;
    const optimistic: ProcessResult = {
      ...editingImage,
      full_path: predictedFullPath,
      url: predictedUrl,
      manual_class: target,
    };
    setResults(prev => prev.map(r => (r.full_path === originalFullPath ? optimistic : r)));
    setEditingImage(optimistic);

    try {
      const res = await axios.post(`${API_BASE_URL}/reclassify`, {
        full_path: originalFullPath,
        target,
        output_dir: serverOutputDir,
      });
      // Reconcile only if the backend disagreed with our prediction.
      const serverFull: string = res.data.full_path;
      const serverUrl: string | null = res.data.url ?? null;
      if (serverFull !== predictedFullPath || (serverUrl ?? null) !== (predictedUrl ?? null)) {
        const reconciled: ProcessResult = {
          ...optimistic,
          full_path: serverFull,
          url: serverUrl ?? optimistic.url,
        };
        setResults(prev => prev.map(r => (r.full_path === predictedFullPath ? reconciled : r)));
        setEditingImage(prev => (prev && prev.full_path === predictedFullPath ? reconciled : prev));
      }
    } catch (err: any) {
      // Revert on failure — UI snaps back to the original.
      setResults(prev => prev.map(r => (r.full_path === predictedFullPath ? original : r)));
      setEditingImage(prev => (prev && prev.full_path === predictedFullPath ? original : prev));
      setError(`RECLASSIFY_ERROR: ${err.message}`);
    }
  }, [editingImage, serverOutputDir]);

  useEffect(() => {
    if (!isMaskEditorOpen) return;
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
      if (propagating || isSegmenting) return;
      // Modifier keys are passed through (e.g. Cmd+S for browser save).
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (e.key === 'ArrowRight') { e.preventDefault(); navigateImage(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); navigateImage(-1); }
      else if (k === 's') { e.preventDefault(); handleReclassify('sharp'); }
      else if (k === 'b') { e.preventDefault(); handleReclassify('blur'); }
      else if (k === 'd') { e.preventDefault(); handleReclassify('drop'); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isMaskEditorOpen, navigateImage, propagating, isSegmenting, handleReclassify]);

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
              VIDEO_TO_IMAGE_AI_PIPELINE <Chip label="V4.4.0" size="small" sx={{ ml: 1, height: 20, fontSize: '10px' }} />
            </Typography>
            {(metadata || serverInfo) && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="caption" color={serverInfo?.ai_ready ? "success.main" : "error.main"} sx={{ fontWeight: 'bold' }}>
                  {(() => {
                    const tag = (serverInfo?.backend === 'sam3') ? 'SAM3' : 'SAM2';
                    return serverInfo?.ai_ready ? `● ${tag}_ONLINE` : `○ ${tag}_OFFLINE`;
                  })()}
                </Typography>
                <Typography variant="caption" color="success.main">● PIPELINE_READY</Typography>
              </Box>
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl">
          {/* ---------- Left sticky stage indicator ---------- */}
          {/* Hidden on small screens (<lg) where vertical real estate matters
              more than the navigator. Click a step to jump there. */}
          <Box
            sx={{
              display: { xs: 'none', lg: 'flex' },
              position: 'fixed',
              left: 16,
              top: '50%',
              transform: 'translateY(-50%)',
              zIndex: 1100,
              flexDirection: 'column',
              gap: 1.5,
              px: 1.5,
              py: 2,
              bgcolor: 'rgba(253, 252, 251, 0.92)',
              border: '1px solid #E6E1D6',
              borderRadius: 1,
              boxShadow: 1,
              backdropFilter: 'blur(4px)',
            }}
          >
            <Typography variant="caption" sx={{ fontSize: '9px', fontWeight: 'bold', color: 'text.secondary', letterSpacing: 0.5 }}>
              STAGE
            </Typography>
            {STAGES.map((s, i) => {
              const active = activeStage === i;
              const reachable = i < 3 || !!serverOutputDir; // [04] only after pipeline ran
              return (
                <Box
                  key={s.key}
                  onClick={() => reachable && scrollToStage(i)}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    cursor: reachable ? 'pointer' : 'not-allowed',
                    opacity: reachable ? 1 : 0.4,
                    px: 0.5,
                    py: 0.5,
                    borderRadius: 0.5,
                    transition: 'background-color 120ms',
                    '&:hover': reachable ? { bgcolor: '#F4F1EA' } : {},
                  }}
                >
                  <Box
                    sx={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      flexShrink: 0,
                      bgcolor: active ? '#4A4238' : 'transparent',
                      border: '2px solid',
                      borderColor: active ? '#4A4238' : '#D4CBB3',
                      boxShadow: active ? '0 0 0 3px rgba(74, 66, 56, 0.15)' : 'none',
                      transition: 'all 120ms',
                    }}
                  />
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography
                      variant="caption"
                      sx={{
                        fontSize: '10px',
                        fontWeight: active ? 'bold' : 'normal',
                        color: active ? 'text.primary' : 'text.secondary',
                        lineHeight: 1.1,
                      }}
                    >
                      [{String(i + 1).padStart(2, '0')}] {s.label}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ fontSize: '8px', color: 'text.secondary', lineHeight: 1.1, opacity: 0.8 }}
                    >
                      {s.hint}
                    </Typography>
                  </Box>
                </Box>
              );
            })}
          </Box>

          <Grid container spacing={3}>
            {/* Left Column: Input & Config */}
            <Grid item xs={12} md={4}>
              <Paper ref={(el: HTMLElement | null) => { stageRefs.current[0] = el; }} sx={{ p: 3, mb: 3, bgcolor: '#FDFCFB', scrollMarginTop: 80 }}>
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

                {/* Crop region picker — drawn on the source video's first frame.
                    Coordinates are stored in source-pixel space and forwarded
                    to ws_process as `crop: {x,y,width,height}`. */}
                {file && previewSrc && metadata && (
                  <Box sx={{ mt: 2, p: 1.5, border: '1px solid #E6E1D6', bgcolor: '#FFFFFF' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1, flexWrap: 'wrap', gap: 0.5 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                        CROP (optional)
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap' }}>
                        <FormControl size="small" sx={{ minWidth: 90 }}>
                          <Select
                            value={cropAspect}
                            onChange={(e) => setCropAspect(e.target.value as CropAspect)}
                            sx={{ fontSize: '10px', height: 24, '& .MuiSelect-select': { py: 0.25 } }}
                            title="Aspect ratio lock"
                          >
                            <MenuItem value="source" sx={{ fontSize: '11px' }}>
                              Source ({metadata ? `${metadata.width}:${metadata.height}` : '?'})
                            </MenuItem>
                            <MenuItem value="free"  sx={{ fontSize: '11px' }}>Free</MenuItem>
                            <MenuItem value="1:1"   sx={{ fontSize: '11px' }}>1:1</MenuItem>
                            <MenuItem value="4:3"   sx={{ fontSize: '11px' }}>4:3</MenuItem>
                            <MenuItem value="16:9"  sx={{ fontSize: '11px' }}>16:9</MenuItem>
                            <MenuItem value="9:16"  sx={{ fontSize: '11px' }}>9:16</MenuItem>
                          </Select>
                        </FormControl>
                        <Button
                          size="small"
                          variant={cropMode ? 'contained' : 'outlined'}
                          onClick={() => setCropMode(m => !m)}
                          sx={{ fontSize: '9px', py: 0, minWidth: 60 }}
                        >
                          {cropMode ? 'DRAWING' : 'DRAW'}
                        </Button>
                        {cropRect && (
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => { setCropRect(null); setCropMode(false); }}
                            sx={{ fontSize: '9px', py: 0, minWidth: 50 }}
                          >
                            RESET
                          </Button>
                        )}
                      </Box>
                    </Box>
                    <Box
                      sx={{
                        position: 'relative',
                        width: '100%',
                        userSelect: 'none',
                        cursor: cropMode ? 'crosshair' : 'default',
                        border: '1px solid #E6E1D6',
                      }}
                      onMouseDown={(e) => {
                        if (!cropMode || !previewImgRef.current || !metadata) return;
                        const rect = previewImgRef.current.getBoundingClientRect();
                        const sx = (e.clientX - rect.left) / rect.width  * metadata.width;
                        const sy = (e.clientY - rect.top)  / rect.height * metadata.height;
                        setCropDrag({ x: sx, y: sy });
                        setCropRect({ x: sx, y: sy, w: 0, h: 0 });
                      }}
                      onMouseMove={(e) => {
                        if (!cropMode || !cropDrag || !previewImgRef.current || !metadata) return;
                        const rect = previewImgRef.current.getBoundingClientRect();
                        const sx = (e.clientX - rect.left) / rect.width  * metadata.width;
                        const sy = (e.clientY - rect.top)  / rect.height * metadata.height;
                        const ax = cropAspectValue;

                        // Raw drag deltas (signed → direction; abs → magnitude).
                        const dx = sx - cropDrag.x;
                        const dy = sy - cropDrag.y;
                        const sgnX = dx < 0 ? -1 : 1;
                        const sgnY = dy < 0 ? -1 : 1;
                        let absW = Math.abs(dx);
                        let absH = Math.abs(dy);

                        // 1) Enforce aspect ratio relative to the drag anchor.
                        if (ax !== null && ax > 0) {
                          if (absW / Math.max(1, absH) > ax) absW = absH * ax;
                          else                                absH = absW / ax;
                        }

                        // 2) Clamp into source bounds. Anchor stays at cropDrag,
                        //    so shrinking in one direction shrinks both axes when
                        //    a ratio is locked — apply max-extent first.
                        const maxW = sgnX < 0 ? cropDrag.x : metadata.width  - cropDrag.x;
                        const maxH = sgnY < 0 ? cropDrag.y : metadata.height - cropDrag.y;
                        if (ax !== null && ax > 0) {
                          // Pick the limiting axis and recompute the other.
                          if (absW > maxW) { absW = maxW; absH = absW / ax; }
                          if (absH > maxH) { absH = maxH; absW = absH * ax; }
                        } else {
                          absW = Math.min(absW, maxW);
                          absH = Math.min(absH, maxH);
                        }

                        // 3) Position (top-left) once magnitudes are settled.
                        const x = sgnX < 0 ? cropDrag.x - absW : cropDrag.x;
                        const y = sgnY < 0 ? cropDrag.y - absH : cropDrag.y;
                        setCropRect({
                          x: Math.round(x), y: Math.round(y),
                          w: Math.round(absW), h: Math.round(absH),
                        });
                      }}
                      onMouseUp={() => {
                        setCropDrag(null);
                        setCropMode(false);
                        // Tiny drags are treated as accidental — discard.
                        if (cropRect && (cropRect.w < 8 || cropRect.h < 8)) setCropRect(null);
                      }}
                      onMouseLeave={() => { setCropDrag(null); }}
                    >
                      <img
                        ref={previewImgRef}
                        src={previewSrc}
                        alt="source preview"
                        style={{ width: '100%', display: 'block', pointerEvents: 'none' }}
                      />
                      {cropRect && cropRect.w > 0 && cropRect.h > 0 && (
                        <Box
                          sx={{
                            position: 'absolute',
                            border: '2px solid #4A4238',
                            bgcolor: 'rgba(74, 66, 56, 0.15)',
                            pointerEvents: 'none',
                            left: `${(cropRect.x / metadata.width) * 100}%`,
                            top:  `${(cropRect.y / metadata.height) * 100}%`,
                            width:  `${(cropRect.w / metadata.width) * 100}%`,
                            height: `${(cropRect.h / metadata.height) * 100}%`,
                          }}
                        />
                      )}
                    </Box>
                    <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '10px', color: 'text.secondary' }}>
                      {cropRect && cropRect.w > 0 && cropRect.h > 0 ? (
                        <>
                          Selected: {cropRect.w}×{cropRect.h} @ ({cropRect.x}, {cropRect.y}) px
                          {' · '}
                          {cropAspect === 'free'
                            ? `aspect ${(cropRect.w / cropRect.h).toFixed(3)} (free)`
                            : `aspect locked: ${cropAspect}${cropAspect === 'source' && metadata ? ` (${metadata.width}:${metadata.height})` : ''}`}
                        </>
                      ) : cropMode
                        ? `Drag on the preview to define the crop region. Aspect: ${cropAspect}${cropAspect === 'source' && metadata ? ` (${metadata.width}:${metadata.height})` : ''}.`
                        : 'No crop — full frame will be extracted.'}
                    </Typography>
                  </Box>
                )}
              </Paper>

              <Paper ref={(el: HTMLElement | null) => { stageRefs.current[1] = el; }} sx={{ p: 3, bgcolor: '#FDFCFB', scrollMarginTop: 80 }}>
                <Typography variant="subtitle2" gutterBottom color="textSecondary">[02] PIPELINE_CONFIGURATION</Typography>
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, gap: 1 }}>
                    <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center' }}>
                      <BrainIcon sx={{ fontSize: 16, mr: 0.5 }} />
                      ENGINE: {backendOf(samModel) === 'sam3' ? 'SAM_3_MODEL' : 'SAM_2.1_MODEL'}
                    </Typography>
                    {serverInfo?.device && (
                      <Chip
                        label={`device: ${serverInfo.device.toUpperCase()}`}
                        size="small"
                        color={isCudaHost ? 'success' : 'default'}
                        sx={{ height: 18, fontSize: '9px' }}
                      />
                    )}
                  </Box>
                  <FormControl fullWidth size="small">
                    <Select value={samModel} onChange={(e) => handleSamModelChange(e.target.value)} disabled={isModelLoading}>
                      {MODELS.map(m => {
                        const requiresCuda = m.backend === 'sam3';
                        const blocked = requiresCuda && !isCudaHost;
                        return (
                          <MenuItem
                            key={m.value}
                            value={m.value}
                            disabled={blocked}
                            sx={{ opacity: blocked ? 0.5 : 1 }}
                          >
                            {m.label}
                            {blocked && (
                              <Chip
                                label="CUDA only"
                                size="small"
                                color="warning"
                                sx={{ ml: 1, height: 16, fontSize: '9px' }}
                              />
                            )}
                          </MenuItem>
                        );
                      })}
                    </Select>
                  </FormControl>
                  {!isCudaHost && (
                    <Typography
                      variant="caption"
                      sx={{ display: 'block', mt: 0.5, fontSize: '10px', color: 'warning.dark' }}
                    >
                      ⚠ SAM 3 / SAM 3.1 require CUDA. They are disabled on{' '}
                      {(serverInfo?.device ?? 'this').toUpperCase()} hosts because their{' '}
                      <code>triton</code> dependency has no macOS / MPS / CPU build.
                      Use SAM 2.1 here, or deploy this server on a CUDA host.
                    </Typography>
                  )}
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
                <Button fullWidth variant="contained" size="large" sx={{ py: 1.5, bgcolor: '#4A4238' }} onClick={handleProcess} disabled={!fileId || processing || isModelLoading}>
                  {processing ? <CircularProgress size={24} color="inherit" /> : 'START_PIPELINE'}
                </Button>
                {/* APPLY & SYNC FOLDERS is rendered under the classified frames
                    grid (section [03]) so the user can re-apply the threshold
                    after reviewing the actual sharp/blur split. */}
              </Paper>
            </Grid>

            {/* Main Content Area */}
            <Grid item xs={12} md={8}>
              <Paper ref={(el: HTMLElement | null) => { stageRefs.current[2] = el; }} sx={{ p: 3, minHeight: '85vh', borderLeft: '4px solid #4A4238', scrollMarginTop: 80 }}>
                {/* Toolbar: select mode + duplicate-detection controls.
                    Only meaningful after the pipeline has produced results. */}
                {results.length > 0 && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                    <Button
                      size="small"
                      variant={selectMode ? 'contained' : 'outlined'}
                      onClick={() => {
                        setSelectMode(s => !s);
                        if (selectMode) setSelectedPaths(new Set());
                      }}
                      sx={{ minWidth: 100 }}
                    >
                      {selectMode ? 'EXIT SELECT' : 'SELECT MODE'}
                    </Button>
                    {selectMode && (
                      <Typography variant="caption" sx={{ fontSize: '11px', fontWeight: 'bold', color: 'text.secondary' }}>
                        {selectedPaths.size} selected
                      </Typography>
                    )}
                    <Box sx={{ flexGrow: 1 }} />
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={handleOpenDupDialog}
                      disabled={dupBusy}
                    >
                      DETECT DUPLICATES
                    </Button>
                    {(dupCount > 0 || (analyticsData?.dupCount ?? 0) > 0) && (
                      <Button
                        size="small"
                        variant="outlined"
                        color="warning"
                        onClick={handleRestoreDuplicates}
                        disabled={dupBusy}
                      >
                        RESTORE DUP ({analyticsData?.dupCount ?? dupCount})
                      </Button>
                    )}
                  </Box>
                )}

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
                    <Tab label={`DROP (${analyticsData?.dropCount || 0})`} sx={{ fontSize: '11px' }} />
                    <Tab label={`DUP (${analyticsData?.dupCount || 0})`} sx={{ fontSize: '11px' }} />
                  </Tabs>
                </Box>

                <Grid container spacing={1.5}>
                  {results.length > 0 ? (
                    filteredResults.map((res, index) => {
                      const cls = classOf(res);
                      const dimmed = cls !== 'sharp';
                      const chipColor: 'success' | 'error' | 'default' | 'warning' | 'info' =
                        cls === 'sharp' ? 'success'
                          : cls === 'blur' ? 'error'
                            : cls === 'dup' ? 'info'
                              : 'default';
                      const baseLetter = cls === 'sharp' ? 'S'
                        : cls === 'blur' ? 'B'
                          : cls === 'dup' ? 'U'
                            : 'D';
                      const chipLabel = res.manual_class ? `${baseLetter}*` : baseLetter;
                      const selected = !!res.full_path && selectedPaths.has(res.full_path);
                      const opacity = dimmed
                        ? (cls === 'drop' ? 0.4 : cls === 'dup' ? 0.5 : 0.6)
                        : 1;
                      return (
                        <Grid item xs={6} sm={4} md={3} lg={2.4} key={index}>
                          <Card
                            variant="outlined"
                            onClick={() => {
                              if (selectMode) toggleSelected(res.full_path);
                              else handleImageClick(res);
                            }}
                            sx={{
                              cursor: 'pointer',
                              opacity,
                              position: 'relative',
                              borderColor: selected ? 'primary.main' : undefined,
                              borderWidth: selected ? 2 : 1,
                              boxShadow: selected ? '0 0 0 2px rgba(74, 66, 56, 0.25)' : undefined,
                              '&:hover': { borderColor: 'primary.main' },
                            }}
                          >
                            {selectMode && (
                              <Box sx={{ position: 'absolute', top: 4, left: 4, zIndex: 2, bgcolor: 'rgba(255,255,255,0.85)', borderRadius: '50%' }}>
                                <Checkbox size="small" checked={selected} sx={{ p: 0.25 }} onClick={(e) => e.stopPropagation()} onChange={() => toggleSelected(res.full_path)} />
                              </Box>
                            )}
                            <CardMedia component="img" height="100" image={`${API_BASE_URL}${res.url}`} sx={{ filter: dimmed ? 'grayscale(80%)' : 'none' }} />
                            <CardContent sx={{ p: 0.8 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" sx={{ fontSize: '9px', fontWeight: 'bold' }}>S:{res.score}</Typography>
                                <Chip label={chipLabel} size="small" color={chipColor} sx={{ height: 12, fontSize: '7px', minWidth: 16 }} />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })
                  ) : !processing && <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', opacity: 0.2 }}><ImageIcon sx={{ fontSize: 64 }} /></Box>}
                </Grid>

                {/* Footer: re-apply threshold AFTER reviewing the sharp/blur
                    split. Hidden until the pipeline has produced results. */}
                {results.length > 0 && (
                  <Box sx={{ mt: 3, pt: 2, borderTop: '1px dashed #E6E1D6', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ flexGrow: 1, fontSize: '11px' }}>
                      Adjusted the threshold? Re-apply it to physically move frames between sharp/ and blur/.
                      <br />
                      <Box component="span" sx={{ opacity: 0.7 }}>
                        Frames already moved to <code>drop/</code> are preserved.
                      </Box>
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={syncing ? <CircularProgress size={16}/> : <SyncIcon />}
                      onClick={handleSyncFolders}
                      disabled={syncing || processing}
                      sx={{ minWidth: 220 }}
                    >
                      {syncing ? 'SYNCING...' : 'APPLY & SYNC FOLDERS'}
                    </Button>
                  </Box>
                )}
              </Paper>

              {/* ======================================================== */}
              {/* [04] RECONSTRUCTION MASKING (SAM 2.1 + Grounded-SAM 2 /   */}
              {/*       SAM 3 / SAM 3.1 native text)                        */}
              {/* ======================================================== */}
              {serverOutputDir && (
                <Paper ref={(el: HTMLElement | null) => { stageRefs.current[3] = el; }} sx={{ p: 3, mt: 3, bgcolor: '#FDFCFB', scrollMarginTop: 80 }}>
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
                          <TextIcon sx={{ fontSize: 14, mr: 0.5 }} /> TEXT_PROMPT_MASK ({backendOf(samModel) === 'sam3' ? 'SAM 3 native' : 'Grounded-SAM 2'})
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

      {/* Sticky bulk-action bar (visible only in select mode with selection) */}
      {selectMode && selectedPaths.size > 0 && (
        <Box
          sx={{
            position: 'fixed',
            bottom: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1200,
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
            px: 2.5,
            py: 1.5,
            bgcolor: '#FDFCFB',
            border: '1px solid #4A4238',
            borderRadius: 1,
            boxShadow: 4,
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
            {selectedPaths.size} selected →
          </Typography>
          <Button size="small" variant="contained" color="success" disabled={bulkBusy} onClick={() => handleBulkReclassify('sharp')}>SHARP</Button>
          <Button size="small" variant="contained" color="error"   disabled={bulkBusy} onClick={() => handleBulkReclassify('blur')}>BLUR</Button>
          <Button size="small" variant="contained" color="inherit" disabled={bulkBusy} onClick={() => handleBulkReclassify('drop')}>DROP</Button>
          <Box sx={{ width: 1, height: 24, bgcolor: '#E6E1D6', mx: 0.5 }} />
          <Button size="small" variant="text" disabled={bulkBusy} onClick={() => setSelectedPaths(new Set())}>Clear</Button>
          {bulkBusy && <CircularProgress size={16} sx={{ ml: 0.5 }} />}
        </Box>
      )}

      {/* Duplicate detection dialog */}
      <Dialog open={dupOpen} onClose={() => !dupBusy && setDupOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontSize: '14px', fontWeight: 'bold' }}>
          DETECT_DUPLICATES — perceptual-hash near-duplicate finder
        </DialogTitle>
        <DialogContent>
          <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mb: 2, fontSize: '11px' }}>
            Scans <code>&lt;scene&gt;/sharp/</code> with an 8×8 dHash. Frames within
            the chosen Hamming distance of the previous kept anchor are flagged.
            Apply moves them to <code>&lt;scene&gt;/dup/</code> (reversible —
            click <b>RESTORE DUP</b> on the [03] toolbar to bring them back).
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Typography variant="caption" sx={{ minWidth: 90, fontWeight: 'bold' }}>
              Threshold: {dupThreshold}
            </Typography>
            <Slider
              size="small"
              value={dupThreshold}
              min={0}
              max={20}
              step={1}
              marks={[{ value: 0, label: '0 (identical)' }, { value: 5, label: '5' }, { value: 10, label: '10' }, { value: 20, label: '20' }]}
              onChange={(_, v) => setDupThreshold(v as number)}
              sx={{ flexGrow: 1 }}
            />
          </Box>
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Button variant="outlined" size="small" disabled={dupBusy || !serverOutputDir} onClick={handleScanDuplicates}>
              {dupBusy ? <CircularProgress size={16} /> : 'SCAN'}
            </Button>
            <Typography variant="caption" sx={{ alignSelf: 'center', color: 'text.secondary', fontSize: '11px' }}>
              {dupCandidates.length > 0 ? `${dupCandidates.length} candidates / ${dupScanned} scanned` : 'No scan yet'}
            </Typography>
          </Box>
          {dupCandidates.length > 0 && (
            <Box sx={{ maxHeight: 360, overflowY: 'auto', border: '1px solid #E6E1D6' }}>
              <Grid container>
                {dupCandidates.map((c) => (
                  <Grid item xs={6} sm={4} md={3} key={c.full_path}>
                    <Card variant="outlined" sx={{ m: 0.5 }}>
                      {c.url && <CardMedia component="img" height="80" image={`${API_BASE_URL}${c.url}`} />}
                      <CardContent sx={{ p: 0.75 }}>
                        <Typography variant="caption" sx={{ fontSize: '9px', display: 'block', fontWeight: 'bold' }}>
                          {c.filename}
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: '8px', color: 'text.secondary', display: 'block' }}>
                          ≈ {c.anchor_filename} (d={c.distance})
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDupOpen(false)} disabled={dupBusy}>CLOSE</Button>
          <Button
            variant="contained"
            color="primary"
            disabled={dupBusy || dupCandidates.length === 0}
            onClick={async () => { await handleApplyDuplicates(); }}
          >
            APPLY → MOVE TO DUP
          </Button>
        </DialogActions>
      </Dialog>

      {/* Masking Editor Dialog */}
      <Dialog open={isMaskEditorOpen} onClose={() => !propagating && setIsMaskEditorOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          AI MASKING EDITOR ({backendOf(samModel) === 'sam3' ? (samModel === 'sam3.1' ? 'SAM 3.1' : 'SAM 3') : 'SAM 2.1'})
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
              {editingImage && filteredResults.length > 0 && positionInFiltered >= 0
                ? `${positionInFiltered + 1} / ${filteredResults.length}`
                : '— / —'}
            </Typography>
            <IconButton
              size="small"
              onClick={() => navigateImage(1)}
              disabled={propagating || isSegmenting || filteredResults.length < 2}
            >
              <ChevronRightIcon />
            </IconButton>
            {editingImage && serverOutputDir && (() => {
              const cls = classOf(editingImage);
              return (
                <ButtonGroup size="small" variant="outlined" sx={{ ml: 2 }}>
                  <Button
                    variant={cls === 'sharp' ? 'contained' : 'outlined'}
                    color="success"
                    onClick={() => handleReclassify('sharp')}
                    disabled={propagating || isSegmenting}
                    title="Mark as SHARP (shortcut: S)"
                  >
                    SHARP <Box component="span" sx={{ ml: 0.5, opacity: 0.6, fontSize: '9px' }}>(S)</Box>
                  </Button>
                  <Button
                    variant={cls === 'blur' ? 'contained' : 'outlined'}
                    color="error"
                    onClick={() => handleReclassify('blur')}
                    disabled={propagating || isSegmenting}
                    title="Mark as BLUR (shortcut: B)"
                  >
                    BLUR <Box component="span" sx={{ ml: 0.5, opacity: 0.6, fontSize: '9px' }}>(B)</Box>
                  </Button>
                  <Button
                    variant={cls === 'drop' ? 'contained' : 'outlined'}
                    color="inherit"
                    onClick={() => handleReclassify('drop')}
                    disabled={propagating || isSegmenting}
                    title="Mark as DROP — excluded from masking and reconstruction (shortcut: D)"
                  >
                    DROP <Box component="span" sx={{ ml: 0.5, opacity: 0.6, fontSize: '9px' }}>(D)</Box>
                  </Button>
                </ButtonGroup>
              );
            })()}
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
