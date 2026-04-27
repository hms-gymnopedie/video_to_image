# Project Update History

## [2026-04-27] Update V4.4.0 — Bulk select, duplicate detection, source crop
- **Added:** Multi-frame selection mode in `[03] BUFFER & ANALYTICS`. Toggle `SELECT MODE` to show checkboxes on every thumbnail; clicking a card now toggles selection instead of opening the editor. Selected cards render with a thicker primary-colored border.
- **Added:** Sticky bulk action bar (centered at the bottom of the viewport when at least one frame is selected): `→ SHARP / → BLUR / → DROP / Clear`. Calls the new backend endpoint and updates results state in place.
- **Added:** Backend `POST /reclassify-bulk` — accepts `{full_paths[], target, output_dir}` and returns per-path success / failure with the new full_path / URL of each moved file. Idempotent (already-in-target paths short-circuit).
- **Added:** Perceptual-hash duplicate finder. `compute_dhash()` + `detect_duplicate_frames()` in `ai_engine.py` (8×8 difference hash, Hamming-distance scan with anchor walk). New endpoints:
  - `POST /detect-duplicates` — scan `<scene>/sharp/` with a user-chosen Hamming threshold; returns each candidate's anchor + distance.
  - `POST /apply-duplicates` — move named candidates from `sharp/` → `dup/`.
  - `POST /restore-duplicates` — move `dup/` (or a named subset) back to `sharp/`. Fully reversible.
- **Added:** `<scene>/dup/` bucket as a fourth, fully-reversible classification alongside `sharp/`, `blur/`, `drop/`. Created on every START_PIPELINE and **preserved across re-extractions**.
- **Added:** `[03]` toolbar buttons: `SELECT MODE`, `DETECT DUPLICATES`, and (when `dup/` is non-empty) `RESTORE DUP (n)`.
- **Added:** Duplicate-detection dialog with a Hamming-distance slider (0–20, marks at 0 / 5 / 10 / 20), live-scan button, candidate preview grid showing each frame's anchor and distance, and an APPLY button.
- **Added:** Fifth tab `DUP (n)` in `[03]`. `analyticsData.dupCount` exposed; chip on dup'd frames renders blue `U / U*`.
- **Changed:** `/sync-folders` now also skips frames in `dup/` (in addition to `drop/`) so the threshold slider cannot undo auto-detected duplicates.
- **Added:** Source-video region-of-interest cropping in `[01] INPUT_SOURCE`.
  - Backend `GET /preview-frame/{file_id}` returns the first frame of the uploaded video as JPEG (cached in `backend/masks/preview/`).
  - `[01]` shows the preview image with a `DRAW / RESET` toggle. Drag a rectangle on the preview to define the crop in source-pixel space.
  - **Aspect-ratio lock.** Default is `Source` (the input video's exact ratio,
    so SfM camera intrinsics stay consistent). Dropdown also offers `Free`,
    `1:1`, `4:3`, `16:9`, `9:16`. The drag handler enforces the locked aspect
    while clamping into source bounds; switching the lock re-fits an existing
    crop rectangle anchored at its top-left.
  - `WS /ws/process/{file_id}` accepts an optional `crop: {x, y, width, height}` payload and applies an FFmpeg `crop=W:H:X:Y` filter after the FPS filter. Omitted or zero-sized crop = full-frame extraction (unchanged behavior).
  - Crop selection survives between START_PIPELINE runs of the same upload until the user explicitly resets or uploads a new video.
- **Updated:** Versioning to V4.4.0.

## [2026-04-27] Update V4.3.0 — UX: stage navigator + SYNC relocation
- **Added:** Left sticky stage indicator (`[01] INPUT / [02] PIPELINE / [03] FRAMES / [04] MASKING`). Tracks the active section via `IntersectionObserver` (rootMargin centered on the viewport) and updates as the user scrolls. Click a step to `scrollIntoView` the corresponding `<Paper>`. Auto-hidden below the `lg` breakpoint.
- **Added:** Each section `<Paper>` now carries a `scrollMarginTop: 80` so that smooth scroll lands below the sticky AppBar.
- **Moved:** `APPLY & SYNC FOLDERS` button from `[02] PIPELINE_CONFIGURATION` to a footer inside `[03] BUFFER & ANALYTICS`, below the frame grid. The button now appears only after the user has reviewed the actual sharp/blur/drop split, with a one-line caption clarifying that `drop/` is preserved across syncs.
- **Updated:** Versioning to V4.3.0.

## [2026-04-27] Update V4.2.0 — Three-way classification + sharp-only masking
- **Added:** `drop/` bucket as a third manual classification alongside `sharp/` and `blur/`. `<scene>/drop/` is created on every START_PIPELINE and **preserved across re-extractions** so user-curated discards survive an FPS / threshold change.
- **Added:** Keyboard shortcuts in the AI MASKING EDITOR: **`S`** → SHARP, **`B`** → BLUR, **`D`** → DROP. Existing `←` / `→` navigation unchanged. Modifier keys (Cmd/Ctrl/Alt) and INPUT/TEXTAREA focus pass through.
- **Added:** `DROP` button in the editor footer ButtonGroup; thumbnail chips now render `S / S* / B / B* / D / D*` with appropriate colors (success / error / default).
- **Added:** Fourth tab `DROP` in `[03] BUFFER & ANALYTICS`. `analyticsData` returns a `dropCount`.
- **Changed:** `/reclassify` accepts `target ∈ {sharp, blur, drop}` (was `{sharp, blur}`). Constant `RECLASSIFY_TARGETS` exposes the canonical list.
- **Changed:** `/sync-folders` skips frames already in `drop/` so the threshold slider cannot undo manual discards.
- **Added:** Backend safety guards — `WS /ws/segment-video` and `WS /ws/segment-text-batch` reject any `frames_dir` whose basename is not `sharp`. The frontend `handlePropagate` now always submits `<scene>/sharp` regardless of which bucket the clicked frame currently lives in. Triple protection ensures BLUR / DROP frames are never masked.
- **Updated:** Versioning to V4.2.0.

## [2026-04-27] Update V4.1.0 — SAM 3 / SAM 3.1 backend (CUDA-gated)
- **Added:** `Sam3Engine` path inside `ai_engine.py`. The existing `SAMEngine` class becomes a facade with a `backend` attribute (`"sam2"` | `"sam3"`); the three inference methods (`segment_image_points`, `propagate_video`, `segment_text`) dispatch to `_sam2_*` / `_sam3_*` implementations internally. SAM 2.1 path is byte-identical.
- **Added:** Native text prompts via SAM 3 (Promptable Concept Segmentation). On the SAM 3 backend, Grounding DINO is bypassed entirely — `load_grounding_dino()` becomes a no-op and `segment_text` calls `Sam3Processor.set_text_prompt(...)` directly. The transformers `box_threshold` / `threshold` fallback is no longer reached.
- **Added:** `load_model(name)` dispatcher that picks the right backend by canonical model name (`tiny / small / base_plus / large` → SAM 2.1, `sam3 / sam3.1` → SAM 3). Legacy aliases `vit_b/l/h` continue to map to SAM 2.1 sizes.
- **Added:** Output extractors `_extract_sam3_masks`, `_extract_sam3_scores`, `_unpack_sam3_video_item` to absorb the dict / dataclass / tuple return shapes the upstream `sam3` package has shipped under.
- **Added:** Frontend model selector now lists `SAM 3 (Text-native, gated)` and `SAM 3.1 (Latest, gated)` alongside the four SAM 2.1 sizes. UI labels (`ENGINE: SAM_2.1_MODEL` / `SAM_3_MODEL`, `TEXT_PROMPT_MASK (Grounded-SAM 2 / SAM 3 native)`, `AI MASKING EDITOR (SAM 2.1 / SAM 3 / SAM 3.1)`) reflect the active backend.
- **Added:** `/health` polling every 5s on the frontend → `serverInfo.{device, backend, ai_ready, model}`. The top status indicator now correctly shows `SAM2_ONLINE` / `SAM3_ONLINE` based on the running backend (was reading a non-existent field on the video metadata response).
- **Added:** Device-aware CUDA gating. SAM 3 / SAM 3.1 require `triton`, which has no macOS / MPS / CPU build. When `serverInfo.device !== 'cuda'`, the SAM 3 / 3.1 dropdown items are disabled with a yellow `CUDA only` chip, and a warning caption appears under the selector. `handleSamModelChange` short-circuits with a clear error if invoked on a non-CUDA host.
- **Improved:** `/change-model/{model_type}` returns the active `backend` in its response and surfaces a HuggingFace gating-aware error message (request access on `huggingface.co/facebook/sam3{,.1}` then `hf auth login`) when the SAM 3 load fails.
- **Improved:** SAM 3 import error reporting — distinguishes between "sam3 package missing" and "sam3 transitive dep missing" (e.g. `einops`, `triton`) and prints a targeted install hint.
- **Documented:** `requirements.txt` now lists the optional `git+https://github.com/facebookresearch/sam3.git` install line and notes that checkpoints are gated.
- **Note:** SAM 3 cannot run on Apple Silicon today due to the upstream `triton` hard-dependency (see `facebookresearch/sam3#154`, `#164`). The frontend gate prevents users from hitting this at runtime.
- **Updated:** Versioning to V4.1.0.

## [2026-04-20] Update V4.0.0 — SAM 2.1 + Grounded-SAM 2 Migration
- **Replaced:** SAM v1 (`segment-anything`) with SAM 2.1 (`sam2`) for superior temporal consistency on video inputs. Image + video predictors now loaded via Hugging Face Hub (`facebook/sam2.1-hiera-*`).
- **Added:** `ai_engine.py` module encapsulating SAM 2.1, Grounding DINO, and mask I/O helpers.
- **Added:** Phase 2 — video propagation endpoint `WS /ws/segment-video`. One click on any frame produces consistent masks for the entire sequence.
- **Added:** Phase 3 — reconstruction-ready binary mask output (`<scene>/masks/<frame>.png`, 255=keep / 0=exclude) plus `/reconstruction-mask-info` helper for 3DGS/2DGS/COLMAP conventions.
- **Added:** Phase 4 — Grounded-SAM 2 text-prompt segmentation. Endpoints `POST /segment-text` (single frame) and `WS /ws/segment-text-batch` (full sequence) accept queries like `["person", "sky", "car"]`.
- **Improved:** Apple Silicon (M-series) MPS auto-detection; CUDA remains preferred when available.
- **Preserved:** Legacy `vit_b/vit_l/vit_h` names remapped to `tiny/base_plus/large` so the existing frontend dropdown keeps working.
- **Updated:** Versioning to V4.0.0. Install note: `pip install "git+https://github.com/facebookresearch/sam2.git"` is required in addition to `requirements.txt`.

## [2026-04-16] Update V3.4.0
- **Improved:** SAM model download resilience. Implemented browser-imitation headers for `curl` and `requests` to bypass 403 Forbidden errors on Meta's public servers.
- **Improved:** Automated model verification. Model files are now strictly checked for minimum size (100MB) to prevent corrupt/HTML downloads.
- **Added:** Detailed manual download instructions in logs for critical failure scenarios.
- **Updated:** Versioning to V3.4.0.

## [2026-04-16] Update V3.3.9
- **Fixed:** Persistent model download 403 Forbidden errors by switching to stable Hugging Face mirrors.
- **Improved:** Download reliability using system `curl -L` as the primary transfer method on macOS.
- **Improved:** Automated download verification ensuring model files are fully transferred before loading.
- **Updated:** Versioning to V3.3.9.

## [2026-04-16] Update V3.3.8
- **Fixed:** `invalid load key, '<'` error by improving the model download logic. Tiny error/HTML files are now detected and deleted automatically.
- **Improved:** Added User-Agent headers and download verification to ensure model weights are downloaded correctly from Meta's servers.
- **Updated:** Versioning to V3.3.8.

## [2026-04-16] Update V3.3.7
- **Fixed:** PyTorch 2.6 weights loading security error. Implemented a monkeypatch for `torch.load` to ensure compatibility with Segment Anything Model (SAM) registry.
- **Fixed:** Port already in use error by terminating background processes on 8080.
- **Improved:** Backend startup resilience.
- **Updated:** Versioning to V3.3.7.

## [2026-04-16] Update V3.5.5
- **Fixed:** Critical issue where automated model downloads were being blocked (403 Forbidden).
- **Improved:** Switched to a manual-first/browser-download strategy for AI model weights to ensure 100% reliability.
- **Improved:** Server now provides clear terminal instructions for manual setup if the model is missing.
- **Improved:** Robust model verification (size-based) before loading.
- **Updated:** Versioning to V3.5.5.

## [2026-04-16] Update V3.5.0
- **Added:** Startup-resilient backend. The server now starts immediately even if the SAM model is missing, allowing core features (extraction, metadata) to function.
- **Improved:** Reliable model mirror using Hugging Face.
- **Added:** Real-time AI Engine status indicator (Online/Offline) in the dashboard.
- **Improved:** Simplified backend code for better maintainability and error handling.
- **Updated:** Versioning to V3.5.0.

## [2026-04-16] Update V3.3.5
- **Fixed:** Persistent environment and dependency sync issues by providing a definitive clean-build script.
- **Improved:** Backend version identification for easier debugging.
- **Updated:** Versioning to V3.3.5.

## [2026-04-16] Update V3.3.4
- **Improved:** Backend environment setup guidance. Provided a comprehensive script to reset `venv` and install all AI/Backend dependencies.
- **Updated:** Versioning to V3.3.4.

## [2026-04-16] Update V3.3.3
- **Restored:** Complete video metadata view (Resolution, FPS, Length, Aspect Ratio) during summary.
- **Fixed:** Critical 404 errors by consolidating backend routing and ensuring all endpoints are registered.
- **Fixed:** Dependency issues by updating `requirements.txt` with `torch`, `requests`, and `segment-anything`.
- **Improved:** Robustness of the AI Masking pipeline.
- **Updated:** Versioning to V3.3.3.

## [2026-04-16] Update V3.3.2
- **Fixed:** `ModuleNotFoundError: No module named 'requests'` by adding the library to the virtual environment and updating `requirements.txt`.
- **Improved:** Standardized project dependencies for robust setup on new environments.
- **Updated:** Versioning to V3.3.2.

## [2026-04-16] Update V3.3.1
- **Fixed:** 404 error when creating a new folder by restoring the missing `/create-directory` endpoint in the backend.
- **Improved:** Code consistency and endpoint organization in `main.py`.
- **Updated:** Versioning to V3.3.1.

## [2026-04-16] Update V3.3.0
- **Restored:** Full video metadata summary (Resolution, FPS, Length, Aspect Ratio) after upload.
- **Restored:** Interactive folder management and application presets.
- **Improved:** Integrated AI Masking workflow with all legacy preprocessing tools.
- **Updated:** Versioning to V3.3.0.

## [2026-04-16] Update V3.2.0
- **Restored:** Missing UI features from previous AI update, including Folder Browser controls (Refresh/New Folder).
- **Restored:** Application Presets (3DGS, 2DGS, COLMAP) for quick pipeline configuration.
- **Restored:** Real-time Threshold Guide under the slider to assist in quality filtering.
- **Improved:** Integrated AI Model Selection with restored pipeline settings for a complete dashboard experience.
- **Updated:** Versioning to V3.2.0.

## [2026-04-16] Update V3.1.0
- **Added:** SAM Multi-Model Support. Users can now choose between ViT-B (Fast), ViT-L (Large), and ViT-H (Highest Accuracy) models.
- **Added:** Dynamic Model Loading. Backend automatically downloads and switches models upon user request.
- **Improved:** Interactive Masking Editor with point-based segmentation.
- **Updated:** Versioning to V3.1.0.

## [2026-04-16] Update V3.0.0
- **Added:** AI-powered Segment Anything Model (SAM) integration. Users can now mask objects in extracted frames interactively.
- **Added:** Interactive Masking Editor. Click on objects to automatically generate high-quality masks.
- **Added:** Automatic model management. Backend automatically downloads SAM weights (vit_b) on first run.
- **Improved:** Sidebar and Analytics UI for AI-focused workflow.
- **Updated:** Major version bump to V3.0.0.

## [2026-04-16] Update V2.0.2
- **Fixed:** Process error (`'list' object has no attribute 'strip'`) caused by unexpected data type of `output_path` in WebSocket requests.
- **Improved:** Robust input handling for output directory paths.
- **Updated:** Versioning to V2.0.2.
