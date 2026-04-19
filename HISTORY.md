# Project Update History

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
