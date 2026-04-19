# Project Update History

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
