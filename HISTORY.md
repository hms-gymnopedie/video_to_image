# Project Update History

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
