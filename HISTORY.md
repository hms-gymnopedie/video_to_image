# Project Update History

## [2026-04-16] Update V1.5.2
- **Fixed:** WebSocket processing error (`AttributeError: 'coroutine' object has no attribute 'done'`) by properly wrapping the background thread in an asyncio Task.
- **Improved:** Robustness of real-time monitoring loop for FFmpeg frame extraction.
- **Updated:** Versioning to V1.5.2.
