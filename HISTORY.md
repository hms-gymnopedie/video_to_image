# Project Update History

## [2026-04-16] Update V1.5.3
- **Fixed:** Recurring WebSocket coroutine error. Replaced polling-based monitoring with a more robust concurrent task approach (`asyncio.create_task` for monitor + `await to_thread` for processing).
- **Improved:** Stability of frame extraction and blur analysis pipeline.
- **Updated:** Versioning to V1.5.3.
