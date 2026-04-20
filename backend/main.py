import os
import shutil
import uuid
import traceback
import asyncio
from typing import List, Optional

import cv2
import numpy as np
import ffmpeg
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_engine import (
    SAMEngine,
    generate_mask_previews,
    resolve_model_size,
    save_binary_mask,
    save_overlay,
)

VERSION = "4.0.0"

# --- Directories ---
UPLOAD_DIR, OUTPUT_DIR, MASK_DIR = "uploads", "output", "masks"
PREVIEW_DIR = os.path.join(MASK_DIR, "preview")
for d in [UPLOAD_DIR, OUTPUT_DIR, MASK_DIR, PREVIEW_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SAMEngine()


def _cleanup_upload(file_id: str) -> None:
    """Remove the uploaded source video for a given file_id."""
    try:
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(file_id):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, f))
                except OSError as e:
                    print(f"[CLEANUP] upload remove failed {f}: {e}")
    except FileNotFoundError:
        pass


def _cleanup_segment_previews() -> None:
    """Wipe the per-click mask editor preview debris in backend/masks/preview/."""
    try:
        if not os.path.isdir(PREVIEW_DIR):
            return
        removed = 0
        for f in os.listdir(PREVIEW_DIR):
            p = os.path.join(PREVIEW_DIR, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    removed += 1
                except OSError as e:
                    print(f"[CLEANUP] preview remove failed {f}: {e}")
        if removed:
            print(f"[CLEANUP] removed {removed} editor preview files")
    except FileNotFoundError:
        pass


@app.on_event("startup")
async def startup():
    # Load SAM 2.1 default in background so the server comes up fast.
    asyncio.create_task(asyncio.to_thread(engine.load_sam2, "base_plus"))
    # Grounding DINO on-demand only (first /segment-text call).


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION, **engine.status()}


# --------------------------------------------------------------- Model control
@app.post("/change-model/{model_type}")
async def change_model(model_type: str):
    try:
        size = resolve_model_size(model_type)
    except ValueError:
        raise HTTPException(400, detail=f"Unknown model: {model_type}")
    ok = await asyncio.to_thread(engine.load_sam2, size)
    if ok:
        return {"status": "success", "model": size}
    raise HTTPException(
        500,
        detail=(
            f"Failed to load SAM 2.1 ({size}). "
            "Check server logs. If sam2 is missing, install with: "
            'pip install "git+https://github.com/facebookresearch/sam2.git"'
        ),
    )


# --------------------------------------------------------------- Dashboard I/O
@app.get("/directories")
async def list_dirs():
    def get_struct(p):
        d = {"name": os.path.basename(p), "path": p, "children": []}
        try:
            with os.scandir(p) as it:
                for e in it:
                    if e.is_dir() and not e.name.startswith("."):
                        d["children"].append(get_struct(e.path))
        except Exception:
            pass
        return d

    return get_struct(os.path.abspath(OUTPUT_DIR))


@app.post("/create-directory")
async def create_dir(data: dict):
    p = os.path.join(data["parent_path"], data["new_name"])
    os.makedirs(p, exist_ok=True)
    return {"status": "success", "path": p}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}{os.path.splitext(file.filename)[1]}")
    with open(path, "wb") as b:
        shutil.copyfileobj(file.file, b)
    return {"file_id": file_id}


@app.get("/metadata/{file_id}")
async def meta(file_id: str):
    p = next(
        (os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id)),
        None,
    )
    if not p:
        raise HTTPException(404)
    probe = ffmpeg.probe(p)
    s = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = eval(s.get("avg_frame_rate", "30/1"))
    return {
        "duration": float(probe["format"].get("duration", 0)),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "avg_frame_rate": f"{round(fps, 2)} FPS",
        "ai_ready": engine.is_ready,
    }


# -------------------------------------------------------- Frame extraction + blur
@app.websocket("/ws/process/{file_id}")
async def ws_process(ws: WebSocket, file_id: str):
    await ws.accept()
    try:
        data = await ws.receive_json()
        fps = float(data.get("fps", 1.0))
        threshold = float(data.get("threshold", 100.0))
        out_raw = data.get("output_path", "")
        # TreeView selection can arrive as a list (["path"]) — coerce to string.
        if isinstance(out_raw, list):
            out = out_raw[0] if out_raw else ""
        else:
            out = out_raw
        out = (out or "").strip()

        vid = next(
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.startswith(file_id)
        )
        base = os.path.abspath(out) if out else os.path.join(os.path.abspath(OUTPUT_DIR), file_id)
        os.makedirs(base, exist_ok=True)

        sd = os.path.join(base, "sharp")
        bd = os.path.join(base, "blur")
        for d in [sd, bd]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        tmp = os.path.join(base, f"tmp_{uuid.uuid4().hex}")
        os.makedirs(tmp, exist_ok=True)

        await ws.send_json({"type": "status", "message": "1/2 EXTRACTING"})

        def run():
            (
                ffmpeg.input(vid)
                .filter("fps", fps=fps)
                .output(os.path.join(tmp, "f_%04d.jpg"), qscale=2)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

        await asyncio.to_thread(run)

        await ws.send_json({"type": "status", "message": "2/2 ANALYZING"})
        files = sorted([f for f in os.listdir(tmp) if f.endswith(".jpg")])
        res = []
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(tmp, f))
            var = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            is_b = var < threshold
            dst_dir = bd if is_b else sd
            shutil.move(os.path.join(tmp, f), os.path.join(dst_dir, f))
            p_url = f"/images/{os.path.relpath(os.path.join(dst_dir, f), os.path.abspath(OUTPUT_DIR))}"
            res.append(
                {
                    "filename": f,
                    "url": p_url,
                    "score": round(var, 2),
                    "is_blurry": is_b,
                    "full_path": os.path.join(dst_dir, f),
                }
            )
            if i % 10 == 0:
                await ws.send_json({"type": "progress", "current": i + 1, "total": len(files)})
        shutil.rmtree(tmp)
        # Source video is no longer needed — frames are on disk.
        _cleanup_upload(file_id)
        await ws.send_json({"type": "complete", "results": res, "output_dir": base})
    except Exception as e:
        traceback.print_exc()
        await ws.send_json({"type": "error", "message": str(e)})


# ------------------------------------------------------- Phase 1: point masking
@app.post("/segment")
async def segment(data: dict):
    """Single-frame point-prompt segmentation (backward-compatible endpoint)."""
    if not engine.is_ready:
        raise HTTPException(503, "AI Engine Offline")
    try:
        mask, score = engine.segment_image_points(
            data["image_path"], data["points"], data.get("labels")
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

    preview_name = f"m_{uuid.uuid4().hex}.png"
    preview_path = os.path.join(PREVIEW_DIR, preview_name)
    save_overlay(data["image_path"], mask, preview_path)
    return {
        "mask_url": f"/masks/preview/{preview_name}",
        "score": score,
    }


# ---------------------------------------------------- Phase 2: video propagation
class PropagateRequest(BaseModel):
    frames_dir: str            # e.g. /abs/path/to/output/<scene>/sharp
    init_frame_idx: int = 0
    points: List[List[float]]  # [[x, y], ...]
    labels: Optional[List[int]] = None  # 1=foreground, 0=background
    scene_dir: Optional[str] = None     # defaults to parent of frames_dir
    invert_mask: bool = True   # for reconstruction: detected region -> black
    write_overlay: bool = False
    combine: bool = True       # AND with existing mask if present
    skip_empty: bool = True    # skip frames where SAM2 finds nothing


@app.websocket("/ws/segment-video")
async def ws_segment_video(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_json()
        req = PropagateRequest(**raw)
        if not engine.is_ready:
            await ws.send_json({"type": "error", "message": "AI Engine Offline"})
            return

        frames_dir = os.path.abspath(req.frames_dir)
        if not os.path.isdir(frames_dir):
            await ws.send_json({"type": "error", "message": f"frames_dir not found: {frames_dir}"})
            return

        scene_dir = os.path.abspath(req.scene_dir) if req.scene_dir else os.path.dirname(frames_dir)
        mask_dir = os.path.join(scene_dir, "masks")
        overlay_dir = os.path.join(scene_dir, "mask_preview")
        os.makedirs(mask_dir, exist_ok=True)
        if req.write_overlay:
            os.makedirs(overlay_dir, exist_ok=True)

        frames_sorted = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        await ws.send_json({"type": "status", "message": "PROPAGATING", "total": len(frames_sorted)})

        def _run(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
            try:
                written = 0
                for frame_idx, mask in engine.propagate_video(
                    frames_dir, req.init_frame_idx, req.points, req.labels
                ):
                    fname = frames_sorted[frame_idx]
                    mask_path = os.path.join(mask_dir, f"{os.path.splitext(fname)[0]}.png")
                    wrote = save_binary_mask(
                        mask,
                        mask_path,
                        invert=req.invert_mask,
                        combine=req.combine,
                        skip_empty=req.skip_empty,
                    )
                    if wrote:
                        written += 1
                        if req.write_overlay:
                            save_overlay(
                                os.path.join(frames_dir, fname),
                                mask,
                                os.path.join(overlay_dir, f"{os.path.splitext(fname)[0]}.png"),
                            )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("progress", frame_idx, fname, wrote)), loop
                    )
                asyncio.run_coroutine_threadsafe(queue.put(("done", written, None)), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc), None)), loop)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        asyncio.create_task(asyncio.to_thread(_run, queue, loop))

        processed = 0
        while True:
            kind, a, b, *rest = await queue.get()
            if kind == "progress":
                processed += 1
                await ws.send_json(
                    {
                        "type": "progress",
                        "current": processed,
                        "total": len(frames_sorted),
                        "frame_idx": a,
                        "filename": b,
                        "wrote": bool(rest[0]) if rest else True,
                    }
                )
            elif kind == "done":
                # Per-click editor previews are no longer needed once the
                # full-sequence masks have been written.
                _cleanup_segment_previews()
                await ws.send_json(
                    {
                        "type": "complete",
                        "mask_dir": mask_dir,
                        "frame_count": a,
                        "processed": processed,
                        "invert_mask": req.invert_mask,
                    }
                )
                return
            else:
                await ws.send_json({"type": "error", "message": a})
                return
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# --------------------------------------------------- Phase 3: sync + preset masks
class ReconstructionPresetRequest(BaseModel):
    scene_dir: str
    preset: str = "3dgs"  # 3dgs | 2dgs | colmap


@app.post("/reconstruction-mask-info")
async def recon_info(req: ReconstructionPresetRequest):
    """Return where binary masks should live for the chosen preset."""
    scene = os.path.abspath(req.scene_dir)
    mask_dir = os.path.join(scene, "masks")
    return {
        "scene_dir": scene,
        "mask_dir": mask_dir,
        "convention": {
            "3dgs": "masks/<frame_basename>.png (255=use, 0=exclude)",
            "2dgs": "masks/<frame_basename>.png (255=use, 0=exclude)",
            "colmap": "masks/<frame_filename>.png (rename if your COLMAP needs <img>.<ext>.png)",
        }.get(req.preset, "masks/<frame_basename>.png"),
        "exists": os.path.isdir(mask_dir),
        "count": len(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else 0,
    }


# ---------------------------------------------------- Phase 4: text-prompt mask
class TextSegmentRequest(BaseModel):
    image_path: str
    queries: List[str]        # ["person", "sky", "car"]
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    write_overlay: bool = True


@app.post("/segment-text")
async def segment_text(req: TextSegmentRequest):
    if not engine.is_ready:
        raise HTTPException(503, "AI Engine Offline")
    if not engine.is_text_ready:
        ok = await asyncio.to_thread(engine.load_grounding_dino)
        if not ok:
            raise HTTPException(503, "Grounding DINO unavailable")

    try:
        mask, det = await asyncio.to_thread(
            engine.segment_text,
            req.image_path,
            req.queries,
            req.box_threshold,
            req.text_threshold,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

    result = {
        "labels": det["labels"],
        "scores": [float(s) for s in det["scores"]],
        "num_detections": int(len(det["labels"])),
    }

    if req.write_overlay:
        preview_name = f"t_{uuid.uuid4().hex}.png"
        preview_path = os.path.join(PREVIEW_DIR, preview_name)
        save_overlay(req.image_path, mask, preview_path)
        result["mask_url"] = f"/masks/preview/{preview_name}"
    return result


class TextBatchRequest(BaseModel):
    frames_dir: str
    queries: List[str]
    scene_dir: Optional[str] = None
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    invert_mask: bool = True
    combine: bool = True
    skip_empty: bool = True


@app.websocket("/ws/segment-text-batch")
async def ws_segment_text_batch(ws: WebSocket):
    """Phase 4 batch: run Grounded-SAM2 on every frame in a directory."""
    await ws.accept()
    try:
        raw = await ws.receive_json()
        req = TextBatchRequest(**raw)

        if not engine.is_ready:
            await ws.send_json({"type": "error", "message": "AI Engine Offline"})
            return
        if not engine.is_text_ready:
            ok = await asyncio.to_thread(engine.load_grounding_dino)
            if not ok:
                await ws.send_json({"type": "error", "message": "Grounding DINO unavailable"})
                return

        frames_dir = os.path.abspath(req.frames_dir)
        scene_dir = os.path.abspath(req.scene_dir) if req.scene_dir else os.path.dirname(frames_dir)
        mask_dir = os.path.join(scene_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        frames = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        total = len(frames)
        await ws.send_json({"type": "status", "message": "TEXT_BATCH", "total": total})

        written = 0
        for i, fname in enumerate(frames):
            img_path = os.path.join(frames_dir, fname)
            mask, _det = await asyncio.to_thread(
                engine.segment_text,
                img_path,
                req.queries,
                req.box_threshold,
                req.text_threshold,
            )
            out_path = os.path.join(mask_dir, f"{os.path.splitext(fname)[0]}.png")
            wrote = save_binary_mask(
                mask,
                out_path,
                invert=req.invert_mask,
                combine=req.combine,
                skip_empty=req.skip_empty,
            )
            if wrote:
                written += 1
            if i % 5 == 0 or i == total - 1:
                await ws.send_json(
                    {
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "filename": fname,
                        "written": written,
                    }
                )

        _cleanup_segment_previews()
        await ws.send_json(
            {
                "type": "complete",
                "mask_dir": mask_dir,
                "frame_count": written,
                "processed": total,
            }
        )
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------- Mask preview (Phase 1+4 verify)
class PreviewRequest(BaseModel):
    scene_dir: str
    frames_subdir: str = "sharp"


@app.post("/generate-mask-previews")
async def gen_mask_previews(req: PreviewRequest):
    """Build red-tint overlay PNGs for every frame that has a mask.
    Returns URLs via /images when the scene is under OUTPUT_DIR, else falls
    back to /overlay-file?path=... so scenes outside OUTPUT_DIR still render.
    """
    scene = os.path.abspath(req.scene_dir)
    items = await asyncio.to_thread(generate_mask_previews, scene, req.frames_subdir)

    output_abs = os.path.abspath(OUTPUT_DIR)
    enriched = []
    for it in items:
        abs_overlay = os.path.join(scene, it["overlay_rel"])
        url = None
        try:
            rel = os.path.relpath(abs_overlay, output_abs)
            if not rel.startswith(".."):
                url = f"/images/{rel}"
        except ValueError:
            pass
        if url is None and os.path.isfile(abs_overlay):
            # Fall back to direct file serving (scene outside OUTPUT_DIR).
            url = f"/overlay-file?path={abs_overlay}"
        enriched.append({**it, "overlay_url": url, "overlay_abs": abs_overlay})

    print(
        f"[PREVIEW] scene={scene} frames={req.frames_subdir} "
        f"items={len(enriched)} mask_dir_exists="
        f"{os.path.isdir(os.path.join(scene, 'masks'))}"
    )
    return {
        "scene_dir": scene,
        "count": len(enriched),
        "items": enriched,
    }


@app.get("/overlay-file")
async def overlay_file(path: str):
    """Serve a single overlay/mask file by absolute path.
    Only allows files that end in .jpg/.jpeg/.png and live under a
    `mask_preview` or `masks` directory, to keep this off the general FS.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise HTTPException(404, "not found")
    if not abs_path.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "unsupported extension")
    parent = os.path.basename(os.path.dirname(abs_path))
    if parent not in ("mask_preview", "masks"):
        raise HTTPException(403, "only mask_preview/masks files are servable")
    return FileResponse(abs_path)


# ------------------------------------------------------- Manual reclassification
class ReclassifyRequest(BaseModel):
    full_path: str
    target: str        # "sharp" or "blur"
    output_dir: str    # scene root containing sharp/ and blur/


@app.post("/reclassify")
async def reclassify(req: ReclassifyRequest):
    if req.target not in ("sharp", "blur"):
        raise HTTPException(400, "target must be 'sharp' or 'blur'")

    src = os.path.abspath(req.full_path)
    scene = os.path.abspath(req.output_dir)
    if not os.path.isfile(src):
        raise HTTPException(404, f"file not found: {src}")

    target_dir = os.path.join(scene, req.target)
    os.makedirs(target_dir, exist_ok=True)
    fname = os.path.basename(src)
    dst = os.path.join(target_dir, fname)

    if src == dst:
        # Already in target folder; treat as idempotent success.
        pass
    else:
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)

    output_abs = os.path.abspath(OUTPUT_DIR)
    try:
        rel = os.path.relpath(dst, output_abs)
        url = f"/images/{rel}" if not rel.startswith("..") else None
    except ValueError:
        url = None

    return {
        "status": "success",
        "full_path": dst,
        "url": url,
        "target": req.target,
    }


# --------------------------------------------------------- Sync & static mounts
@app.post("/sync-folders")
async def sync(data: dict):
    sd = os.path.join(data["output_dir"], "sharp")
    bd = os.path.join(data["output_dir"], "blur")
    for r in data["results"]:
        is_b = r["score"] < data["threshold"]
        fn = r["filename"]
        src = os.path.join(bd if os.path.exists(os.path.join(bd, fn)) else sd, fn)
        dst = os.path.join(bd if is_b else sd, fn)
        if src != dst:
            shutil.move(src, dst)
    return {"status": "success"}


app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
