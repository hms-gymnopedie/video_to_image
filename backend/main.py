import os
import shutil
import uuid
import subprocess
import cv2
import json
import traceback
import time
import math
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import ffmpeg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path.startswith("/ws"): return await call_next(request)
    print(f"DEBUG: {request.method} {request.url}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"DEBUG: SERVER_ERROR: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e), "trace": traceback.format_exc()}, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/health")
async def health_check(): return {"status": "ok", "version": "1.9.0"}

UPLOAD_DIR = "uploads"
DEFAULT_OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="images")

@app.get("/directories")
async def list_directories():
    def get_dir_structure(root_path):
        d = {'name': os.path.basename(root_path), 'path': root_path, 'children': []}
        try:
            with os.scandir(root_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.name.startswith('.'):
                        d['children'].append(get_dir_structure(entry.path))
        except Exception: pass
        return d
    return get_dir_structure(os.path.abspath(DEFAULT_OUTPUT_DIR))

class CreateDirRequest(BaseModel):
    parent_path: str
    new_name: str

@app.post("/create-directory")
async def create_directory(data: CreateDirRequest):
    new_path = os.path.join(data.parent_path, data.new_name)
    try:
        os.makedirs(new_path, exist_ok=True)
        return {"status": "success", "path": new_path}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    with open(video_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    return {"file_id": file_id, "filename": file.filename}

@app.get("/metadata/{file_id}")
async def get_metadata(file_id: str):
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id):
            video_path = os.path.join(UPLOAD_DIR, f)
            break
    if not video_path: raise HTTPException(status_code=404, detail="Video not found")
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        format_info = probe.get('format', {})
        fps_raw = video_stream.get('avg_frame_rate', '0/0')
        fps = eval(fps_raw) if '/' in fps_raw else float(fps_raw)
        return {
            "duration": float(format_info.get('duration', 0)),
            "width": int(video_stream.get('width', 0)),
            "height": int(video_stream.get('height', 0)),
            "avg_frame_rate": f"{round(fps, 2)} FPS",
        }
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/process/{file_id}")
async def websocket_process(websocket: WebSocket, file_id: str):
    await websocket.accept()
    try:
        raw_data = await websocket.receive_text()
        data = json.loads(raw_data)
        
        fps = float(data.get('fps', 1.0))
        scale = data.get('scale', "-1:-1")
        qscale = int(data.get('qscale', 2))
        naming_rule = data.get('naming_rule', "frame_%04d.jpg")
        threshold = float(data.get('threshold', 100.0))
        output_path = data.get('output_path', "")

        video_path = None
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(file_id):
                video_path = os.path.join(UPLOAD_DIR, f)
                break
        if not video_path:
            await websocket.send_json({"type": "error", "message": "Video not found"})
            return

        base_output_dir = os.path.abspath(output_path.strip()) if output_path and output_path.strip() else os.path.join(os.path.abspath(DEFAULT_OUTPUT_DIR), file_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Subdirectories
        sharp_dir = os.path.join(base_output_dir, "sharp")
        blur_dir = os.path.join(base_output_dir, "blur")
        
        # Clean and create
        for d in [sharp_dir, blur_dir]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # Extraction phase
        temp_extract_dir = os.path.join(base_output_dir, f"temp_{uuid.uuid4().hex}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        await websocket.send_json({"type": "status", "message": "1/3 EXTRACTING_FRAMES"})
        
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps=fps)
        if scale != "-1:-1":
            try:
                w, h = scale.split(':')
                stream = ffmpeg.filter(stream, 'scale', w=w, h=h)
            except: pass
        
        output_pattern = os.path.join(temp_extract_dir, naming_rule)
        
        def run_ffmpeg_sync():
            try:
                (ffmpeg.output(stream, output_pattern, qscale=qscale).overwrite_output().run(capture_stdout=True, capture_stderr=True))
                return True
            except ffmpeg.Error as e:
                print(f"FFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
                return False

        async def monitor_progress():
            while True:
                try:
                    files = [f for f in os.listdir(temp_extract_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                    await websocket.send_json({"type": "progress", "current": len(files), "message": f"EXTRACTING: {len(files)} generated"})
                except: break
                await asyncio.sleep(1.0)

        monitor_task = asyncio.create_task(monitor_progress())
        success = await asyncio.to_thread(run_ffmpeg_sync)
        monitor_task.cancel()

        if not success:
            await websocket.send_json({"type": "error", "message": "FFmpeg failed"})
            return

        # ANALYSIS AND MOVE PHASE
        await websocket.send_json({"type": "status", "message": "2/3 ANALYZING_AND_SORTING"})
        
        # Give OS a moment to finish file writing
        time.sleep(1.0)
        
        results = []
        raw_files = sorted([f for f in os.listdir(temp_extract_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        is_internal = base_output_dir.startswith(os.path.abspath(DEFAULT_OUTPUT_DIR))
        
        all_scores = []
        sharp_scores = []
        
        for i, filename in enumerate(raw_files):
            src_path = os.path.join(temp_extract_dir, filename)
            image = cv2.imread(src_path)
            if image is None: continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if math.isnan(variance) or math.isinf(variance): variance = 0.0
            
            is_blurry_val = bool(variance < threshold)
            all_scores.append(variance)
            
            target_dir = blur_dir if is_blurry_val else sharp_dir
            if not is_blurry_val: sharp_scores.append(variance)
            
            dst_path = os.path.join(target_dir, filename)
            shutil.move(src_path, dst_path) # PHYSICALLY MOVE FILE
            
            preview_url = f"/images/{os.path.relpath(dst_path, os.path.abspath(DEFAULT_OUTPUT_DIR))}" if is_internal else None
            results.append({"filename": str(filename), "url": preview_url, "score": round(variance, 2), "is_blurry": is_blurry_val})
            
            if i % 10 == 0 or i == len(raw_files) - 1:
                await websocket.send_json({"type": "progress", "current": i + 1, "total": len(raw_files), "message": f"SORTING: {i+1}/{len(raw_files)}"})

        # Final cleanup
        try: shutil.rmtree(temp_extract_dir)
        except: pass

        avg_all = sum(all_scores) / len(all_scores) if all_scores else 0
        avg_sharp = sum(sharp_scores) / len(sharp_scores) if sharp_scores else 0

        await websocket.send_json({
            "type": "complete",
            "results": results,
            "output_dir": str(base_output_dir),
            "analytics": {
                "avg_all": float(round(avg_all, 2)),
                "avg_sharp": float(round(avg_sharp, 2)),
                "total_count": len(all_scores),
                "sharp_count": len(sharp_scores),
                "blur_count": len(all_scores) - len(sharp_scores)
            }
        })

    except WebSocketDisconnect: print("WebSocket disconnected")
    except Exception as e:
        print(f"WS Error: {str(e)}")
        traceback.print_exc()
        try: await websocket.send_json({"type": "error", "message": str(e)})
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
