import os
import shutil
import uuid
import subprocess
import cv2
import json
import traceback
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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
    print(f"DEBUG: {request.method} {request.url}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"DEBUG: SERVER_ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "trace": traceback.format_exc()},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.3.9"}

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
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
        if '/' in fps_raw:
            num, den = map(int, fps_raw.split('/'))
            fps = round(num / den, 2) if den != 0 else 0
        else:
            fps = float(fps_raw)
        return {
            "duration": float(format_info.get('duration', 0)),
            "width": int(video_stream.get('width', 0)),
            "height": int(video_stream.get('height', 0)),
            "avg_frame_rate": f"{fps} FPS",
        }
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{file_id}")
async def process_video(
    file_id: str,
    fps: float = Form(1.0),
    scale: str = Form("-1:-1"),
    qscale: int = Form(2),
    naming_rule: str = Form("frame_%04d.jpg"),
    threshold: float = Form(100.0),
    output_path: Optional[str] = Form(None)
):
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id):
            video_path = os.path.join(UPLOAD_DIR, f)
            break
    if not video_path: raise HTTPException(status_code=404, detail="Video not found")
    
    final_output_dir = os.path.abspath(output_path.strip()) if output_path and output_path.strip() else os.path.join(os.path.abspath(DEFAULT_OUTPUT_DIR), file_id)
    os.makedirs(final_output_dir, exist_ok=True)
    
    try:
        # Build FFmpeg command
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps=fps)
        if scale != "-1:-1":
            w, h = scale.split(':')
            stream = ffmpeg.filter(stream, 'scale', w=w, h=h)
        
        output_pattern = os.path.join(final_output_dir, naming_rule)
        process = (
            ffmpeg
            .output(stream, output_pattern, qscale=qscale)
            .overwrite_output()
        )
        
        print(f"DEBUG: Running FFmpeg to {output_pattern}")
        process.run(capture_stdout=True, capture_stderr=True)
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
        print(f"DEBUG: FFMPEG_ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {error_msg[:200]}...")

    # Wait a brief moment for filesystem to sync
    time.sleep(0.5)

    results = []
    is_internal = final_output_dir.startswith(os.path.abspath(DEFAULT_OUTPUT_DIR))
    
    files = sorted([f for f in os.listdir(final_output_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    for filename in files:
        img_path = os.path.join(final_output_dir, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        preview_url = None
        if is_internal:
            rel_path = os.path.relpath(img_path, os.path.abspath(DEFAULT_OUTPUT_DIR))
            preview_url = f"/images/{rel_path}"
            
        results.append({
            "filename": filename,
            "url": preview_url,
            "score": round(variance, 2),
            "is_blurry": variance < threshold
        })

    return {"results": results, "output_dir": final_output_dir, "total": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
