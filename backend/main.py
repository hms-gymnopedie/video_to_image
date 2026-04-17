import os
import shutil
import uuid
import subprocess
import cv2
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional
import ffmpeg

app = FastAPI()

# Ultra-permissive CORS
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
        print(f"DEBUG: Response Status: {response.status_code}")
        return response
    except Exception as e:
        print(f"DEBUG: SERVER_ERROR: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "trace": traceback.format_exc()},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.get("/")
async def root():
    return {"message": "Video to Image API is running", "version": "1.3.6"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.3.6"}

UPLOAD_DIR = "uploads"
DEFAULT_OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="images")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"file_id": file_id, "filename": file.filename, "path": video_path}

@app.get("/metadata/{file_id}")
async def get_metadata(file_id: str):
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id):
            video_path = os.path.join(UPLOAD_DIR, f)
            break
    
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        format_info = probe.get('format', {})

        if not video_stream:
            raise HTTPException(status_code=400, detail="No video stream found")

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
    except Exception as e:
        print(f"Metadata error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    # Use custom path if provided, else use default internal output dir
    if output_path and output_path.strip():
        final_output_dir = os.path.abspath(output_path.strip())
    else:
        final_output_dir = os.path.join(os.path.abspath(DEFAULT_OUTPUT_DIR), file_id)

    os.makedirs(final_output_dir, exist_ok=True)
    output_pattern = os.path.join(final_output_dir, naming_rule)
    
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps=fps)
        if scale != "-1:-1":
            w, h = scale.split(':')
            stream = ffmpeg.filter(stream, 'scale', w=w, h=h)
            
        stream = ffmpeg.output(stream, output_pattern, qscale=qscale)
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
        raise HTTPException(status_code=500, detail=f"FFmpeg process failed: {stderr}")

    results = []
    for filename in sorted(os.listdir(final_output_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(final_output_dir, filename)
            image = cv2.imread(img_path)
            if image is None: continue
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # For result display, we still need a URL. 
            # If it's a custom path outside of DEFAULT_OUTPUT_DIR, we handle it as a relative link for now.
            results.append({
                "filename": filename,
                "url": f"/images/{file_id}/{filename}" if not output_path else f"/images/{filename}", # Simplification for preview
                "score": round(variance, 2),
                "is_blurry": variance < threshold,
                "full_path": img_path
            })

    return {"results": results, "output_dir": final_output_dir}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
