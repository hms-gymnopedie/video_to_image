import os
import shutil
import uuid
import subprocess
import cv2
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import ffmpeg

app = FastAPI()

# Ultra-permissive CORS for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"DEBUG: {request.method} {request.url}")
    response = await call_next(request)
    print(f"DEBUG: Response Status: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "Video to Image API is running", "version": "1.3.2"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.3.2"}

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve output images statically
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

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
        # Use ffprobe to get detailed stream information
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        format_info = probe.get('format', {})

        if not video_stream:
            raise HTTPException(status_code=400, detail="No video stream found in file")

        # Calculate FPS accurately
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
            "bitrate": format_info.get('bit_rate'),
            "codec": video_stream.get('codec_name')
        }
    except Exception as e:
        print(f"Metadata error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FFprobe error: {str(e)}")

@app.post("/process/{file_id}")
async def process_video(
    file_id: str,
    fps: float = Form(1.0),
    scale: str = Form("-1:-1"),
    qscale: int = Form(2),
    naming_rule: str = Form("frame_%04d.jpg"),
    threshold: float = Form(100.0)
):
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id):
            video_path = os.path.join(UPLOAD_DIR, f)
            break
    
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    # Create a specific output folder for this request
    request_output_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(request_output_dir, exist_ok=True)

    # 1. FFmpeg Extraction
    output_pattern = os.path.join(request_output_dir, naming_rule)
    try:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', w=scale.split(':')[0], h=scale.split(':')[1])
            .output(output_pattern, qscale=qscale)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}")

    # 2. OpenCV Blur Detection (Laplacian Variance)
    results = []
    for filename in sorted(os.listdir(request_output_dir)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(request_output_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            results.append({
                "filename": filename,
                "url": f"/images/{file_id}/{filename}",
                "score": round(variance, 2),
                "is_blurry": variance < threshold
            })

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
