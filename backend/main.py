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
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import ffmpeg
import torch
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAM Model Configurations
MODEL_CONFIGS = {
    "vit_b": {
        "checkpoint": "sam_vit_b_01ec10.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec10.pth"
    },
    "vit_l": {
        "checkpoint": "sam_vit_l_0b31ee.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b31ee.pth"
    },
    "vit_h": {
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
}

current_model_type = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = None

def download_model(model_type):
    config = MODEL_CONFIGS[model_type]
    checkpoint = config["checkpoint"]
    if not os.path.exists(checkpoint):
        print(f"Downloading {model_type} model ({checkpoint})... This may take several minutes.")
        response = requests.get(config["url"], stream=True)
        with open(checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete: {checkpoint}")

def load_sam_model(model_type):
    global predictor, current_model_type
    try:
        download_model(model_type)
        print(f"Loading {model_type} into memory ({DEVICE})...")
        if predictor is not None:
            del predictor
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        sam = sam_model_registry[model_type](checkpoint=MODEL_CONFIGS[model_type]["checkpoint"])
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        current_model_type = model_type
        print(f"SAM {model_type} loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_sam_model("vit_b")

@app.post("/change-model/{model_type}")
async def change_model(model_type: str):
    if model_type not in MODEL_CONFIGS: raise HTTPException(status_code=400, detail="Invalid model type")
    success = load_sam_model(model_type)
    if success: return {"status": "success", "model": model_type, "device": DEVICE}
    else: raise HTTPException(status_code=500, detail=f"Failed to load model {model_type}")

@app.get("/health")
async def health_check(): 
    return {"status": "ok", "version": "3.3.2", "device": DEVICE, "current_model": current_model_type}

UPLOAD_DIR = "uploads"
DEFAULT_OUTPUT_DIR = "output"
MASK_DIR = "masks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="images")
app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")

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
        return {"duration": float(format_info.get('duration', 0)), "width": int(video_stream.get('width', 0)), "height": int(video_stream.get('height', 0)), "avg_frame_rate": f"{round(fps, 2)} FPS"}
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

        base_output_dir = os.path.abspath(output_path.strip()) if output_path and str(output_path).strip() else os.path.join(os.path.abspath(DEFAULT_OUTPUT_DIR), file_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        sharp_dir = os.path.join(base_output_dir, "sharp")
        blur_dir = os.path.join(base_output_dir, "blur")
        for d in [sharp_dir, blur_dir]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        temp_extract_dir = os.path.join(base_output_dir, f"temp_{uuid.uuid4().hex}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        await websocket.send_json({"type": "status", "message": "1/3 EXTRACTING"})
        stream = ffmpeg.input(video_path).filter('fps', fps=fps)
        if scale != "-1:-1":
            try:
                w, h = scale.split(':')
                stream = stream.filter('scale', w=w, h=h)
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
                    await websocket.send_json({"type": "progress", "current": int(len(files)), "message": f"EXTRACTING: {len(files)} generated"})
                except: break
                await asyncio.sleep(1.0)

        monitor_task = asyncio.create_task(monitor_progress())
        success = await asyncio.to_thread(run_ffmpeg_sync)
        monitor_task.cancel()

        if not success:
            await websocket.send_json({"type": "error", "message": "FFmpeg failed"})
            return

        await websocket.send_json({"type": "status", "message": "2/3 ANALYZING & SORTING"})
        time.sleep(1.0)
        
        results = []
        raw_files = sorted([f for f in os.listdir(temp_extract_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        is_internal = base_output_dir.startswith(os.path.abspath(DEFAULT_OUTPUT_DIR))
        
        for i, filename in enumerate(raw_files):
            src_path = os.path.join(temp_extract_dir, filename)
            image = cv2.imread(src_path)
            if image is None: continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if math.isnan(variance) or math.isinf(variance): variance = 0.0
            
            is_blurry = bool(variance < threshold)
            target_dir = blur_dir if is_blurry else sharp_dir
            dst_path = os.path.join(target_dir, filename)
            shutil.move(src_path, dst_path)
            
            preview_url = f"/images/{os.path.relpath(dst_path, os.path.abspath(DEFAULT_OUTPUT_DIR))}" if is_internal else None
            results.append({"filename": str(filename), "url": preview_url, "score": round(variance, 2), "is_blurry": is_blurry, "full_path": dst_path})
            
            if i % 10 == 0 or i == len(raw_files) - 1:
                await websocket.send_json({"type": "progress", "current": i + 1, "total": len(raw_files), "message": f"SORTING: {i+1}/{len(raw_files)}"})

        try: shutil.rmtree(temp_extract_dir)
        except: pass

        await websocket.send_json({"type": "complete", "results": results, "output_dir": str(base_output_dir)})

    except WebSocketDisconnect: print("WebSocket disconnected")
    except Exception as e:
        traceback.print_exc()
        try: await websocket.send_json({"type": "error", "message": str(e)})
        except: pass

class SegmentRequest(BaseModel):
    image_path: str
    points: List[List[int]]
    labels: List[int]

@app.post("/segment")
async def segment_image(data: SegmentRequest):
    if predictor is None: raise HTTPException(status_code=503, detail="SAM model not initialized")
    try:
        image = cv2.imread(data.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(point_coords=np.array(data.points), point_labels=np.array(data.labels), multimask_output=True)
        mask = masks[np.argmax(scores)]
        mask_overlay = image.copy()
        mask_overlay[mask] = [0, 0, 255]
        mask_id = str(uuid.uuid4())
        mask_filename = f"mask_{mask_id}.png"
        mask_path = os.path.join(MASK_DIR, mask_filename)
        cv2.imwrite(mask_path, cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
        return {"mask_url": f"/masks/{mask_filename}", "status": "success"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class SyncRequest(BaseModel):
    output_dir: str
    threshold: float
    results: List[dict]

@app.post("/sync-folders")
async def sync_folders(data: SyncRequest):
    try:
        sharp_dir, blur_dir = os.path.join(data.output_dir, "sharp"), os.path.join(data.output_dir, "blur")
        for d in [sharp_dir, blur_dir]: os.makedirs(d, exist_ok=True)
        for res in data.results:
            filename = res['filename']
            should_be_blurry = res['score'] < data.threshold
            current_sharp, current_blur = os.path.join(sharp_dir, filename), os.path.join(blur_dir, filename)
            if should_be_blurry and os.path.exists(current_sharp): shutil.move(current_sharp, current_blur)
            elif not should_be_blurry and os.path.exists(current_blur): shutil.move(current_blur, current_sharp)
        return {"status": "success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
