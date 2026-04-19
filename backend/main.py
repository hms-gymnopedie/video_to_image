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
import functools
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import ffmpeg
import torch
from segment_anything import sam_model_registry, SamPredictor

# Ensure essential directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
MASK_DIR = "masks"
for d in [UPLOAD_DIR, OUTPUT_DIR, MASK_DIR]:
    os.makedirs(d, exist_ok=True)

# --- PyTorch 2.6 Compatibility Fix ---
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
# -------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# High-stability Mirror URLs (Hugging Face)
MODEL_CONFIGS = {
    "vit_b": {
        "checkpoint": "sam_vit_b_01ec10.pth",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec10.pth"
    },
    "vit_l": {
        "checkpoint": "sam_vit_l_0b31ee.pth",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_l_0b31ee.pth"
    },
    "vit_h": {
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth"
    }
}

current_model_type = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = None

def download_model(model_type):
    config = MODEL_CONFIGS[model_type]
    checkpoint = config["checkpoint"]
    url = config["url"]
    
    if os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 100 * 1024 * 1024:
        return True

    print(f"--- SAM MODEL DOWNLOAD START ({model_type}) ---")
    print(f"Source: {url}")
    
    # Try system curl first (most robust on Mac)
    try:
        print("Executing system curl download...")
        subprocess.run(["curl", "-L", url, "-o", checkpoint], check=True)
        if os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 100 * 1024 * 1024:
            print("Download successful via curl.")
            return True
    except Exception as e:
        print(f"Curl failed, falling back to requests: {e}")

    # Fallback to requests with browser headers
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(checkpoint, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        print("Download successful via requests.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: All download methods failed for {model_type}.")
        print(f"Error detail: {e}")
        print(f"PLEASE MANUALLY DOWNLOAD FROM: {url} AND PLACE IN backend/ FOLDER.")
        return False

def load_sam_model(model_type):
    global predictor, current_model_type
    try:
        if not download_model(model_type): return False
        if predictor is not None:
            del predictor
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        print(f"Loading SAM {model_type} into memory ({DEVICE})...")
        sam = sam_model_registry[model_type](checkpoint=MODEL_CONFIGS[model_type]["checkpoint"])
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        current_model_type = model_type
        print(f"SUCCESS: SAM {model_type} is now active.")
        return True
    except Exception as e:
        print(f"LOAD ERROR: {e}")
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    load_sam_model("vit_b")

@app.get("/health")
async def health_check(): 
    return {"status": "ok", "version": "3.3.9", "device": DEVICE, "model": current_model_type}

@app.post("/change-model/{model_type}")
async def change_model(model_type: str):
    if model_type not in MODEL_CONFIGS: raise HTTPException(status_code=400, detail="Invalid model")
    if load_sam_model(model_type): return {"status": "success", "model": model_type}
    raise HTTPException(status_code=500, detail="Model switch failed. See server logs.")

# --- Remaining Core Logic (Directories, Upload, Process, Segment, Sync) ---
@app.get("/directories")
async def list_directories():
    def get_dir_structure(root_path):
        d = {'name': os.path.basename(root_path), 'path': root_path, 'children': []}
        try:
            with os.scandir(root_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.name.startswith('.'):
                        d['children'].append(get_dir_structure(entry.path))
        except: pass
        return d
    return get_dir_structure(os.path.abspath(OUTPUT_DIR))

class CreateDirRequest(BaseModel):
    parent_path: str
    new_name: str

@app.post("/create-directory")
async def create_directory(data: CreateDirRequest):
    new_path = os.path.join(data.parent_path, data.new_name)
    os.makedirs(new_path, exist_ok=True)
    return {"status": "success", "path": new_path}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}{os.path.splitext(file.filename)[1]}")
    with open(video_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    return {"file_id": file_id, "filename": file.filename}

@app.get("/metadata/{file_id}")
async def get_metadata(file_id: str):
    video_path = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id)), None)
    if not video_path: raise HTTPException(status_code=404, detail="Not found")
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = eval(video_stream.get('avg_frame_rate', '30/1'))
    return {"duration": float(probe['format'].get('duration', 0)), "width": int(video_stream['width']), "height": int(video_stream['height']), "avg_frame_rate": f"{round(fps, 2)} FPS"}

@app.websocket("/ws/process/{file_id}")
async def websocket_process(websocket: WebSocket, file_id: str):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        fps, threshold, output_path = float(data.get('fps', 1.0)), float(data.get('threshold', 100.0)), data.get('output_path', "")
        video_path = next(os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id))
        base_output_dir = os.path.abspath(output_path) if output_path else os.path.join(os.path.abspath(OUTPUT_DIR), file_id)
        os.makedirs(base_output_dir, exist_ok=True)
        sharp_dir, blur_dir = os.path.join(base_output_dir, "sharp"), os.path.join(base_output_dir, "blur")
        for d in [sharp_dir, blur_dir]: 
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        temp_dir = os.path.join(base_output_dir, f"temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        await websocket.send_json({"type": "status", "message": "1/3 EXTRACTING"})
        stream = ffmpeg.input(video_path).filter('fps', fps=fps)
        def run_ffmpeg():
            try:
                (ffmpeg.output(stream, os.path.join(temp_dir, "frame_%04d.jpg"), qscale=2).overwrite_output().run(capture_stdout=True, capture_stderr=True))
                return True
            except: return False
        ffmpeg_task = asyncio.to_thread(run_ffmpeg)
        while not (await asyncio.to_thread(lambda: os.path.exists(temp_dir))): await asyncio.sleep(0.5)
        await ffmpeg_task
        await websocket.send_json({"type": "status", "message": "2/3 ANALYZING"})
        raw_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".jpg")])
        results = []
        for i, f in enumerate(raw_files):
            img = cv2.imread(os.path.join(temp_dir, f))
            var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            target = blur_dir if var < threshold else sharp_dir
            shutil.move(os.path.join(temp_dir, f), os.path.join(target, f))
            preview_url = f"/images/{os.path.relpath(os.path.join(target, f), os.path.abspath(OUTPUT_DIR))}"
            results.append({"filename": f, "url": preview_url, "score": round(var, 2), "is_blurry": bool(var < threshold), "full_path": os.path.join(target, f)})
            if i % 5 == 0: await websocket.send_json({"type": "progress", "current": i+1, "total": len(raw_files), "message": f"SORTING: {i+1}/{len(raw_files)}"})
        shutil.rmtree(temp_dir)
        await websocket.send_json({"type": "complete", "results": results, "output_dir": base_output_dir})
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})

@app.post("/segment")
async def segment_image(data: dict):
    if predictor is None: raise HTTPException(status_code=503, detail="SAM not loaded")
    image = cv2.imread(data['image_path'])
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(point_coords=np.array(data['points']), point_labels=np.array([1]*len(data['points'])), multimask_output=True)
    mask_overlay = image.copy()
    mask_overlay[masks[np.argmax(scores)]] = [0, 0, 255]
    mask_path = os.path.join(MASK_DIR, f"mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(mask_path, mask_overlay)
    return {"mask_url": f"/masks/{os.path.basename(mask_path)}"}

@app.post("/sync-folders")
async def sync_folders(data: dict):
    sharp_dir, blur_dir = os.path.join(data['output_dir'], "sharp"), os.path.join(data['output_dir'], "blur")
    for res in data['results']:
        is_blur = res['score'] < data['threshold']
        src = os.path.join(blur_dir if os.path.exists(os.path.join(blur_dir, res['filename'])) else sharp_dir, res['filename'])
        dst = os.path.join(blur_dir if is_blur else sharp_dir, res['filename'])
        if src != dst: shutil.move(src, dst)
    return {"status": "success"}

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
