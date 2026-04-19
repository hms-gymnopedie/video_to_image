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

# Robust CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAM Model Configs
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
        print(f"Downloading {model_type} model ({checkpoint})...")
        response = requests.get(config["url"], stream=True)
        with open(checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

def load_sam_model(model_type):
    global predictor, current_model_type
    try:
        download_model(model_type)
        if predictor is not None:
            del predictor
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        sam = sam_model_registry[model_type](checkpoint=MODEL_CONFIGS[model_type]["checkpoint"])
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        current_model_type = model_type
        print(f"SAM {model_type} loaded.")
        return True
    except Exception as e:
        print(f"Load Error: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_sam_model("vit_b")

@app.get("/health")
async def health_check(): 
    return {"status": "ok", "version": "3.3.3", "device": DEVICE, "model": current_model_type}

@app.post("/change-model/{model_type}")
async def change_model(model_type: str):
    if model_type not in MODEL_CONFIGS: raise HTTPException(status_code=400, detail="Invalid model")
    if load_sam_model(model_type): return {"status": "success", "model": model_type}
    raise HTTPException(status_code=500, detail="Model load failed")

# Directory Browser APIs
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
    return get_dir_structure(os.path.abspath("output"))

class CreateDirRequest(BaseModel):
    parent_path: str
    new_name: str

@app.post("/create-directory")
async def create_directory(data: CreateDirRequest):
    new_path = os.path.join(data.parent_path, data.new_name)
    os.makedirs(new_path, exist_ok=True)
    return {"status": "success", "path": new_path}

# File Upload
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join("uploads", f"{file_id}{os.path.splitext(file.filename)[1]}")
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    return {"file_id": file_id, "filename": file.filename}

@app.get("/metadata/{file_id}")
async def get_metadata(file_id: str):
    video_path = None
    for f in os.listdir("uploads"):
        if f.startswith(file_id):
            video_path = os.path.join("uploads", f)
            break
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
        
        # Resolve paths
        video_path = next(os.path.join("uploads", f) for f in os.listdir("uploads") if f.startswith(file_id))
        base_output_dir = os.path.abspath(output_path) if output_path else os.path.join(os.path.abspath("output"), file_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        sharp_dir, blur_dir = os.path.join(base_output_dir, "sharp"), os.path.join(base_output_dir, "blur")
        for d in [sharp_dir, blur_dir]: 
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        temp_dir = os.path.join(base_output_dir, f"temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        # Extraction
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
        
        # Analysis
        await websocket.send_json({"type": "status", "message": "2/3 ANALYZING"})
        raw_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".jpg")])
        results = []
        for i, f in enumerate(raw_files):
            img = cv2.imread(os.path.join(temp_dir, f))
            var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            is_blur = var < threshold
            target = blur_dir if is_blur else sharp_dir
            shutil.move(os.path.join(temp_dir, f), os.path.join(target, f))
            
            preview_url = f"/images/{os.path.relpath(os.path.join(target, f), os.path.abspath('output'))}"
            results.append({"filename": f, "url": preview_url, "score": round(var, 2), "is_blurry": bool(is_blur), "full_path": os.path.join(target, f)})
            
            if i % 5 == 0: await websocket.send_json({"type": "progress", "current": i+1, "total": len(raw_files), "message": f"SORTING: {i+1}/{len(raw_files)}"})

        shutil.rmtree(temp_dir)
        await websocket.send_json({"type": "complete", "results": results, "output_dir": base_output_dir})
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})

@app.post("/segment")
async def segment_image(data: dict):
    image = cv2.imread(data['image_path'])
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(point_coords=np.array(data['points']), point_labels=np.array([1]*len(data['points'])), multimask_output=True)
    mask_overlay = image.copy()
    mask_overlay[masks[np.argmax(scores)]] = [0, 0, 255]
    mask_path = os.path.join("masks", f"mask_{uuid.uuid4().hex}.png")
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(mask_path, mask_overlay)
    return {"mask_url": f"/{mask_path.replace(os.sep, '/')}"}

@app.post("/sync-folders")
async def sync_folders(data: dict):
    sharp_dir, blur_dir = os.path.join(data['output_dir'], "sharp"), os.path.join(data['output_dir'], "blur")
    for res in data['results']:
        is_blur = res['score'] < data['threshold']
        src = os.path.join(blur_dir if os.path.exists(os.path.join(blur_dir, res['filename'])) else sharp_dir, res['filename'])
        dst = os.path.join(blur_dir if is_blur else sharp_dir, res['filename'])
        if src != dst: shutil.move(src, dst)
    return {"status": "success"}

# Statics
app.mount("/images", StaticFiles(directory="output"), name="images")
app.mount("/masks", StaticFiles(directory="masks"), name="masks")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
