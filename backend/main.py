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

# --- Initial Setup ---
UPLOAD_DIR, OUTPUT_DIR, MASK_DIR = "uploads", "output", "masks"
for d in [UPLOAD_DIR, OUTPUT_DIR, MASK_DIR]: os.makedirs(d, exist_ok=True)

# --- PyTorch 2.6 Compatibility ---
original_torch_load = torch.load
@functools.wraps(original_torch_load)
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# --- AI Engine State ---
MODEL_CONFIGS = {
    "vit_b": {"checkpoint": "sam_vit_b_01ec10.pth", "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/sam_vit_b_01ec10.pth"},
    "vit_l": {"checkpoint": "sam_vit_l_0b31ee.pth", "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/sam_vit_l_0b31ee.pth"},
    "vit_h": {"checkpoint": "sam_vit_h_4b8939.pth", "url": "https://huggingface.co/ybelkada/segment-anything/resolve/main/sam_vit_h_4b8939.pth"}
}

current_model_type = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = None
is_model_ready = False

def load_sam_model(model_type):
    global predictor, current_model_type, is_model_ready
    checkpoint = MODEL_CONFIGS[model_type]["checkpoint"]
    if not os.path.exists(checkpoint) or os.path.getsize(checkpoint) < 100*1024*1024:
        print(f"ERROR: Model file {checkpoint} missing or incomplete.")
        is_model_ready = False
        return False
    try:
        if predictor is not None: del predictor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        current_model_type, is_model_ready = model_type, True
        print(f"SUCCESS: SAM {model_type} loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"LOAD ERROR: {e}"); is_model_ready = False; return False

@app.on_event("startup")
async def startup():
    # Attempt load but don't block server if it fails
    asyncio.create_task(asyncio.to_thread(load_sam_model, "vit_b"))

@app.get("/health")
async def health(): 
    return {"status": "ok", "version": "3.5.0", "ai_ready": is_model_ready, "model": current_model_type, "device": DEVICE}

@app.post("/change-model/{model_type}")
async def change_model(model_type: str):
    if model_type not in MODEL_CONFIGS: raise HTTPException(status_code=400)
    if load_sam_model(model_type): return {"status": "success"}
    raise HTTPException(status_code=500, detail="Model file missing. Please download it manually.")

# --- Standard Logic ---
@app.get("/directories")
async def list_dirs():
    def get_struct(p):
        d = {'name': os.path.basename(p), 'path': p, 'children': []}
        try:
            with os.scandir(p) as it:
                for e in it:
                    if e.is_dir() and not e.name.startswith('.'): d['children'].append(get_struct(e.path))
        except: pass
        return d
    return get_struct(os.path.abspath(OUTPUT_DIR))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{id}{os.path.splitext(file.filename)[1]}")
    with open(path, "wb") as b: shutil.copyfileobj(file.file, b)
    return {"file_id": id}

@app.get("/metadata/{id}")
async def meta(id: str):
    p = next((os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(id)), None)
    if not p: raise HTTPException(404)
    s = next(s for s in ffmpeg.probe(p)['streams'] if s['codec_type'] == 'video')
    return {"duration": float(ffmpeg.probe(p)['format']['duration']), "width": int(s['width']), "height": int(s['height']), "avg_frame_rate": s['avg_frame_rate']}

@app.websocket("/ws/process/{id}")
async def ws_process(ws: WebSocket, id: str):
    await ws.accept()
    try:
        data = await ws.receive_json()
        fps, threshold, out = float(data.get('fps', 1.0)), float(data.get('threshold', 100.0)), data.get('output_path', "")
        vid = next(os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.startswith(id))
        base = os.path.abspath(out) if out else os.path.join(os.path.abspath(OUTPUT_DIR), id)
        os.makedirs(base, exist_ok=True)
        s_dir, b_dir = os.path.join(base, "sharp"), os.path.join(base, "blur")
        for d in [s_dir, b_dir]: 
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        tmp = os.path.join(base, f"tmp_{uuid.uuid4().hex}")
        os.makedirs(tmp, exist_ok=True)
        await ws.send_json({"type": "status", "message": "1/2 EXTRACTING"})
        def run():
            try: (ffmpeg.input(vid).filter('fps', fps=fps).output(os.path.join(tmp, "f_%04d.jpg"), qscale=2).overwrite_output().run(capture_stdout=True, capture_stderr=True)); return True
            except: return False
        await asyncio.to_thread(run)
        await ws.send_json({"type": "status", "message": "2/2 ANALYZING"})
        files = sorted([f for f in os.listdir(tmp) if f.endswith(".jpg")])
        res = []
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(tmp, f))
            var = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            is_b = var < threshold
            shutil.move(os.path.join(tmp, f), os.path.join(b_dir if is_b else s_dir, f))
            p_url = f"/images/{os.path.relpath(os.path.join(b_dir if is_b else s_dir, f), os.path.abspath(OUTPUT_DIR))}"
            res.append({"filename": f, "url": p_url, "score": round(var, 2), "is_blurry": is_b, "full_path": os.path.join(b_dir if is_b else s_dir, f)})
            if i % 10 == 0: await ws.send_json({"type": "progress", "current": i+1, "total": len(files)})
        shutil.rmtree(tmp)
        await ws.send_json({"type": "complete", "results": res, "output_dir": base})
    except Exception as e: await ws.send_json({"type": "error", "message": str(e)})

@app.post("/segment")
async def segment(data: dict):
    if not is_model_ready: raise HTTPException(503, "AI Engine Offline")
    img = cv2.imread(data['image_path'])
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    m, s, _ = predictor.predict(point_coords=np.array(data['points']), point_labels=np.array([1]*len(data['points'])), multimask_output=True)
    mask_overlay = img.copy()
    mask_overlay[m[np.argmax(s)]] = [0, 0, 255]
    path = os.path.join(MASK_DIR, f"m_{uuid.uuid4().hex}.png")
    cv2.imwrite(path, mask_overlay)
    return {"mask_url": f"/masks/{os.path.basename(path)}"}

@app.post("/sync-folders")
async def sync(data: dict):
    sd, bd = os.path.join(data['output_dir'], "sharp"), os.path.join(data['output_dir'], "blur")
    for r in data['results']:
        is_b = r['score'] < data['threshold']
        fn = r['filename']
        src = os.path.join(bd if os.path.exists(os.path.join(bd, fn)) else sd, fn)
        dst = os.path.join(bd if is_b else sd, fn)
        if src != dst: shutil.move(src, dst)
    return {"status": "success"}

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/masks", StaticFiles(directory=MASK_DIR), name="masks")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
