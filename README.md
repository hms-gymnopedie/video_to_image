# Video to Image Converter Dashboard

A dashboard for converting videos to images with blur preprocessing using FFmpeg and OpenCV.

## Prerequisites
- Node.js & npm
- Python 3.11+
- FFmpeg (installed and available in PATH)

## How to Run

### 1. Backend
```bash
cd backend
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt # (I will create this)
python main.py
```
The backend will run at `http://localhost:8000`.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
The frontend will run at `http://localhost:5173`.

## Features
- Video metadata extraction.
- Frame extraction with customizable FPS, Scale, and Quality (Qscale).
- Blur detection using Laplacian Variance.
- Results gallery with Sharp vs Blurry categorization.
