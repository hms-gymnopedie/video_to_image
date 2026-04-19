import importlib
import sys

modules = [
    'fastapi',
    'uvicorn',
    'requests',
    'cv2',
    'ffmpeg',
    'torch',
    'segment_anything',
    'websockets'
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

missing = []
for m in modules:
    try:
        importlib.import_module(m)
        print(f"[OK] {m}")
    except ImportError:
        print(f"[MISSING] {m}")
        missing.append(m)

if not missing:
    print("\nAll required modules are installed correctly.")
    sys.exit(0)
else:
    print(f"\nMissing modules: {', '.join(missing)}")
    sys.exit(1)
