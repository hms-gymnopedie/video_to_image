"""SAM 2.1 + Grounded-SAM 2 engine.

Exposes:
    SAMEngine.load_sam2(size)            - image + video predictors via HF Hub
    SAMEngine.load_grounding_dino()      - text-prompted detector (transformers)
    SAMEngine.segment_image_points(...)  - Phase 1: single frame, point prompts
    SAMEngine.propagate_video(...)       - Phase 2: one-click video propagation
    SAMEngine.segment_text(...)          - Phase 4: text -> boxes -> masks
"""
import os
import functools
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# --- PyTorch 2.6 Compatibility ---
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


SAM2_HF_IDS = {
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}

# Legacy SAM-v1 model names from the existing frontend dropdown.
LEGACY_ALIAS = {
    "vit_b": "tiny",
    "vit_l": "base_plus",
    "vit_h": "large",
}

GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_size(name: str) -> str:
    """Map legacy aliases (vit_b/l/h) to SAM 2.1 sizes."""
    if name in SAM2_HF_IDS:
        return name
    if name in LEGACY_ALIAS:
        return LEGACY_ALIAS[name]
    raise ValueError(f"Unknown model size: {name}")


class SAMEngine:
    def __init__(self):
        self.device = get_device()
        self.current_size: Optional[str] = None
        self.image_predictor = None
        self.video_predictor = None
        self.dino_processor = None
        self.dino_model = None

    @property
    def is_ready(self) -> bool:
        return self.image_predictor is not None

    @property
    def is_text_ready(self) -> bool:
        return self.dino_model is not None

    def status(self) -> dict:
        return {
            "ai_ready": self.is_ready,
            "text_ready": self.is_text_ready,
            "model": self.current_size,
            "device": self.device,
        }

    # ------------------------------------------------------------------ SAM 2.1
    def load_sam2(self, size: str = "base_plus") -> bool:
        size = resolve_model_size(size)
        hf_id = SAM2_HF_IDS[size]

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2_video_predictor_hf
        except ImportError as e:
            print(
                "[AI] sam2 package missing. Install with:\n"
                '     pip install "git+https://github.com/facebookresearch/sam2.git"'
            )
            print(f"[AI] Import error: {e}")
            return False

        try:
            if self.image_predictor is not None:
                del self.image_predictor
            if self.video_predictor is not None:
                del self.video_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[AI] Loading SAM 2.1 ({size}) from {hf_id} on {self.device}...")
            self.image_predictor = SAM2ImagePredictor.from_pretrained(
                hf_id, device=self.device
            )
            self.video_predictor = build_sam2_video_predictor_hf(
                hf_id, device=self.device
            )
            self.current_size = size
            print(f"[AI] SAM 2.1 {size} READY on {self.device}.")
            return True
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[AI] SAM 2.1 load FAILED: {e}")
            self.image_predictor = None
            self.video_predictor = None
            return False

    # ----------------------------------------------------------- Grounding DINO
    def load_grounding_dino(self) -> bool:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError:
            print("[AI] transformers missing. pip install transformers>=4.40")
            return False

        try:
            print(f"[AI] Loading Grounding DINO ({GROUNDING_DINO_ID}) on {self.device}...")
            self.dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_ID
            ).to(self.device)
            self.dino_model.eval()
            print("[AI] Grounding DINO READY.")
            return True
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[AI] Grounding DINO load FAILED: {e}")
            self.dino_model = None
            self.dino_processor = None
            return False

    # ----------------------------------------------------------- Inference APIs
    def segment_image_points(
        self, image_path: str, points: List[List[float]], labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        if not self.is_ready:
            raise RuntimeError("SAM 2.1 not loaded")
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if labels is None:
            labels = [1] * len(points)

        with torch.inference_mode():
            self.image_predictor.set_image(img_rgb)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=True,
            )

        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])

    def segment_image_boxes(
        self, image_path: str, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_ready:
            raise RuntimeError("SAM 2.1 not loaded")
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            self.image_predictor.set_image(img_rgb)
            masks, scores, _ = self.image_predictor.predict(
                box=boxes.astype(np.float32),
                multimask_output=False,
            )
        return masks, scores

    def propagate_video(
        self,
        frames_dir: str,
        init_frame_idx: int,
        points: List[List[float]],
        labels: Optional[List[int]] = None,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_idx, bool_mask) for every frame in the sequence.

        SAM 2's init_state sorts frames by int(basename), so arbitrary names
        like `f_0001.jpg` fail. We stage integer-named symlinks in a temp dir
        to satisfy the loader while keeping the caller unaware.
        """
        if self.video_predictor is None:
            raise RuntimeError("SAM 2.1 video predictor not loaded")

        import shutil
        import tempfile

        if labels is None:
            labels = [1] * len(points)

        frames_sorted = sorted(
            f for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        )
        if not frames_sorted:
            raise ValueError(f"No .jpg/.jpeg frames in {frames_dir}")
        if not (0 <= init_frame_idx < len(frames_sorted)):
            raise ValueError(
                f"init_frame_idx {init_frame_idx} out of range [0, {len(frames_sorted)})"
            )

        ext = os.path.splitext(frames_sorted[0])[1].lower()
        staging = tempfile.mkdtemp(prefix="sam2_propagate_")
        try:
            for i, fname in enumerate(frames_sorted):
                src = os.path.abspath(os.path.join(frames_dir, fname))
                link = os.path.join(staging, f"{i:06d}{ext}")
                try:
                    os.symlink(src, link)
                except OSError:
                    shutil.copy2(src, link)

            with torch.inference_mode():
                state = self.video_predictor.init_state(video_path=staging)
                self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=init_frame_idx,
                    obj_id=1,
                    points=np.array(points, dtype=np.float32),
                    labels=np.array(labels, dtype=np.int32),
                )
                for frame_idx, _obj_ids, mask_logits in self.video_predictor.propagate_in_video(
                    state
                ):
                    mask = (mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(bool)
                    yield frame_idx, mask
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def detect_text(
        self,
        image_path: str,
        queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> dict:
        """Grounding DINO: text queries -> boxes/scores/labels."""
        if not self.is_text_ready:
            raise RuntimeError("Grounding DINO not loaded")

        image = Image.open(image_path).convert("RGB")
        prompt = ". ".join(q.strip().lower() for q in queries) + "."
        inputs = self.dino_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.dino_model(**inputs)

        # transformers >=4.51 renamed `box_threshold` -> `threshold`. Try the
        # new name first, fall back for older installs.
        post = self.dino_processor.post_process_grounded_object_detection
        try:
            results = post(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": results["labels"],
        }

    def segment_text(
        self,
        image_path: str,
        queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, dict]:
        """Text prompt -> union mask of every detected instance."""
        det = self.detect_text(image_path, queries, box_threshold, text_threshold)
        boxes = det["boxes"]
        if len(boxes) == 0:
            import cv2

            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            return np.zeros((h, w), dtype=bool), det

        masks, _ = self.segment_image_boxes(image_path, boxes)
        # masks: (N, 1, H, W) or (N, H, W) depending on SAM2 version.
        if masks.ndim == 4:
            masks = masks[:, 0]
        union = np.any(masks.astype(bool), axis=0)
        return union, det


# ----------------------------------------------------------- Mask I/O helpers
def save_binary_mask(
    mask: np.ndarray,
    path: str,
    invert: bool = False,
    combine: bool = False,
    skip_empty: bool = True,
) -> bool:
    """Save a boolean mask as 0/255 PNG.

    Args:
        mask: boolean array. True = detected region (before invert).
        invert: if True, the detected region becomes 0 (exclude) and
                everything else 255 (keep). Use for reconstruction masks.
        combine: if True and path already exists, AND the new KEEP mask
                 with the existing KEEP mask (so both runs' excluded
                 regions are excluded). Requires invert=True to be
                 semantically meaningful.
        skip_empty: if True and mask has no True pixels (nothing detected),
                    do not write the file at all. Keeps reconstruction-mask
                    folders free of no-op files.

    Returns:
        True if a file was written, False if skipped.
    """
    import cv2

    m = mask.astype(bool)
    if skip_empty and not m.any():
        return False

    if invert:
        m = ~m

    if combine and os.path.exists(path):
        existing = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if existing is not None and existing.shape == m.shape:
            existing_keep = existing > 127
            m = m & existing_keep

    out = (m.astype(np.uint8)) * 255
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    return True


def save_overlay(
    image_path: str, mask: np.ndarray, out_path: str, color=(0, 0, 255), alpha: float = 0.5
) -> None:
    """Red-tinted overlay for UI preview (legacy behavior)."""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, blended)


def generate_mask_previews(
    scene_dir: str, frames_subdir: str = "sharp"
) -> list:
    """Composite each existing mask over its source frame for UI preview.

    Returns a list of dicts: {filename, overlay_rel, coverage} where
    overlay_rel is the path relative to scene_dir (for URL construction)
    and coverage is the fraction of pixels marked as EXCLUDE.
    """
    import cv2

    frames_dir = os.path.join(scene_dir, frames_subdir)
    mask_dir = os.path.join(scene_dir, "masks")
    preview_dir = os.path.join(scene_dir, "mask_preview")

    if not os.path.isdir(frames_dir) or not os.path.isdir(mask_dir):
        return []

    os.makedirs(preview_dir, exist_ok=True)
    results = []
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, f"{base}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        keep = mask > 127
        exclude = ~keep

        overlay_path = os.path.join(preview_dir, f"{base}.jpg")
        save_overlay(
            os.path.join(frames_dir, fname),
            exclude,
            overlay_path,
            color=(0, 0, 255),
            alpha=0.45,
        )
        coverage = float(exclude.sum()) / exclude.size
        results.append(
            {
                "filename": fname,
                "overlay_rel": os.path.join("mask_preview", f"{base}.jpg"),
                "coverage": round(coverage, 4),
            }
        )
    return results
