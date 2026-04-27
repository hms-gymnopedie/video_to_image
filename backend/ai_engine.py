"""SAM 2.1 + Grounded-SAM 2 engine, with optional SAM 3 / SAM 3.1 backend.

Two backends coexist behind a single facade:
    backend="sam2"  -> SAM 2.1 (Hiera tiny/small/base_plus/large) + Grounding DINO
    backend="sam3"  -> SAM 3 / SAM 3.1 (text prompts native, no DINO needed)

Exposes:
    SAMEngine.load_sam2(size)            - SAM 2.1 image + video predictors (HF Hub)
    SAMEngine.load_sam3(variant)         - SAM 3 / SAM 3.1 image + video predictors
    SAMEngine.load_model(name)           - dispatcher that picks sam2/sam3 by name
    SAMEngine.load_grounding_dino()      - SAM 2 text path (transformers)
    SAMEngine.segment_image_points(...)  - Phase 1: single frame, point prompts
    SAMEngine.propagate_video(...)       - Phase 2: one-click video propagation
    SAMEngine.segment_text(...)          - Phase 4: text -> mask
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

# SAM 3 / SAM 3.1 — gated checkpoints on HF Hub (request access + hf auth login).
# Text prompts are native; Grounding DINO is not used in this backend.
SAM3_HF_IDS = {
    "sam3":   "facebook/sam3",
    "sam3.1": "facebook/sam3.1",
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
    """Map legacy aliases and SAM 3 names to canonical model identifiers.

    Returns one of: tiny / small / base_plus / large / sam3 / sam3.1.
    The same string is later dispatched to the right backend by `resolve_backend`.
    """
    if name in SAM2_HF_IDS or name in SAM3_HF_IDS:
        return name
    if name in LEGACY_ALIAS:
        return LEGACY_ALIAS[name]
    raise ValueError(f"Unknown model size: {name}")


def resolve_backend(name: str) -> str:
    """Pick the backend ("sam2" or "sam3") that owns a given model name."""
    canon = resolve_model_size(name)
    if canon in SAM3_HF_IDS:
        return "sam3"
    return "sam2"


class SAMEngine:
    """Facade over SAM 2.1 and SAM 3 / SAM 3.1.

    The active backend is recorded in `self.backend`. All public inference
    methods (segment_image_points, propagate_video, segment_text) dispatch
    on this attribute, so callers in main.py do not need to change.
    """

    def __init__(self):
        self.device = get_device()
        self.backend: str = "sam2"            # "sam2" | "sam3"
        self.current_size: Optional[str] = None

        # SAM 2.1 state
        self.image_predictor = None
        self.video_predictor = None
        self.dino_processor = None
        self.dino_model = None

        # SAM 3 / 3.1 state
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3_video_predictor = None

    # ----------------------------------------------------------- readiness flags
    @property
    def is_ready(self) -> bool:
        if self.backend == "sam3":
            return self.sam3_processor is not None
        return self.image_predictor is not None

    @property
    def is_text_ready(self) -> bool:
        # SAM 3 has native text prompts; if the SAM 3 image stack is loaded,
        # text masking is implicitly available (no Grounding DINO required).
        if self.backend == "sam3":
            return self.sam3_processor is not None
        return self.dino_model is not None

    @property
    def is_video_ready(self) -> bool:
        if self.backend == "sam3":
            return self.sam3_video_predictor is not None
        return self.video_predictor is not None

    def status(self) -> dict:
        return {
            "ai_ready": self.is_ready,
            "text_ready": self.is_text_ready,
            "video_ready": self.is_video_ready,
            "backend": self.backend,
            "model": self.current_size,
            "device": self.device,
        }

    # ------------------------------------------------------------- dispatcher
    def load_model(self, name: str) -> bool:
        """Top-level loader that picks the right backend by model name."""
        backend = resolve_backend(name)
        if backend == "sam3":
            return self.load_sam3(name)
        return self.load_sam2(name)

    # ------------------------------------------------------------------ SAM 2.1
    def load_sam2(self, size: str = "base_plus") -> bool:
        size = resolve_model_size(size)
        if size not in SAM2_HF_IDS:
            print(f"[AI] {size} is a SAM 3 model — routing to load_sam3()")
            return self.load_sam3(size)
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
            # Free any previously-loaded predictor (sam2 or sam3) before swap.
            self._free_sam2()
            self._free_sam3()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[AI] Loading SAM 2.1 ({size}) from {hf_id} on {self.device}...")
            self.image_predictor = SAM2ImagePredictor.from_pretrained(
                hf_id, device=self.device
            )
            self.video_predictor = build_sam2_video_predictor_hf(
                hf_id, device=self.device
            )
            self.backend = "sam2"
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

    # ------------------------------------------------------------ SAM 3 / 3.1
    def load_sam3(self, variant: str = "sam3.1") -> bool:
        """Load SAM 3 / SAM 3.1 image + video stacks via the `sam3` package.

        The checkpoints are gated on HuggingFace; the user must run
        `hf auth login` (or set HF_TOKEN) after being granted access on
        https://huggingface.co/facebook/sam3 and .../sam3.1 .

        API verified against facebookresearch/sam3 main branch (2026-04). If
        upstream renames symbols, only this method needs adjustment.
        """
        canon = resolve_model_size(variant)
        if canon not in SAM3_HF_IDS:
            print(f"[AI] {variant} is not a SAM 3 model — routing to load_sam2()")
            return self.load_sam2(variant)
        hf_id = SAM3_HF_IDS[canon]

        try:
            from sam3.model_builder import (
                build_sam3_image_model,
                build_sam3_video_predictor,
            )
            from sam3.model.sam3_image_processor import Sam3Processor
        except ModuleNotFoundError as e:
            missing = e.name or "(unknown)"
            if missing.split(".")[0] == "sam3":
                print(
                    "[AI] sam3 package missing. Install with:\n"
                    '     pip install "git+https://github.com/facebookresearch/sam3.git"\n'
                    "     and run `hf auth login` after checkpoint access is granted."
                )
            else:
                # sam3 imports a transitive dep that isn't installed (common:
                # einops, omegaconf, pycocotools). Tell the user exactly which.
                print(
                    f"[AI] sam3 is installed but its dependency '{missing}' is "
                    f"missing. Install it with: pip install {missing}"
                )
            print(f"[AI] Import error: {e}")
            return False
        except ImportError as e:
            print(f"[AI] sam3 import failed: {e}")
            return False

        try:
            self._free_sam2()
            self._free_sam3()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[AI] Loading {canon.upper()} ({hf_id}) on {self.device}...")
            # Image stack
            model = build_sam3_image_model(hf_id, device=self.device)
            self.sam3_model = model
            self.sam3_processor = Sam3Processor(model)

            # Video stack — best-effort: if the installed sam3 build does not
            # expose a video predictor, image+text still works.
            try:
                self.sam3_video_predictor = build_sam3_video_predictor(
                    hf_id, device=self.device
                )
            except Exception as ve:
                print(f"[AI] SAM 3 video predictor unavailable: {ve}")
                self.sam3_video_predictor = None

            self.backend = "sam3"
            self.current_size = canon
            print(f"[AI] {canon.upper()} READY on {self.device}.")
            return True
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"[AI] {canon.upper()} load FAILED: {e}\n"
                "     If this is a 401/403, check HuggingFace access for "
                f"{hf_id} and that you ran `hf auth login`."
            )
            self.sam3_model = None
            self.sam3_processor = None
            self.sam3_video_predictor = None
            return False

    # ----------------------------------------------------------- backend cleanup
    def _free_sam2(self) -> None:
        self.image_predictor = None
        self.video_predictor = None

    def _free_sam3(self) -> None:
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3_video_predictor = None

    # ----------------------------------------------------------- Grounding DINO
    def load_grounding_dino(self) -> bool:
        # SAM 3 has native text prompts — DINO is unnecessary in that backend.
        if self.backend == "sam3":
            print("[AI] SAM 3 backend: text prompts are native, skipping Grounding DINO.")
            return self.is_text_ready

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
            raise RuntimeError(f"{self.backend.upper()} not loaded")
        if self.backend == "sam3":
            return self._sam3_segment_points(image_path, points, labels)
        return self._sam2_segment_points(image_path, points, labels)

    def _sam2_segment_points(
        self, image_path: str, points: List[List[float]], labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
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

    def _sam3_segment_points(
        self, image_path: str, points: List[List[float]], labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """SAM 3 image predictor with point prompts."""
        if labels is None:
            labels = [1] * len(points)

        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            state = self.sam3_processor.set_image(image)
            # SAM 3 accepts visual prompts via set_visual_prompt(points, labels);
            # surface the keyword names that match the upstream API.
            out = self.sam3_processor.set_visual_prompt(
                state=state,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
            )

        masks = _extract_sam3_masks(out)
        scores = _extract_sam3_scores(out)
        if masks is None or len(masks) == 0:
            h, w = image.size[1], image.size[0]
            return np.zeros((h, w), dtype=bool), 0.0
        if scores is not None and len(scores) > 0:
            best = int(np.argmax(scores))
            return masks[best].astype(bool), float(scores[best])
        # No scores returned — take the union of all returned masks.
        return np.any(masks.astype(bool), axis=0), 1.0

    def segment_image_boxes(
        self, image_path: str, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_ready:
            raise RuntimeError(f"{self.backend.upper()} not loaded")
        if self.backend == "sam3":
            # SAM 3 with box prompts — supported via set_visual_prompt(boxes=...).
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode():
                state = self.sam3_processor.set_image(image)
                out = self.sam3_processor.set_visual_prompt(
                    state=state, boxes=boxes.astype(np.float32)
                )
            masks = _extract_sam3_masks(out)
            scores = _extract_sam3_scores(out)
            if masks is None:
                h, w = image.size[1], image.size[0]
                return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)
            if scores is None:
                scores = np.ones((len(masks),), dtype=np.float32)
            return masks, scores

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
        """Yield (frame_idx, bool_mask) for every frame in the sequence."""
        if self.backend == "sam3":
            yield from self._sam3_propagate_video(
                frames_dir, init_frame_idx, points, labels
            )
            return
        yield from self._sam2_propagate_video(
            frames_dir, init_frame_idx, points, labels
        )

    def _sam2_propagate_video(
        self,
        frames_dir: str,
        init_frame_idx: int,
        points: List[List[float]],
        labels: Optional[List[int]] = None,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """SAM 2's init_state sorts frames by int(basename), so arbitrary names
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

    def _sam3_propagate_video(
        self,
        frames_dir: str,
        init_frame_idx: int,
        points: List[List[float]],
        labels: Optional[List[int]] = None,
        text_prompt: Optional[str] = None,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """SAM 3 session-based video propagation.

        The session API (start_session / add_prompt / iter_outputs) is the
        primary upstream interface. Same staging trick as SAM 2 to normalize
        frame names to integer order.
        """
        if self.sam3_video_predictor is None:
            raise RuntimeError(
                "SAM 3 video predictor not loaded. The installed sam3 build "
                "may not include video support — switch to SAM 2.1 for "
                "video propagation, or update the sam3 package."
            )

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
        staging = tempfile.mkdtemp(prefix="sam3_propagate_")
        try:
            for i, fname in enumerate(frames_sorted):
                src = os.path.abspath(os.path.join(frames_dir, fname))
                link = os.path.join(staging, f"{i:06d}{ext}")
                try:
                    os.symlink(src, link)
                except OSError:
                    shutil.copy2(src, link)

            vp = self.sam3_video_predictor
            with torch.inference_mode():
                session_id = vp.start_session(video_path=staging)
                prompt_kwargs = {
                    "session_id": session_id,
                    "frame_idx": init_frame_idx,
                }
                if text_prompt:
                    prompt_kwargs["text"] = text_prompt
                else:
                    prompt_kwargs["points"] = np.array(points, dtype=np.float32)
                    prompt_kwargs["labels"] = np.array(labels, dtype=np.int32)
                vp.add_prompt(**prompt_kwargs)

                # Upstream exposes an iterator of per-frame outputs. Method name
                # has been `iter_outputs` / `propagate_in_video` across versions
                # — try both for compatibility.
                iter_fn = getattr(vp, "iter_outputs", None) or getattr(
                    vp, "propagate_in_video", None
                )
                if iter_fn is None:
                    raise RuntimeError(
                        "SAM 3 video predictor has no iter_outputs / "
                        "propagate_in_video method — please open an issue "
                        "with the installed sam3 version."
                    )

                for item in iter_fn(session_id):
                    frame_idx, mask = _unpack_sam3_video_item(item)
                    yield frame_idx, mask
        finally:
            try:
                if hasattr(self.sam3_video_predictor, "end_session"):
                    self.sam3_video_predictor.end_session(session_id)
            except Exception:
                pass
            shutil.rmtree(staging, ignore_errors=True)

    def detect_text(
        self,
        image_path: str,
        queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> dict:
        """Grounding DINO: text queries -> boxes/scores/labels (SAM 2 path)."""
        if self.backend == "sam3":
            raise RuntimeError(
                "detect_text() is SAM-2-only. Use segment_text() with the "
                "SAM 3 backend — text is processed natively."
            )
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
        if self.backend == "sam3":
            return self._sam3_segment_text(
                image_path, queries, box_threshold, text_threshold
            )
        return self._sam2_segment_text(
            image_path, queries, box_threshold, text_threshold
        )

    def _sam2_segment_text(
        self,
        image_path: str,
        queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, dict]:
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

    def _sam3_segment_text(
        self,
        image_path: str,
        queries: List[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, dict]:
        """SAM 3 native text prompts.

        SAM 3 returns instance masks for every concept directly; no detector
        + segmenter chaining and no Grounding DINO needed. We pass each query
        as a separate prompt and union the resulting masks so the caller's
        contract (single union mask + detection metadata) is preserved.
        """
        if not self.is_text_ready:
            raise RuntimeError("SAM 3 image processor not loaded")

        image = Image.open(image_path).convert("RGB")
        h, w = image.size[1], image.size[0]
        all_masks = []
        all_scores: List[float] = []
        all_labels: List[str] = []

        with torch.inference_mode():
            state = self.sam3_processor.set_image(image)
            for q in queries:
                qn = q.strip().lower()
                if not qn:
                    continue
                # SAM 3.1 supports a per-prompt threshold; pass it when accepted.
                try:
                    out = self.sam3_processor.set_text_prompt(
                        state=state, prompt=qn, threshold=box_threshold
                    )
                except TypeError:
                    out = self.sam3_processor.set_text_prompt(
                        state=state, prompt=qn
                    )

                masks = _extract_sam3_masks(out)
                scores = _extract_sam3_scores(out)
                if masks is None or len(masks) == 0:
                    continue
                # Optional secondary threshold filter.
                if scores is not None and text_threshold > 0:
                    keep = scores >= text_threshold
                    if not keep.any():
                        continue
                    masks = masks[keep]
                    scores = scores[keep]
                all_masks.append(masks.astype(bool))
                if scores is not None:
                    all_scores.extend(float(s) for s in scores)
                else:
                    all_scores.extend([1.0] * len(masks))
                all_labels.extend([qn] * len(masks))

        det = {
            "boxes": np.zeros((0, 4), dtype=np.float32),  # SAM 3 may not expose boxes
            "scores": np.array(all_scores, dtype=np.float32),
            "labels": all_labels,
        }
        if not all_masks:
            return np.zeros((h, w), dtype=bool), det
        stacked = np.concatenate(all_masks, axis=0)
        union = np.any(stacked, axis=0)
        return union, det


# ----------------------------------------------------------- SAM 3 output utils
def _extract_sam3_masks(out) -> Optional[np.ndarray]:
    """Pull a stack of HxW boolean masks out of a SAM 3 processor return value.

    SAM 3 has shipped under several output shapes (dict, dataclass, tuple).
    We accept anything that surfaces a `masks` field of shape (N, H, W) or
    (N, 1, H, W). Returns None if no masks are present.
    """
    if out is None:
        return None
    masks = None
    if isinstance(out, dict):
        masks = out.get("masks")
    else:
        masks = getattr(out, "masks", None)
    if masks is None:
        return None
    arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr.astype(bool)


def _extract_sam3_scores(out) -> Optional[np.ndarray]:
    if out is None:
        return None
    scores = None
    if isinstance(out, dict):
        scores = out.get("scores", out.get("iou_scores"))
    else:
        scores = getattr(out, "scores", None) or getattr(out, "iou_scores", None)
    if scores is None:
        return None
    arr = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    return arr.astype(np.float32).reshape(-1)


def _unpack_sam3_video_item(item) -> Tuple[int, np.ndarray]:
    """Convert one SAM 3 video iterator item to (frame_idx, bool_mask)."""
    # Common shapes: (frame_idx, masks, ...), or dict with frame_idx + masks.
    if isinstance(item, tuple) and len(item) >= 2:
        frame_idx = int(item[0])
        masks = item[-1] if len(item) == 2 else item[2]  # (idx, obj_ids, masks)
    elif isinstance(item, dict):
        frame_idx = int(item["frame_idx"])
        masks = item.get("masks")
    else:
        frame_idx = int(getattr(item, "frame_idx"))
        masks = getattr(item, "masks", None)

    if masks is None:
        raise RuntimeError(f"SAM 3 video item has no masks field: {item!r}")
    arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    # Reduce (N, H, W) or (N, 1, H, W) or logits -> single bool mask via union.
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.dtype != bool:
        # Logits convention: positive -> foreground.
        arr = arr > 0.0 if arr.dtype.kind == "f" else arr.astype(bool)
    if arr.ndim == 3:
        arr = np.any(arr, axis=0)
    return frame_idx, arr.astype(bool)


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


# ----------------------------------------------------------- Duplicate detection
def compute_dhash(image_path: str, hash_size: int = 8) -> Optional[int]:
    """Difference hash (dHash). Returns a 64-bit int for hash_size=8.

    Robust to small lighting changes and JPEG noise. Hamming distance between
    two hashes correlates with perceptual similarity:
      0       — pixel-identical
      1-3     — same scene, minor changes (compression, slight motion)
      4-8     — same scene, noticeable change
      >10     — different content
    """
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to (hash_size+1, hash_size) so we can compute hash_size**2 diffs.
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    """Bit-count of XOR — population count for Python ints."""
    return bin(a ^ b).count("1")


def detect_duplicate_frames(
    frames_dir: str,
    threshold: int = 5,
) -> list:
    """Scan a frames directory in sorted order; flag each frame whose hash is
    within `threshold` Hamming distance of its most recent kept anchor.

    Returns list of dicts:
        {filename, full_path, anchor_filename, distance}
    Each entry is a candidate to be considered a duplicate of `anchor_filename`.

    Strategy: an "anchor" is the last frame we decided to keep. As we walk the
    sorted frames, any frame within `threshold` of the current anchor is a
    duplicate; otherwise it becomes the new anchor.
    """
    import os as _os

    if not _os.path.isdir(frames_dir):
        return []
    files = sorted(
        f for f in _os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    duplicates: list = []
    anchor_name: Optional[str] = None
    anchor_hash: Optional[int] = None
    for f in files:
        path = _os.path.join(frames_dir, f)
        h = compute_dhash(path)
        if h is None:
            continue
        if anchor_hash is None:
            anchor_name = f
            anchor_hash = h
            continue
        d = hamming(h, anchor_hash)
        if d <= threshold:
            duplicates.append({
                "filename": f,
                "full_path": path,
                "anchor_filename": anchor_name,
                "distance": d,
            })
        else:
            anchor_name = f
            anchor_hash = h
    return duplicates


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
