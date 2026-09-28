"""InsightFace adapter restricted to face detection and recognition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from insight_cup.common.onnx import cpu_session_options
from insight_cup.paths import MODELS_ROOT


REQUIRED_MODULES = frozenset({"detection", "recognition"})
DEFAULT_MODEL_NAME = "face_only_r50"
DEFAULT_MODEL_ROOT = MODELS_ROOT / "face"
DEFAULT_MODEL_FILES = frozenset({"det_10g.onnx", "w600k_r50.onnx"})


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """One detected face and its normalized ArcFace embedding."""

    embedding: np.ndarray
    detection_score: float
    bbox_xyxy: tuple[float, float, float, float]
    crop_shape: tuple[int, int]
    roll_degrees: float = 0.0


def _resolve_providers(provider: str) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    normalized = provider.strip().lower()
    if normalized == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install a compatible "
                "onnxruntime-gpu build or use --provider cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError("provider must be one of: auto, cpu, cuda")


class InsightFaceEmbeddingEngine:
    """Load only SCRFD detection and ArcFace recognition from InsightFace."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_root: str | Path = DEFAULT_MODEL_ROOT,
        provider: str = "auto",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.20,
        upsample_min_side: int = 320,
        max_upsample: float = 6.0,
        cpu_threads: int = 4,
    ) -> None:
        try:
            from insight_cup.vendor.insightface_core.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "The focused InsightFace runtime could not be imported. "
                "Install requirements-face.txt first."
            ) from exc

        if not 0.0 < det_thresh < 1.0:
            raise ValueError("det_thresh must be between 0 and 1")
        if len(det_size) != 2 or min(det_size) <= 0:
            raise ValueError("det_size must contain two positive integers")

        self.model_name = model_name
        self.model_root = Path(model_root).expanduser().resolve()
        self.providers = _resolve_providers(provider)
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.det_thresh = float(det_thresh)
        self.upsample_min_side = max(0, int(upsample_min_side))
        self.max_upsample = max(1.0, float(max_upsample))
        self.cpu_threads = max(1, int(cpu_threads))

        model_dir = self.model_root / "models" / self.model_name
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Face model directory not found: {model_dir}. "
                "Run tools/data/install_face_models.py first."
            )
        if self.model_name == DEFAULT_MODEL_NAME:
            model_files = frozenset(path.name for path in model_dir.glob("*.onnx"))
            if model_files != DEFAULT_MODEL_FILES:
                raise RuntimeError(
                    "The face-only model directory must contain exactly "
                    f"{sorted(DEFAULT_MODEL_FILES)}, got {sorted(model_files)}"
                )

        session_options = None
        if self.providers == ["CPUExecutionProvider"]:
            import onnxruntime as ort

            session_options = cpu_session_options(ort, self.cpu_threads)

        self._app = FaceAnalysis(
            name=self.model_name,
            root=str(self.model_root),
            allowed_modules=sorted(REQUIRED_MODULES),
            providers=self.providers,
            sess_options=session_options,
        )
        loaded_modules = frozenset(self._app.models)
        if loaded_modules != REQUIRED_MODULES:
            raise RuntimeError(
                "Expected only InsightFace detection and recognition models, "
                f"but loaded: {sorted(loaded_modules)}"
            )

        ctx_id = 0 if "CUDAExecutionProvider" in self.providers else -1
        self._app.prepare(
            ctx_id=ctx_id,
            det_thresh=self.det_thresh,
            det_size=self.det_size,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "insightface_core",
            "upstream_commit": "7fadd420c2351d0ffa8cac403421c1a3ed733365",
            "model_name": self.model_name,
            "model_root": str(self.model_root),
            "loaded_modules": sorted(REQUIRED_MODULES),
            "providers": list(self.providers),
            "det_size": list(self.det_size),
            "det_thresh": self.det_thresh,
            "cpu_threads": self.cpu_threads,
        }

    def _prepare_crop(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a non-empty BGR image")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image must have positive dimensions")

        if self.upsample_min_side <= 0:
            return image, 1.0
        scale = max(1.0, self.upsample_min_side / min(height, width))
        scale = min(scale, self.max_upsample)
        if scale <= 1.0:
            return image, 1.0
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        return resized, scale

    @staticmethod
    def _face_rank(face: Any, image_shape: tuple[int, int]) -> float:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        area = width * height
        image_height, image_width = image_shape
        center_x = (bbox[0] + bbox[2]) * 0.5
        center_y = (bbox[1] + bbox[3]) * 0.5
        offset = abs(center_x - image_width * 0.5) / max(image_width, 1)
        offset += abs(center_y - image_height * 0.5) / max(image_height, 1)
        return area * max(0.5, 1.0 - 0.15 * offset)

    def extract(self, image: np.ndarray) -> FaceEmbeddingResult | None:
        """Return the dominant face embedding, or None when no face is detected."""

        prepared, scale = self._prepare_crop(image)
        original_shape = (int(image.shape[0]), int(image.shape[1]))
        faces = self._app.get(prepared, max_num=0)
        if not faces:
            return None

        face = max(faces, key=lambda item: self._face_rank(item, prepared.shape[:2]))
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vector_norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(vector_norm) or vector_norm <= 0.0:
            return None
        vector = vector / vector_norm
        bbox = np.asarray(face.bbox, dtype=np.float32) / scale
        roll_degrees = 0.0
        keypoints = getattr(face, "kps", None)
        if keypoints is not None:
            points = np.asarray(keypoints, dtype=np.float32)
            if points.shape[0] >= 2 and points.shape[1] >= 2:
                left_eye, right_eye = points[0], points[1]
                roll_degrees = math.degrees(
                    math.atan2(
                        float(right_eye[1] - left_eye[1]),
                        float(right_eye[0] - left_eye[0]),
                    )
                )
        return FaceEmbeddingResult(
            embedding=vector,
            detection_score=float(face.det_score),
            bbox_xyxy=tuple(float(value) for value in bbox[:4]),
            crop_shape=original_shape,
            roll_degrees=roll_degrees,
        )
