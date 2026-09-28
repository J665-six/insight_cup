"""Configuration and default paths for the unified runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from insight_cup.paths import DATA_ROOT, MODELS_ROOT, OUTPUTS_ROOT

DEFAULT_YOLO_WEIGHTS = Path(
    "/home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.pt"
)
DEFAULT_FACE_GALLERY = DATA_ROOT / "face_gallery" / "trainv5_gallery_r50.npz"
DEFAULT_FACE_MODEL_ROOT = MODELS_ROOT / "face"
DEFAULT_KNIFE_MODEL_DIR = (
    MODELS_ROOT / "knife" / "pplcnetv2_base_knife10"
)
DEFAULT_KNIFE_RETRIEVAL_MODEL = (
    MODELS_ROOT / "knife" / "ppshitu_v2_retrieval" / "inference.onnx"
)
DEFAULT_KNIFE_GALLERY = DATA_ROOT / "knife_gallery" / "trainv5_ppshitu_v2.npz"
DEFAULT_OUTPUT_ROOT = OUTPUTS_ROOT / "runtime"


@dataclass(frozen=True)
class RuntimeConfig:
    source: str = "realsense"
    weights: Path = DEFAULT_YOLO_WEIGHTS
    output_root: Path = DEFAULT_OUTPUT_ROOT
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    yolo_device: str = "auto"
    yolo_cpu_threads: int = 1
    provider: str = "auto"
    face_gallery: Path = DEFAULT_FACE_GALLERY
    face_model_root: Path = DEFAULT_FACE_MODEL_ROOT
    face_model_name: str = "face_only_r50"
    face_threshold: float = 0.40
    face_min_margin: float = 0.03
    face_detection_threshold: float = 0.18
    face_detection_size: int = 320
    face_upsample_min_side: int = 320
    face_cpu_threads: int = 1
    face_rotation_retry_degrees: float = 0.0
    knife_model_dir: Path = DEFAULT_KNIFE_MODEL_DIR
    knife_mode: str = "classification"
    knife_retrieval_model: Path = DEFAULT_KNIFE_RETRIEVAL_MODEL
    knife_gallery: Path = DEFAULT_KNIFE_GALLERY
    knife_threshold: float = 0.50
    knife_min_margin: float = 0.15
    knife_retrieval_threshold: float = 0.50
    knife_retrieval_min_margin: float = 0.02
    knife_cpu_threads: int = 1
    stage2_refresh_frames: int = 15
    stage2_stable_refresh_frames: int = 30
    stage2_cache_iou: float = 0.55
    stage2_workers: int = 2
    stage2_smoothing: str = "vote"
    stage2_smoothing_window: int = 3
    stage2_switch_confirmations: int = 2
    realsense_serial: str | None = None
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    video_stride: int = 1
    pace_video: bool = True
    max_frames: int | None = None
    save_video: bool = True
    save_crops: bool = True
    show_raw_label: bool = False
    publish_preview: bool = True
    jpeg_quality: int = 82
    preview_width: int = 960
    frame_log_every: int = 30

    def validate(self) -> None:
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        if not self.face_gallery.is_file():
            raise FileNotFoundError(f"Face gallery not found: {self.face_gallery}")
        if not self.face_model_root.is_dir():
            raise FileNotFoundError(f"Face model root not found: {self.face_model_root}")
        if self.knife_mode not in {"classification", "retrieval"}:
            raise ValueError("Knife mode must be classification or retrieval")
        if self.knife_mode == "classification" and not self.knife_model_dir.is_dir():
            raise FileNotFoundError(f"Knife model directory not found: {self.knife_model_dir}")
        if self.knife_mode == "retrieval":
            if not self.knife_retrieval_model.is_file():
                raise FileNotFoundError(
                    f"Knife retrieval model not found: {self.knife_retrieval_model}"
                )
            if not self.knife_gallery.is_file():
                raise FileNotFoundError(
                    f"Knife retrieval gallery not found: {self.knife_gallery}"
                )
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("YOLO confidence must be in (0, 1]")
        if not 0.0 < self.iou <= 1.0:
            raise ValueError("YOLO IoU must be in (0, 1]")
        for name, value in (
            ("knife threshold", self.knife_threshold),
            ("knife margin", self.knife_min_margin),
            ("knife retrieval threshold", self.knife_retrieval_threshold),
            ("knife retrieval margin", self.knife_retrieval_min_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.image_size <= 0:
            raise ValueError("Image size must be positive")
        if self.yolo_cpu_threads <= 0:
            raise ValueError("YOLO CPU threads must be positive")
        if self.video_stride <= 0:
            raise ValueError("Video stride must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("Max frames must be positive")
        if self.knife_cpu_threads <= 0:
            raise ValueError("Knife CPU threads must be positive")
        if self.face_cpu_threads <= 0:
            raise ValueError("Face CPU threads must be positive")
        if not 0.0 <= self.face_rotation_retry_degrees <= 45.0:
            raise ValueError("Face rotation retry must be between zero and 45 degrees")
        if self.stage2_refresh_frames <= 0:
            raise ValueError("Stage-2 refresh interval must be positive")
        if self.stage2_stable_refresh_frames < self.stage2_refresh_frames:
            raise ValueError(
                "Stable stage-2 refresh interval must be at least the normal interval"
            )
        if not 0.0 < self.stage2_cache_iou <= 1.0:
            raise ValueError("Stage-2 cache IoU must be in (0, 1]")
        if self.stage2_workers <= 0:
            raise ValueError("Stage-2 worker count must be positive")
        if self.stage2_smoothing not in {"none", "mean", "vote"}:
            raise ValueError("Stage-2 smoothing must be none, mean, or vote")
        if self.stage2_smoothing_window <= 0:
            raise ValueError("Stage-2 smoothing window must be positive")
        if self.stage2_switch_confirmations <= 0:
            raise ValueError("Stage-2 switch confirmations must be positive")
        if min(self.camera_width, self.camera_height, self.camera_fps) <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        if self.preview_width <= 0:
            raise ValueError("Preview width must be positive")
        if self.frame_log_every <= 0:
            raise ValueError("Frame log interval must be positive")

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
