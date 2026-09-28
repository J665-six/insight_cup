"""End-to-end source reader, inference pipeline, annotation, and persistence."""

from __future__ import annotations

import io
import logging
import re
import statistics
import threading
import time
from collections import deque
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from insight_cup.face.engine import InsightFaceEmbeddingEngine
from insight_cup.face.gallery import FaceGallery
from insight_cup.face.service import FaceRecognitionService
from insight_cup.knife.engine import PaddleClasONNXEngine
from insight_cup.knife.retrieval import PaddleClasRetrievalEngine
from insight_cup.knife.service import KnifeRecognitionService

from .config import RuntimeConfig
from .contracts import DetectedRegion, Stage2Decision, YoloDebugInfo
from .detector import YoloMajorClassDetector
from .logging_utils import EventJournal, write_json
from .router import RecognitionRouter
from .sources import RealSenseColorSource, is_realsense_source, is_video_source
from .stage2 import AsyncStage2Scheduler, FinalStage2Result, Stage2Resolution
from .state import RuntimeState


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
COLORS = {
    "b0": (45, 156, 225),
    "f": (52, 166, 92),
    "k": (67, 75, 218),
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def create_session_directory(output_root: Path, session_id: str | None = None) -> Path:
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if session_id:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("._-")
        if not normalized:
            raise ValueError("Session ID contains no usable characters")
    else:
        normalized = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidate = root / normalized
    suffix = 1
    while candidate.exists():
        candidate = root / f"{normalized}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


class RecognitionRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        session_dir: Path,
        state: RuntimeState,
        journal: EventJournal,
        logger: logging.Logger,
    ) -> None:
        cv2.setNumThreads(4)
        self.config = config
        self.session_dir = session_dir
        self.state = state
        self.journal = journal
        self.logger = logger
        self.detector: YoloMajorClassDetector | None = None
        self.router: RecognitionRouter | None = None
        self.stage2_scheduler: AsyncStage2Scheduler | None = None
        self._model_metadata: dict[str, Any] = {}
        self._media_outputs: list[str] = []
        self._frame_intervals_ms: deque[float] = deque(maxlen=30)
        self._last_frame_completed_at: float | None = None
        self._last_event_frame: np.ndarray | None = None
        self._last_processed_index: int | None = None
        self._last_source_label: str | None = None
        self._last_source_time_ms: float | None = None
        self._track_debug: dict[int, YoloDebugInfo] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> threading.Thread:
        if self._thread is not None:
            raise RuntimeError("Runtime has already been started")
        self._thread = threading.Thread(
            target=self.run,
            name="recognition-runtime",
            daemon=False,
        )
        self._thread.start()
        return self._thread

    def run(self) -> None:
        self._write_manifest()
        try:
            self.config.validate()
            self.state.set_status("loading")
            self._load_models()
            self._write_manifest()
            if self.state.stop_requested:
                self.state.set_status("stopped")
            else:
                self.state.set_status("running")
                self._run_source()
                self._drain_stage2()
                self.state.set_status(
                    "stopped" if self.state.stop_requested else "completed"
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.logger.exception("Runtime failed: %s", message)
            self.state.set_status("error", message)
        finally:
            if self.stage2_scheduler is not None:
                self.stage2_scheduler.close()
            self._write_summary()
            self.journal.close()

    def _load_models(self) -> None:
        self.state.set_module("yolo", "loading", str(self.config.weights))
        self.logger.info("LOAD module=yolo model=%s", self.config.weights)
        started = time.perf_counter()
        detector = YoloMajorClassDetector(
            weights=self.config.weights,
            confidence=self.config.confidence,
            iou=self.config.iou,
            image_size=self.config.image_size,
            device=self.config.yolo_device,
            cpu_threads=self.config.yolo_cpu_threads,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.detector = detector
        self._model_metadata["yolo"] = detector.metadata
        self.state.set_module("yolo", "ready", f"{elapsed:.0f} ms")
        self.logger.info("READY module=yolo load_ms=%.1f", elapsed)

        self.state.set_module("face", "loading", str(self.config.face_gallery))
        self.logger.info(
            "LOAD module=face gallery=%s model=%s",
            self.config.face_gallery,
            self.config.face_model_name,
        )
        started = time.perf_counter()
        gallery = FaceGallery.load(self.config.face_gallery)
        upstream_output = (
            nullcontext()
            if self.logger.isEnabledFor(logging.DEBUG)
            else redirect_stdout(io.StringIO())
        )
        with upstream_output:
            face_engine = InsightFaceEmbeddingEngine(
                model_name=self.config.face_model_name,
                model_root=self.config.face_model_root,
                provider=self.config.provider,
                det_size=(
                    self.config.face_detection_size,
                    self.config.face_detection_size,
                ),
                det_thresh=self.config.face_detection_threshold,
                upsample_min_side=self.config.face_upsample_min_side,
                cpu_threads=self.config.face_cpu_threads,
            )
        face_service = FaceRecognitionService(
            face_engine,
            gallery,
            threshold=self.config.face_threshold,
            min_margin=self.config.face_min_margin,
            rotation_retry_degrees=self.config.face_rotation_retry_degrees,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self._model_metadata["face"] = {
            **face_engine.metadata,
            "gallery": str(self.config.face_gallery),
            "identities": gallery.identities,
            "reference_embeddings": int(gallery.labels.size),
            "threshold": self.config.face_threshold,
            "min_margin": self.config.face_min_margin,
            "rotation_retry_degrees": self.config.face_rotation_retry_degrees,
        }
        self.state.set_module(
            "face", "ready", f"{len(gallery.identities)} identities / {elapsed:.0f} ms"
        )
        self.logger.info(
            "READY module=face identities=%d references=%d load_ms=%.1f",
            len(gallery.identities),
            gallery.labels.size,
            elapsed,
        )

        knife_source = (
            self.config.knife_model_dir
            if self.config.knife_mode == "classification"
            else self.config.knife_retrieval_model
        )
        self.state.set_module("knife", "loading", str(knife_source))
        self.logger.info(
            "LOAD module=knife mode=%s model=%s",
            self.config.knife_mode,
            knife_source,
        )
        started = time.perf_counter()
        if self.config.knife_mode == "retrieval":
            knife_engine = PaddleClasRetrievalEngine(
                model_path=self.config.knife_retrieval_model,
                gallery_path=self.config.knife_gallery,
                provider=self.config.provider,
                cpu_threads=self.config.knife_cpu_threads,
            )
            knife_threshold = self.config.knife_retrieval_threshold
            knife_min_margin = self.config.knife_retrieval_min_margin
        else:
            knife_engine = PaddleClasONNXEngine(
                model_dir=self.config.knife_model_dir,
                provider=self.config.provider,
                cpu_threads=self.config.knife_cpu_threads,
            )
            knife_threshold = self.config.knife_threshold
            knife_min_margin = self.config.knife_min_margin
        knife_service = KnifeRecognitionService(
            knife_engine,
            threshold=knife_threshold,
            min_margin=knife_min_margin,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self._model_metadata["knife"] = {
            **knife_engine.metadata,
            "mode": self.config.knife_mode,
            "threshold": knife_threshold,
            "min_margin": knife_min_margin,
        }
        self.state.set_module(
            "knife",
            "ready",
            f"{self.config.knife_mode} / {len(knife_engine.labels)} classes / "
            f"{elapsed:.0f} ms",
        )
        self.logger.info(
            "READY module=knife mode=%s classes=%d load_ms=%.1f",
            self.config.knife_mode,
            len(knife_engine.labels),
            elapsed,
        )
        self.router = RecognitionRouter(face_service, knife_service)
        self.stage2_scheduler = AsyncStage2Scheduler(
            refresh_frames=self.config.stage2_refresh_frames,
            stable_refresh_frames=self.config.stage2_stable_refresh_frames,
            iou_threshold=self.config.stage2_cache_iou,
            workers_per_class=self.config.stage2_workers,
            smoothing_method=self.config.stage2_smoothing,  # type: ignore[arg-type]
            smoothing_window=self.config.stage2_smoothing_window,
            switch_confirmations=self.config.stage2_switch_confirmations,
            face_threshold=self.config.face_threshold,
            face_min_margin=self.config.face_min_margin,
            knife_threshold=knife_threshold,
            knife_min_margin=knife_min_margin,
            asynchronous=(
                is_realsense_source(self.config.source)
                or self.config.source.strip().isdigit()
                or is_video_source(self.config.source)
            ),
        )
        self._model_metadata["temporal"] = {
            "method": self.config.stage2_smoothing,
            "window_size": self.config.stage2_smoothing_window,
            "switch_confirmations": self.config.stage2_switch_confirmations,
            "refresh_frames": self.config.stage2_refresh_frames,
            "stable_refresh_frames": self.config.stage2_stable_refresh_frames,
            "scope": "per_track_id",
        }
        self.logger.info(
            "READY module=temporal method=%s window=%d switch_confirmations=%d "
            "refresh_frames=%d stable_refresh_frames=%d",
            self.config.stage2_smoothing,
            self.config.stage2_smoothing_window,
            self.config.stage2_switch_confirmations,
            self.config.stage2_refresh_frames,
            self.config.stage2_stable_refresh_frames,
        )

    def _run_source(self) -> None:
        if is_realsense_source(self.config.source):
            self._run_realsense()
            return
        source_path = Path(self.config.source).expanduser()
        if source_path.is_dir():
            self._run_image_directory(source_path.resolve())
            return
        if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            self._run_image_directory(source_path.resolve())
            return
        self._run_capture()

    def _run_realsense(self) -> None:
        source = RealSenseColorSource(
            serial=self.config.realsense_serial,
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
        )
        writer: cv2.VideoWriter | None = None
        processed_index = 0
        try:
            metadata = source.start()
            self.state.set_source_info(
                **metadata,
                total_frames=0,
                video_stride=self.config.video_stride,
            )
            source_label = f"realsense:{metadata['serial']}"
            self.logger.info(
                "SOURCE type=realsense model=%s serial=%s stream=color "
                "size=%dx%d fps=%d",
                metadata["model"],
                metadata["serial"],
                metadata["width"],
                metadata["height"],
                metadata["fps"],
            )
            while self.state.wait_until_runnable():
                camera_frame = source.read()
                if camera_frame.index % self.config.video_stride != 0:
                    continue
                frame = camera_frame.image
                if writer is None and self.config.save_video:
                    output_path = self.session_dir / "annotated.mp4"
                    output_fps = max(
                        1.0, self.config.camera_fps / self.config.video_stride
                    )
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not write video: {output_path}")
                    self._media_outputs.append(str(output_path))

                annotated = self._process_frame(
                    frame=frame,
                    frame_index=camera_frame.index,
                    processed_index=processed_index,
                    source_label=source_label,
                    source_time_ms=camera_frame.timestamp_ms,
                )
                if writer is not None:
                    writer.write(annotated)
                processed_index += 1
                if (
                    self.config.max_frames is not None
                    and processed_index >= self.config.max_frames
                ):
                    break
        finally:
            source.close()
            if writer is not None:
                writer.release()

        if processed_index == 0 and not self.state.stop_requested:
            raise RuntimeError("RealSense camera produced no color frames")

    def _run_image_directory(self, source: Path) -> None:
        if source.is_dir():
            image_paths = [
                path
                for path in sorted(source.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            image_paths = [source]
        if self.config.max_frames is not None:
            image_paths = image_paths[: self.config.max_frames]
        if not image_paths:
            raise RuntimeError(f"No supported images found: {source}")

        self.state.set_source_info(type="images", total_frames=len(image_paths))
        output_dir = self.session_dir / "annotated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_index = 0
        for frame_index, image_path in enumerate(image_paths):
            if not self.state.wait_until_runnable():
                break
            frame = cv2.imread(str(image_path))
            if frame is None:
                self.logger.warning("SKIP source=%s reason=read_error", image_path)
                continue
            annotated = self._process_frame(
                frame=frame,
                frame_index=frame_index,
                processed_index=processed_index,
                source_label=str(image_path),
                source_time_ms=None,
            )
            destination = output_dir / f"{frame_index:06d}_{image_path.name}"
            if not cv2.imwrite(str(destination), annotated):
                raise RuntimeError(f"Could not write annotated image: {destination}")
            self._media_outputs.append(str(destination))
            processed_index += 1

    def _run_capture(self) -> None:
        capture_source: int | str
        capture_source = (
            int(self.config.source)
            if self.config.source.strip().isdigit()
            else self.config.source
        )
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {self.config.source}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps < 1.0 or source_fps > 240.0:
            source_fps = 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_type = "camera" if isinstance(capture_source, int) else "video"
        pace_video = source_type == "video" and self.config.pace_video
        self.state.set_source_info(
            type=source_type,
            fps=round(source_fps, 3),
            total_frames=max(total_frames, 0),
            video_stride=self.config.video_stride,
            paced=pace_video,
        )

        writer: cv2.VideoWriter | None = None
        raw_index = 0
        processed_index = 0
        playback_started = time.perf_counter()
        try:
            while self.state.wait_until_runnable():
                ok, frame = capture.read()
                if not ok:
                    break
                if raw_index % self.config.video_stride != 0:
                    raw_index += 1
                    continue
                if pace_video and raw_index > 0:
                    target_time = playback_started + raw_index / source_fps
                    remaining = target_time - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    elif remaining < -1.0:
                        playback_started = time.perf_counter() - raw_index / source_fps
                if writer is None and self.config.save_video:
                    height, width = frame.shape[:2]
                    output_path = self.session_dir / "annotated.mp4"
                    output_fps = max(1.0, source_fps / self.config.video_stride)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not write video: {output_path}")
                    self._media_outputs.append(str(output_path))

                source_time_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                annotated = self._process_frame(
                    frame=frame,
                    frame_index=raw_index,
                    processed_index=processed_index,
                    source_label=self.config.source,
                    source_time_ms=(
                        source_time_ms if np.isfinite(source_time_ms) else None
                    ),
                )
                if writer is not None:
                    writer.write(annotated)

                raw_index += 1
                processed_index += 1
                if (
                    self.config.max_frames is not None
                    and processed_index >= self.config.max_frames
                ):
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if processed_index == 0 and not self.state.stop_requested:
            raise RuntimeError(f"Source produced no frames: {self.config.source}")

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        processed_index: int,
        source_label: str,
        source_time_ms: float | None,
    ) -> np.ndarray:
        if (
            self.detector is None
            or self.router is None
            or self.stage2_scheduler is None
        ):
            raise RuntimeError("Models are not loaded")
        frame_started = time.perf_counter()
        detection_frame = self.detector.detect(frame, frame_index)
        annotated = frame.copy()
        stage2_total_ms = 0.0

        for raw_name in detection_frame.ignored_classes:
            self.logger.warning(
                "DROP frame=%d raw_class=%s reason=unsupported_class",
                frame_index,
                raw_name,
            )

        regions = list(detection_frame.regions)
        for region in regions:
            self.state.record_detection(region.handoff.major_class)

        resolutions = self.stage2_scheduler.resolve_frame(
            [region.handoff for region in regions],
            self.router.route,
            [
                (lambda region=region: self._save_crop(region))
                for region in regions
            ],
        )
        for region, resolution in zip(regions, resolutions):
            if resolution.track_id is not None:
                self._track_debug[resolution.track_id] = region.debug
            decision = resolution.decision
            stage2_ms = resolution.inference_ms
            stage2_total_ms += stage2_ms
            if resolution.has_new_result:
                event = self._build_event(
                    region=region,
                    resolution=resolution,
                    frame=frame,
                    processed_index=processed_index,
                    source_label=source_label,
                    source_time_ms=source_time_ms,
                    yolo_ms=detection_frame.inference_ms,
                )
                self.journal.write(event)
                self.state.record_event(event)
            self._draw_result(annotated, region, decision)

        self._last_event_frame = frame
        self._last_processed_index = processed_index
        self._last_source_label = source_label
        self._last_source_time_ms = source_time_ms

        frame_jpeg: bytes | None = None
        if self.config.publish_preview:
            preview = annotated
            if annotated.shape[1] > self.config.preview_width:
                preview_height = max(
                    1,
                    round(
                        annotated.shape[0]
                        * self.config.preview_width
                        / annotated.shape[1]
                    ),
                )
                preview = cv2.resize(
                    annotated,
                    (self.config.preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg",
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("Could not encode the latest debug frame")
            frame_jpeg = encoded.tobytes()
        frame_completed_at = time.perf_counter()
        total_ms = (frame_completed_at - frame_started) * 1000.0
        if self._last_frame_completed_at is not None:
            interval_ms = (frame_completed_at - self._last_frame_completed_at) * 1000.0
            self._frame_intervals_ms.append(max(interval_ms, 0.001))
        self._last_frame_completed_at = frame_completed_at
        fps = (
            1000.0 / statistics.median(self._frame_intervals_ms)
            if self._frame_intervals_ms
            else 0.0
        )
        timing = {
            "yolo": detection_frame.inference_ms,
            "stage2": stage2_total_ms,
            "total": total_ms,
        }
        self.state.publish_frame(
            jpeg=frame_jpeg,
            frame_index=frame_index,
            processed_frames=processed_index + 1,
            fps=fps,
            timing_ms=timing,
            detection_count=len(detection_frame.regions),
        )
        if (processed_index + 1) % self.config.frame_log_every == 0:
            self.logger.debug(
                "FRAME frame=%d processed=%d detections=%d yolo_ms=%.1f "
                "stage2_ms=%.1f total_ms=%.1f fps=%.2f",
                frame_index,
                processed_index + 1,
                len(detection_frame.regions),
                detection_frame.inference_ms,
                stage2_total_ms,
                total_ms,
                fps,
            )
        return annotated

    def _drain_stage2(self) -> None:
        """Wait for final async results so short videos keep complete logs."""
        scheduler = self.stage2_scheduler
        if scheduler is None or not scheduler.asynchronous:
            return
        started = time.perf_counter()
        idle = scheduler.wait_for_idle(timeout=120.0)
        results = scheduler.finalize_results() if idle else []
        if not idle:
            self.logger.warning("STAGE2 drain_timeout pending_results_not_written=true")
        for result in results:
            self._write_final_stage2_event(result)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.logger.info(
            "STAGE2 drain_ms=%.1f completed=%d idle=%s",
            elapsed,
            len(results),
            idle,
        )

    def _write_final_stage2_event(self, result: FinalStage2Result) -> None:
        if (
            self._last_event_frame is None
            or self._last_processed_index is None
            or self._last_source_label is None
        ):
            return
        debug = self._track_debug.get(
            result.resolution.track_id,
            YoloDebugInfo(
                raw_class_id=-1,
                raw_class_name=result.detection.major_class,
            ),
        )
        region = DetectedRegion(handoff=result.detection, debug=debug)
        event = self._build_event(
            region=region,
            resolution=result.resolution,
            frame=self._last_event_frame,
            processed_index=self._last_processed_index,
            source_label=self._last_source_label,
            source_time_ms=self._last_source_time_ms,
            yolo_ms=0.0,
        )
        self.journal.write(event)
        self.state.record_event(event)

    def _save_crop(self, region: DetectedRegion) -> str | None:
        detection = region.handoff
        if not self.config.save_crops or detection.major_class == "b0":
            return None
        filename = detection.detection_id.replace(":", "_") + ".jpg"
        relative = Path("crops") / detection.major_class / filename
        destination = self.session_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), detection.crop):
            self.logger.warning("Could not save crop: %s", destination)
            return None
        return str(relative)

    def _build_event(
        self,
        region: DetectedRegion,
        resolution: Stage2Resolution,
        frame: np.ndarray,
        processed_index: int,
        source_label: str,
        source_time_ms: float | None,
        yolo_ms: float,
    ) -> dict[str, Any]:
        detection = region.handoff
        decision = resolution.decision
        stage2_ms = resolution.inference_ms
        return {
            "schema": "recognition_event.v1",
            "id": detection.detection_id,
            "timestamp": _timestamp(),
            "frame": {
                "index": detection.frame_index,
                "processed_index": processed_index,
                "source": source_label,
                "source_time_ms": source_time_ms,
                "image_shape": [int(frame.shape[0]), int(frame.shape[1])],
            },
            "stage1": detection.handoff_dict(
                crop_path=resolution.source_crop_path,
            ),
            "route": {
                "module": decision.module,
                "execution": resolution.mode,
                "track_id": resolution.track_id,
                "cache_age_frames": resolution.cache_age_frames,
                "stage2_source_detection_id": resolution.source_detection_id,
                "input_contract": [
                    "id",
                    "major_class",
                    "confidence",
                    "bbox_xyxy",
                    "crop",
                ],
            },
            "stage2": decision.to_dict(),
            "timing_ms": {
                "yolo_frame": round(float(yolo_ms), 3),
                "stage2": round(float(stage2_ms), 3),
                "detection_total": round(float(yolo_ms + stage2_ms), 3),
            },
            "debug": region.debug.to_dict(),
        }

    def _draw_result(
        self,
        image: np.ndarray,
        region: DetectedRegion,
        decision: Stage2Decision,
    ) -> None:
        detection = region.handoff
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = COLORS[detection.major_class]
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 2)

        if decision.status in {"matched", "passthrough"}:
            result = decision.predicted_class or detection.major_class
        else:
            result = decision.status
        label = f"{detection.major_class}>{result}"
        if decision.score is not None:
            label += f" {decision.score:.2f}"
        if self.config.show_raw_label:
            label += f" [{region.debug.raw_class_name}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        image_height, image_width = image.shape[:2]
        left = min(max(x1, 0), max(image_width - text_width - 10, 0))
        bottom = max(text_height + baseline + 7, y1)
        bottom = min(bottom, image_height - 1)
        top = max(0, bottom - text_height - baseline - 7)
        right = min(image_width - 1, left + text_width + 10)
        cv2.rectangle(image, (left, top), (right, bottom), color, -1)
        cv2.putText(
            image,
            label,
            (left + 5, max(text_height + 1, bottom - baseline - 4)),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _write_manifest(self) -> None:
        payload = {
            "schema": "recognition_session.v1",
            "session_id": self.session_dir.name,
            "created_at": self.state.snapshot()["started_at"],
            "source": self.config.source,
            "session_dir": str(self.session_dir),
            "config": self.config.to_manifest(),
            "models": self._model_metadata,
            "data_flow": [
                {
                    "from": "source",
                    "to": "yolo",
                    "payload": "BGR frame",
                },
                {
                    "from": "yolo",
                    "to": "router",
                    "payload": [
                        "id",
                        "major_class",
                        "confidence",
                        "bbox_xyxy",
                        "crop",
                    ],
                    "excluded": ["raw_yolo_class", "raw_yolo_class_id"],
                },
                {
                    "condition": "major_class == f",
                    "from": "router",
                    "to": "InsightFace face recognition",
                },
                {
                    "condition": "major_class == k",
                    "from": "router",
                    "to": "PaddleClas knife recognition",
                },
                {
                    "condition": "major_class == b0",
                    "from": "router",
                    "to": "passthrough",
                },
            ],
            "outputs": {
                "events": str(self.session_dir / "events.jsonl"),
                "summary": str(self.session_dir / "summary.json"),
                "annotated_video": str(self.session_dir / "annotated.mp4"),
                "crops": str(self.session_dir / "crops"),
            },
        }
        write_json(self.session_dir / "manifest.json", payload)

    def _write_summary(self) -> None:
        snapshot = self.state.snapshot()
        write_json(
            self.session_dir / "summary.json",
            {
                "schema": "recognition_session_summary.v1",
                "session_id": self.session_dir.name,
                "status": snapshot["status"],
                "source": snapshot["source"],
                "started_at": snapshot["started_at"],
                "ended_at": snapshot["ended_at"],
                "error": snapshot["error"],
                "processed_frames": snapshot["processed_frames"],
                "major_counts": snapshot["major_counts"],
                "status_counts": snapshot["status_counts"],
                "media_outputs": self._media_outputs,
                "events_jsonl": str(self.session_dir / "events.jsonl"),
            },
        )
        self.logger.info(
            "END status=%s frames=%d output=%s",
            snapshot["status"],
            snapshot["processed_frames"],
            self.session_dir,
        )
