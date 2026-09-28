"""YOLO adapter that collapses fN/kN labels into the three public classes."""

from __future__ import annotations

import ast
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from insight_cup.common.crops import KNIFE_CROP_CONTEXT_SCALE, square_context_crop
from insight_cup.common.onnx import cpu_session_options

from .contracts import (
    DetectedRegion,
    DetectionFrame,
    Stage1Detection,
    YoloDebugInfo,
)


LOGGER = logging.getLogger("recognition.detector")


def collapse_yolo_class(raw_name: str) -> str | None:
    name = raw_name.strip().lower()
    if name == "b0":
        return "b0"
    if re.fullmatch(r"f\d+", name):
        return "f"
    if re.fullmatch(r"k\d+", name):
        return "k"
    return None


def _names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names or [])}


def _clamp_box(
    xyxy: list[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    x1 = min(max(x1, 0), max(width - 1, 0))
    y1 = min(max(y1, 0), max(height - 1, 0))
    x2 = min(max(x2, 0), width)
    y2 = min(max(y2, 0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def suppress_major_class_duplicates(
    regions: list[DetectedRegion],
    iou_threshold: float,
) -> list[DetectedRegion]:
    """Apply a second NMS after raw fN/kN classes become public classes."""

    kept: list[DetectedRegion] = []
    for region in sorted(
        regions,
        key=lambda item: item.handoff.confidence,
        reverse=True,
    ):
        duplicate = any(
            existing.handoff.major_class == region.handoff.major_class
            and _bbox_iou(
                existing.handoff.bbox_xyxy,
                region.handoff.bbox_xyxy,
            )
            >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(region)
    return kept


class YoloMajorClassDetector:
    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = "auto",
        cpu_threads: int = 1,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.device = device
        self.cpu_threads = max(1, int(cpu_threads))
        self._backend = "onnxruntime" if self.weights.suffix.lower() == ".onnx" else "ultralytics"
        self._model: Any | None = None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None

        if self._backend == "onnxruntime":
            if self.device not in {"auto", "cpu"}:
                raise ValueError(
                    "The direct ONNX YOLO backend currently supports CPU only; "
                    "use .pt weights for CUDA."
                )
            import onnxruntime as ort

            options = cpu_session_options(ort, self.cpu_threads)
            self._session = ort.InferenceSession(
                str(self.weights),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("YOLO ONNX model must have one input and one output")
            input_shape = inputs[0].shape
            if (
                len(input_shape) != 4
                or input_shape[0] != 1
                or input_shape[1] != 3
                or input_shape[2] != self.image_size
                or input_shape[3] != self.image_size
            ):
                raise RuntimeError(
                    "YOLO ONNX input shape does not match --imgsz: "
                    f"{input_shape} vs {self.image_size}"
                )
            metadata = self._session.get_modelmeta().custom_metadata_map
            raw_names = ast.literal_eval(metadata.get("names", "{}"))
            self.names = _names_map(raw_names)
            if not self.names:
                raise RuntimeError("YOLO ONNX metadata does not contain class names")
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
        else:
            import torch
            from ultralytics import YOLO

            torch.set_num_threads(self.cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            self._model = YOLO(str(self.weights))
            self.names = _names_map(self._model.names)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "weights": str(self.weights),
            "raw_classes": self.names,
            "public_classes": ["b0", "f", "k"],
            "confidence": self.confidence,
            "iou": self.iou,
            "major_class_nms_iou": self.iou,
            "image_size": self.image_size,
            "device": self.device,
            "cpu_threads": self.cpu_threads,
        }

    def _onnx_predict(
        self,
        frame: np.ndarray,
    ) -> list[tuple[list[float], float, int]]:
        if (
            self._session is None
            or self._input_name is None
            or self._output_name is None
        ):
            raise RuntimeError("YOLO ONNX session is not initialized")

        height, width = frame.shape[:2]
        scale = min(self.image_size / height, self.image_size / width)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        horizontal_padding = (self.image_size - resized_width) / 2.0
        vertical_padding = (self.image_size - resized_height) / 2.0
        left = round(horizontal_padding - 0.1)
        right = round(horizontal_padding + 0.1)
        top = round(vertical_padding - 0.1)
        bottom = round(vertical_padding + 0.1)
        prepared = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(
            prepared[:, :, ::-1].transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        tensor /= 255.0
        output = np.asarray(
            self._session.run(
                [self._output_name],
                {self._input_name: tensor},
            )[0],
            dtype=np.float32,
        )
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError(f"Unexpected YOLO ONNX output shape: {output.shape}")
        predictions = output[0].T
        expected_columns = 4 + len(self.names)
        if predictions.shape[1] != expected_columns:
            raise RuntimeError(
                "YOLO ONNX class count does not match metadata: "
                f"{predictions.shape[1]} vs {expected_columns}"
            )

        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        selected = confidences >= self.confidence
        boxes_xywh = predictions[selected, :4]
        confidences = confidences[selected]
        class_ids = class_ids[selected]
        if boxes_xywh.size == 0:
            return []

        boxes_xyxy = np.empty_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        kept_indexes: list[int] = []
        for class_id in np.unique(class_ids):
            indexes = np.flatnonzero(class_ids == class_id)
            class_boxes = boxes_xyxy[indexes]
            nms_boxes = [
                [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1]),
                ]
                for box in class_boxes
            ]
            relative_kept = cv2.dnn.NMSBoxes(
                nms_boxes,
                confidences[indexes].tolist(),
                self.confidence,
                self.iou,
            )
            kept_indexes.extend(indexes[int(index)] for index in relative_kept)

        detections: list[tuple[list[float], float, int]] = []
        for index in sorted(
            kept_indexes,
            key=lambda item: float(confidences[item]),
            reverse=True,
        )[:300]:
            box = boxes_xyxy[index].copy()
            box[[0, 2]] = (box[[0, 2]] - left) / scale
            box[[1, 3]] = (box[[1, 3]] - top) / scale
            detections.append(
                (
                    box.tolist(),
                    float(confidences[index]),
                    int(class_ids[index]),
                )
            )
        return detections

    def detect(self, frame: np.ndarray, frame_index: int) -> DetectionFrame:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image")
        started = time.perf_counter()
        if self._backend == "onnxruntime":
            values = self._onnx_predict(frame)
        else:
            if self._model is None:
                raise RuntimeError("Ultralytics YOLO model is not initialized")
            kwargs: dict[str, Any] = {
                "conf": self.confidence,
                "iou": self.iou,
                "imgsz": self.image_size,
                "verbose": False,
            }
            if self.device != "auto":
                kwargs["device"] = self.device
            result = self._model.predict(frame, **kwargs)[0]
            boxes = result.boxes
            values = (
                []
                if boxes is None or len(boxes) == 0
                else list(
                    zip(
                        boxes.xyxy.cpu().tolist(),
                        boxes.conf.cpu().tolist(),
                        boxes.cls.cpu().tolist(),
                    )
                )
            )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not values:
            return DetectionFrame(regions=(), inference_ms=inference_ms)

        height, width = frame.shape[:2]
        regions: list[DetectedRegion] = []
        ignored: list[str] = []
        for detection_index, (xyxy, confidence, raw_class_id) in enumerate(values):
            class_id = int(raw_class_id)
            raw_name = self.names.get(class_id, str(class_id))
            major = collapse_yolo_class(raw_name)
            if major is None:
                ignored.append(raw_name)
                continue
            bbox = _clamp_box(xyxy, width, height)
            if bbox is None:
                LOGGER.warning(
                    "DROP frame=%d raw_class=%s reason=invalid_bbox",
                    frame_index,
                    raw_name,
                )
                continue
            x1, y1, x2, y2 = bbox
            if major == "k":
                crop = square_context_crop(
                    frame,
                    bbox,
                    context_scale=KNIFE_CROP_CONTEXT_SCALE,
                )
            else:
                crop = frame[y1:y2, x1:x2].copy()
            handoff = Stage1Detection(
                detection_id=f"frame{frame_index:06d}:det{detection_index:03d}",
                frame_index=frame_index,
                major_class=major,  # type: ignore[arg-type]
                confidence=float(confidence),
                bbox_xyxy=bbox,
                crop=crop,
            )
            regions.append(
                DetectedRegion(
                    handoff=handoff,
                    debug=YoloDebugInfo(class_id, raw_name),
                )
            )
        regions = suppress_major_class_duplicates(regions, self.iou)
        return DetectionFrame(
            regions=tuple(regions),
            inference_ms=inference_ms,
            ignored_classes=tuple(ignored),
        )
