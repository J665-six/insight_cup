"""Explicit in-process contracts between detection and recognition stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


MajorClass = Literal["b0", "f", "k"]


@dataclass(frozen=True)
class Stage1Detection:
    """The only payload accepted by stage 2.

    Raw YOLO subclasses intentionally do not belong to this contract.
    """

    detection_id: str
    frame_index: int
    major_class: MajorClass
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    crop: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.major_class not in {"b0", "f", "k"}:
            raise ValueError(f"Unsupported major class: {self.major_class}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        if self.crop is None or self.crop.ndim != 3 or self.crop.shape[2] != 3:
            raise ValueError("crop must be a non-empty BGR image")
        if min(self.crop.shape[:2]) <= 0:
            raise ValueError("crop must have positive dimensions")

    def handoff_dict(self, crop_path: str | None = None) -> dict[str, Any]:
        return {
            "id": self.detection_id,
            "major_class": self.major_class,
            "confidence": round(float(self.confidence), 6),
            "bbox_xyxy": list(self.bbox_xyxy),
            "crop_path": crop_path,
        }


@dataclass(frozen=True)
class YoloDebugInfo:
    raw_class_id: int
    raw_class_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_yolo_class_id": self.raw_class_id,
            "raw_yolo_class": self.raw_class_name,
        }


@dataclass(frozen=True)
class DetectedRegion:
    handoff: Stage1Detection
    debug: YoloDebugInfo


@dataclass(frozen=True)
class DetectionFrame:
    regions: tuple[DetectedRegion, ...]
    inference_ms: float
    ignored_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage2Decision:
    module: Literal["passthrough", "face", "knife"]
    status: str
    predicted_class: str | None = None
    candidate_class: str | None = None
    score: float | None = None
    score_kind: str | None = None
    second_best_class: str | None = None
    second_best_score: float | None = None
    margin: float | None = None
    class_scores: dict[str, float] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "predicted_class": self.predicted_class,
            "candidate_class": self.candidate_class,
            "score": self.score,
            "score_kind": self.score_kind,
            "second_best_class": self.second_best_class,
            "second_best_score": self.second_best_score,
            "margin": self.margin,
        }
        payload.update(self.metadata)
        return payload
