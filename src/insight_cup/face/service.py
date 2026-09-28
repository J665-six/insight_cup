"""Project-facing API for recognizing one YOLO f-class crop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .engine import InsightFaceEmbeddingEngine
from .gallery import FaceGallery


@dataclass(frozen=True)
class RecognitionResult:
    status: str
    predicted_class: str | None = None
    candidate_class: str | None = None
    similarity: float | None = None
    second_best_class: str | None = None
    second_best_similarity: float | None = None
    margin: float | None = None
    face_detection_confidence: float | None = None
    face_bbox_xyxy: tuple[float, float, float, float] | None = None
    face_roll_degrees: float | None = None
    class_scores: dict[str, float] = field(default_factory=dict)
    attempted_rotation_degrees: tuple[float, ...] = ()
    selected_rotation_degrees: float = 0.0


class FaceRecognitionService:
    """Convert one BGR f-crop into an fN identity decision."""

    def __init__(
        self,
        engine: InsightFaceEmbeddingEngine,
        gallery: FaceGallery,
        threshold: float = 0.40,
        min_margin: float = 0.03,
        rotation_retry_degrees: float = 0.0,
    ) -> None:
        self.engine = engine
        self.gallery = gallery
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.rotation_retry_degrees = max(0.0, float(rotation_retry_degrees))

    def _recognize_once(self, crop: np.ndarray) -> RecognitionResult:
        face = self.engine.extract(crop)
        if face is None:
            return RecognitionResult(status="no_face")
        match = self.gallery.match(
            face.embedding,
            threshold=self.threshold,
            min_margin=self.min_margin,
        )
        return RecognitionResult(
            status=match.status,
            predicted_class=match.predicted_class,
            candidate_class=match.candidate_class,
            similarity=match.similarity,
            second_best_class=match.second_best_class,
            second_best_similarity=match.second_best_similarity,
            margin=match.margin,
            face_detection_confidence=face.detection_score,
            face_bbox_xyxy=face.bbox_xyxy,
            face_roll_degrees=face.roll_degrees,
            class_scores=match.class_scores,
        )

    @staticmethod
    def _rotate_expanded(image: np.ndarray, degrees: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width * 0.5, height * 0.5)
        matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        output_width = max(1, int(round(height * sine + width * cosine)))
        output_height = max(1, int(round(height * cosine + width * sine)))
        matrix[0, 2] += output_width * 0.5 - center[0]
        matrix[1, 2] += output_height * 0.5 - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _rank(result: RecognitionResult) -> tuple[int, float, float]:
        status_rank = {
            "matched": 3,
            "ambiguous": 2,
            "unknown": 1,
            "no_face": 0,
        }.get(result.status, -1)
        return (
            status_rank,
            result.similarity if result.similarity is not None else -1.0,
            (
                result.face_detection_confidence
                if result.face_detection_confidence is not None
                else -1.0
            ),
        )

    def recognize(self, crop: np.ndarray) -> RecognitionResult:
        initial = self._recognize_once(crop)
        attempts: list[tuple[float, RecognitionResult]] = [(0.0, initial)]
        if self.rotation_retry_degrees <= 0.0:
            return replace(initial, attempted_rotation_degrees=(0.0,))

        retry_angles: list[float] = []
        if initial.status == "no_face":
            retry_angles = [
                -self.rotation_retry_degrees,
                self.rotation_retry_degrees,
            ]
        elif initial.status in {"unknown", "ambiguous"} or (
            initial.face_detection_confidence is not None
            and initial.face_detection_confidence < 0.45
        ) or (
            initial.similarity is not None
            and initial.similarity < self.threshold + 0.05
        ):
            if (
                initial.face_roll_degrees is not None
                and abs(initial.face_roll_degrees) >= 8.0
            ):
                retry_angles = [
                    max(
                        -self.rotation_retry_degrees,
                        min(
                            self.rotation_retry_degrees,
                            initial.face_roll_degrees,
                        ),
                    )
                ]

        for degrees in retry_angles:
            rotated = self._rotate_expanded(crop, degrees)
            attempts.append((degrees, self._recognize_once(rotated)))

        selected_degrees, selected = max(
            attempts,
            key=lambda item: self._rank(item[1]),
        )
        return replace(
            selected,
            attempted_rotation_degrees=tuple(item[0] for item in attempts),
            selected_rotation_degrees=selected_degrees,
        )
