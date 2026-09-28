"""Route clean stage-1 detections to their dedicated stage-2 services."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .contracts import Stage1Detection, Stage2Decision


LOGGER = logging.getLogger("recognition.router")


class RecognitionService(Protocol):
    def recognize(self, crop: Any) -> Any: ...


class RecognitionRouter:
    def __init__(
        self,
        face_service: RecognitionService,
        knife_service: RecognitionService,
    ) -> None:
        self.face_service = face_service
        self.knife_service = knife_service

    @staticmethod
    def target_module(major_class: str) -> str:
        return {"b0": "passthrough", "f": "face", "k": "knife"}[major_class]

    @staticmethod
    def _class_scores(result: Any, score_name: str) -> dict[str, float]:
        scores = getattr(result, "class_scores", None)
        if scores:
            return {str(label): float(score) for label, score in scores.items()}
        fallback: dict[str, float] = {}
        candidate = getattr(result, "candidate_class", None)
        score = getattr(result, score_name, None)
        if candidate is not None and score is not None:
            fallback[str(candidate)] = float(score)
        second = getattr(result, "second_best_class", None)
        second_score = getattr(result, f"second_best_{score_name}", None)
        if second is not None and second_score is not None:
            fallback[str(second)] = float(second_score)
        return fallback

    def route(self, detection: Stage1Detection) -> Stage2Decision:
        try:
            if detection.major_class == "b0":
                return Stage2Decision(
                    module="passthrough",
                    status="passthrough",
                    predicted_class="b0",
                    candidate_class="b0",
                    score=detection.confidence,
                    score_kind="yolo_confidence",
                )
            if detection.major_class == "f":
                result = self.face_service.recognize(detection.crop)
                return Stage2Decision(
                    module="face",
                    status=result.status,
                    predicted_class=result.predicted_class,
                    candidate_class=result.candidate_class,
                    score=result.similarity,
                    score_kind="cosine_similarity",
                    second_best_class=result.second_best_class,
                    second_best_score=result.second_best_similarity,
                    margin=result.margin,
                    class_scores=self._class_scores(result, "similarity"),
                    metadata={
                        "face_detection_confidence": result.face_detection_confidence,
                        "face_roll_degrees": getattr(
                            result, "face_roll_degrees", None
                        ),
                        "face_bbox_xyxy_in_crop": (
                            list(result.face_bbox_xyxy)
                            if result.face_bbox_xyxy is not None
                            else None
                        ),
                        "face_rotation_attempts_degrees": list(
                            getattr(result, "attempted_rotation_degrees", ())
                        ),
                        "face_selected_rotation_degrees": (
                            getattr(result, "selected_rotation_degrees", 0.0)
                        ),
                    },
                )

            result = self.knife_service.recognize(detection.crop)
            return Stage2Decision(
                module="knife",
                status=result.status,
                predicted_class=result.predicted_class,
                candidate_class=result.candidate_class,
                score=result.confidence,
                score_kind=getattr(result, "score_kind", "softmax_confidence"),
                second_best_class=result.second_best_class,
                second_best_score=result.second_best_confidence,
                margin=result.margin,
                class_scores=self._class_scores(result, "confidence"),
            )
        except Exception as exc:
            module = self.target_module(detection.major_class)
            LOGGER.exception(
                "Stage-2 failure: id=%s module=%s",
                detection.detection_id,
                module,
            )
            return Stage2Decision(
                module=module,  # type: ignore[arg-type]
                status="error",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
