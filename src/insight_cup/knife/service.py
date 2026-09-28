"""Project-facing API for recognizing one YOLO k-class crop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .engine import KnifePrediction


# Runtime default balances coverage against false accepts. The margin check
# remains the guard against two nearly tied candidates.
DEFAULT_KNIFE_THRESHOLD = 0.50
DEFAULT_KNIFE_MIN_MARGIN = 0.15


class KnifeEngine(Protocol):
    def predict(self, image: np.ndarray) -> KnifePrediction: ...


@dataclass(frozen=True)
class KnifeRecognitionResult:
    status: str
    predicted_class: str | None = None
    candidate_class: str | None = None
    confidence: float | None = None
    second_best_class: str | None = None
    second_best_confidence: float | None = None
    margin: float | None = None
    score_kind: str = "softmax_confidence"
    class_scores: dict[str, float] = field(default_factory=dict)


class KnifeRecognitionService:
    """Convert one BGR k-crop into a k1...k10 decision."""

    def __init__(
        self,
        engine: KnifeEngine,
        threshold: float = DEFAULT_KNIFE_THRESHOLD,
        min_margin: float = DEFAULT_KNIFE_MIN_MARGIN,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between zero and one")
        self.engine = engine
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)

    def recognize(self, crop: np.ndarray) -> KnifeRecognitionResult:
        prediction = self.engine.predict(crop)
        if not prediction.labels:
            raise RuntimeError("Knife classifier returned no candidates")
        candidate = prediction.labels[0]
        confidence = prediction.scores[0]
        second_class = prediction.labels[1] if len(prediction.labels) > 1 else None
        second_confidence = prediction.scores[1] if len(prediction.scores) > 1 else None
        margin = (
            confidence - second_confidence
            if second_confidence is not None
            else None
        )

        if candidate == "other" or confidence < self.threshold:
            status = "unknown"
            predicted = None
        elif margin is not None and margin < self.min_margin:
            status = "ambiguous"
            predicted = None
        else:
            status = "matched"
            predicted = candidate
        return KnifeRecognitionResult(
            status=status,
            predicted_class=predicted,
            candidate_class=candidate,
            confidence=confidence,
            second_best_class=second_class,
            second_best_confidence=second_confidence,
            margin=margin,
            score_kind=prediction.score_kind,
            class_scores=(
                dict(prediction.class_scores)
                if prediction.class_scores
                else dict(zip(prediction.labels, prediction.scores))
            ),
        )
