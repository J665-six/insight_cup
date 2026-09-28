"""Consume stage-1 YOLO k-crops and emit stage-2 knife identities."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from .engine import PaddleClasONNXEngine
from .service import (
    DEFAULT_KNIFE_MIN_MARGIN,
    DEFAULT_KNIFE_THRESHOLD,
    KnifeRecognitionService,
)


ProgressCallback = Callable[[int, int], None]


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _resolve_base_dir(handoff_path: Path, payload: dict[str, Any]) -> Path:
    raw_base = Path(str(payload.get("base_dir", handoff_path.parent))).expanduser()
    if not raw_base.is_absolute():
        raw_base = handoff_path.parent / raw_base
    return raw_base.resolve()


def _count_k_detections(payload: dict[str, Any]) -> int:
    return sum(
        1
        for frame in payload.get("frames", [])
        for detection in frame.get("detections", [])
        if detection.get("major_class") == "k"
    )


def process_handoff(
    handoff_path: str | Path,
    engine: PaddleClasONNXEngine,
    output_path: str | Path,
    threshold: float = DEFAULT_KNIFE_THRESHOLD,
    min_margin: float = DEFAULT_KNIFE_MIN_MARGIN,
    max_detections: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Recognize only major-class k detections from a stage-1 handoff."""

    source_path = Path(handoff_path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage1_yolo_handoff.v1":
        raise ValueError(f"Unsupported stage-1 schema: {payload.get('schema')!r}")

    base_dir = _resolve_base_dir(source_path, payload)
    total_available = _count_k_detections(payload)
    target_total = (
        min(total_available, max_detections)
        if max_detections is not None
        else total_available
    )
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    service = KnifeRecognitionService(
        engine=engine,
        threshold=threshold,
        min_margin=min_margin,
    )

    for frame in payload.get("frames", []):
        frame_index = int(frame.get("frame_index", -1))
        for detection in frame.get("detections", []):
            if detection.get("major_class") != "k":
                continue
            if max_detections is not None and len(records) >= max_detections:
                break

            crop_value = detection.get("crop_path")
            record: dict[str, Any] = {
                "id": detection.get("id"),
                "frame_index": frame_index,
                "major_class": "k",
                "source_confidence": detection.get("confidence"),
                "source_bbox_xyxy": detection.get("bbox_xyxy"),
                "source_crop_path": crop_value,
                "status": None,
                "predicted_class": None,
                "candidate_class": None,
                "confidence": None,
                "second_best_class": None,
                "second_best_confidence": None,
                "margin": None,
                "score_kind": None,
            }
            if not crop_value:
                record["status"] = "missing_crop"
            else:
                crop_path = Path(str(crop_value)).expanduser()
                if not crop_path.is_absolute():
                    crop_path = base_dir / crop_path
                if not crop_path.is_file():
                    record["status"] = "missing_crop"
                else:
                    image = cv2.imread(str(crop_path))
                    if image is None:
                        record["status"] = "read_error"
                    else:
                        recognition = service.recognize(image)
                        record.update(
                            {
                                "status": recognition.status,
                                "predicted_class": recognition.predicted_class,
                                "candidate_class": recognition.candidate_class,
                                "confidence": _round_optional(recognition.confidence),
                                "second_best_class": recognition.second_best_class,
                                "second_best_confidence": _round_optional(
                                    recognition.second_best_confidence
                                ),
                                "margin": _round_optional(recognition.margin),
                                "score_kind": recognition.score_kind,
                            }
                        )

            status_counts[str(record["status"])] += 1
            records.append(record)
            if progress:
                progress(len(records), target_total)
        if max_detections is not None and len(records) >= max_detections:
            break

    identity_counts = Counter(
        str(record["predicted_class"])
        for record in records
        if record["predicted_class"] is not None
    )
    output = {
        "schema": "stage2_knife_recognition.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_handoff": str(source_path),
        "model": engine.metadata,
        "decision": {"threshold": threshold, "min_margin": min_margin},
        "detections": records,
        "summary": {
            "available_k_detections": total_available,
            "processed_k_detections": len(records),
            "status_counts": dict(status_counts.most_common()),
            "identity_counts": dict(identity_counts.most_common()),
        },
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    temp_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(destination)
    return output
