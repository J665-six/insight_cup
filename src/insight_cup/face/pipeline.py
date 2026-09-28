"""Consume stage-1 YOLO f-crops and emit stage-2 face identities."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from .engine import InsightFaceEmbeddingEngine
from .gallery import FaceGallery
from .service import FaceRecognitionService


ProgressCallback = Callable[[int, int], None]


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _resolve_base_dir(handoff_path: Path, payload: dict[str, Any]) -> Path:
    raw_base = Path(str(payload.get("base_dir", handoff_path.parent))).expanduser()
    if not raw_base.is_absolute():
        raw_base = handoff_path.parent / raw_base
    return raw_base.resolve()


def _count_f_detections(payload: dict[str, Any]) -> int:
    return sum(
        1
        for frame in payload.get("frames", [])
        for detection in frame.get("detections", [])
        if detection.get("major_class") == "f"
    )


def process_handoff(
    handoff_path: str | Path,
    gallery: FaceGallery,
    engine: InsightFaceEmbeddingEngine,
    output_path: str | Path,
    threshold: float = 0.40,
    min_margin: float = 0.03,
    max_detections: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Recognize only major-class f detections from a stage-1 handoff."""

    source_path = Path(handoff_path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "stage1_yolo_handoff.v1":
        raise ValueError(f"Unsupported stage-1 schema: {payload.get('schema')!r}")

    base_dir = _resolve_base_dir(source_path, payload)
    total_available = _count_f_detections(payload)
    target_total = (
        min(total_available, max_detections)
        if max_detections is not None
        else total_available
    )
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    service = FaceRecognitionService(
        engine=engine,
        gallery=gallery,
        threshold=threshold,
        min_margin=min_margin,
    )

    for frame in payload.get("frames", []):
        frame_index = int(frame.get("frame_index", -1))
        for detection in frame.get("detections", []):
            if detection.get("major_class") != "f":
                continue
            if max_detections is not None and len(records) >= max_detections:
                break

            crop_value = detection.get("crop_path")
            record: dict[str, Any] = {
                "id": detection.get("id"),
                "frame_index": frame_index,
                "major_class": "f",
                "source_bbox_xyxy": detection.get("bbox_xyxy"),
                "source_crop_path": crop_value,
                "status": None,
                "predicted_class": None,
                "candidate_class": None,
                "similarity": None,
                "second_best_class": None,
                "second_best_similarity": None,
                "margin": None,
                "face_detection_confidence": None,
                "face_bbox_xyxy_in_crop": None,
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
                        if recognition.status == "no_face":
                            record["status"] = "no_face"
                        else:
                            record.update(
                                {
                                    "status": recognition.status,
                                    "predicted_class": recognition.predicted_class,
                                    "candidate_class": recognition.candidate_class,
                                    "similarity": _round_optional(
                                        recognition.similarity
                                    ),
                                    "second_best_class": (
                                        recognition.second_best_class
                                    ),
                                    "second_best_similarity": _round_optional(
                                        recognition.second_best_similarity
                                    ),
                                    "margin": _round_optional(recognition.margin),
                                    "face_detection_confidence": _round_optional(
                                        recognition.face_detection_confidence
                                    ),
                                    "face_bbox_xyxy_in_crop": [
                                        round(value, 2)
                                        for value in recognition.face_bbox_xyxy
                                    ],
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
        "schema": "stage2_face_recognition.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_handoff": str(source_path),
        "gallery_identities": gallery.identities,
        "model": engine.metadata,
        "matching": {"threshold": threshold, "min_margin": min_margin},
        "detections": records,
        "summary": {
            "available_f_detections": total_available,
            "processed_f_detections": len(records),
            "status_counts": dict(status_counts.most_common()),
            "identity_counts": dict(identity_counts.most_common()),
        },
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    temp_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(destination)
    return output
