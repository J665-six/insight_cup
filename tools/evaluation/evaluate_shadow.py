#!/usr/bin/env python3
"""Evaluate same-source recognition shadow records against human labels."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHADOW_SCHEMA = "recognition_shadow.v1"
GOLDEN_SCHEMA = "recognition_golden.v1"
OUTPUT_SCHEMA = "recognition_shadow_evaluation.v1"
UNKNOWN = "unknown"
YOLO_CLASSES = (
    *(f"k{index}" for index in range(1, 11)),
    *(f"f{index}" for index in range(1, 11)),
    "b0",
)
YOLO_CLASS_IDS = {label: index for index, label in enumerate(YOLO_CLASSES)}
STAGE2_CLASSES = {
    major: frozenset(f"{major}{index}" for index in range(1, 11))
    for major in ("f", "k")
}
STAGE2_SCORE_KINDS = {
    "f": frozenset(("cosine_similarity",)),
    "k": frozenset(("cosine_similarity", "softmax_confidence")),
}
GOLDEN_CLASSES = frozenset((*YOLO_CLASSES, UNKNOWN))
RAW_STATUSES = {
    "f": frozenset(("matched", "unknown", "ambiguous", "no_face", "error")),
    "k": frozenset(("matched", "unknown", "ambiguous", "error")),
}
FINAL_STATUSES = {
    major: frozenset((*statuses, "confirming"))
    for major, statuses in RAW_STATUSES.items()
}
ACCEPTED_STATUSES = frozenset(("matched",))
REJECTION_STATUSES = frozenset(("unknown", "ambiguous", "no_face", "confirming"))
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SCORE_ABS_TOLERANCE = 1e-6
STAGE2_CORE_FIELDS = (
    "status",
    "candidate_class",
    "predicted_class",
    "score",
    "score_kind",
    "second_best_class",
    "second_best_score",
    "margin",
    "class_scores",
)
TEMPORAL_RAW_FIELDS = (
    "status",
    "predicted_class",
    "candidate_class",
    "score",
    "second_best_class",
    "second_best_score",
    "margin",
)
TEMPORAL_FIELDS = (
    "method",
    "window_size",
    "samples",
    "switch_confirmations",
    "stable_class",
    "proposed_class",
    "pending_class",
    "pending_count",
    "raw",
)


class EvaluationInputError(ValueError):
    """Raised when an input cannot be evaluated without ambiguous results."""


@dataclass(frozen=True)
class ShadowLog:
    input: dict[str, Any]
    records: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class GoldenSet:
    input: dict[str, Any]
    items: dict[str, dict[str, Any]]


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationInputError(f"{location} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any], fields: Iterable[str], location: str
) -> None:
    missing = sorted(set(fields) - value.keys())
    if missing:
        raise EvaluationInputError(
            f"{location} is missing required field(s): {', '.join(missing)}"
        )


def _require_exact_fields(
    value: Mapping[str, Any], fields: Iterable[str], location: str
) -> None:
    expected = set(fields)
    _require_fields(value, expected, location)
    unexpected = sorted(value.keys() - expected)
    if unexpected:
        raise EvaluationInputError(
            f"{location} has unexpected field(s): {', '.join(unexpected)}"
        )


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"{location} must be a non-empty string")
    return value


def _require_optional_class(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, location)


def _require_optional_number(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(float(value)):
        raise EvaluationInputError(f"{location} must be null or a finite number")
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bbox(value: Any, location: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EvaluationInputError(f"{location} must contain four numbers")
    if not all(_is_number(item) and math.isfinite(float(item)) for item in value):
        raise EvaluationInputError(f"{location} must contain four finite numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise EvaluationInputError(f"{location} must have positive width and height")
    return x1, y1, x2, y2


def _validate_scores(value: Any, location: str) -> Mapping[str, Any]:
    scores = _require_mapping(value, location)
    for label, score in scores.items():
        _require_nonempty_string(label, f"{location} key")
        if not _is_number(score) or not math.isfinite(float(score)):
            raise EvaluationInputError(f"{location}.{label} must be a finite number")
    return scores


def _validate_input(value: Any, location: str) -> dict[str, Any]:
    provenance = _require_mapping(value, location)
    _require_fields(provenance, ("kind", "video_stride"), location)
    kind = _require_nonempty_string(provenance["kind"], f"{location}.kind")
    if kind == "file":
        _require_exact_fields(
            provenance, ("kind", "sha256", "video_stride"), location
        )
        sha256 = _require_nonempty_string(
            provenance["sha256"], f"{location}.sha256"
        )
        if SHA256_PATTERN.fullmatch(sha256) is None:
            raise EvaluationInputError(
                f"{location}.sha256 must be 64 lowercase hexadecimal characters"
            )
    elif kind == "capture":
        _require_exact_fields(
            provenance, ("kind", "capture_id", "video_stride"), location
        )
        _require_nonempty_string(
            provenance["capture_id"], f"{location}.capture_id"
        )
    else:
        raise EvaluationInputError(f"{location}.kind must be file or capture")

    stride = provenance["video_stride"]
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise EvaluationInputError(
            f"{location}.video_stride must be a positive integer"
        )
    return dict(provenance)


def _validate_score_domain(
    scores: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    location: str,
    *,
    allow_empty: bool = False,
) -> None:
    actual = set(scores)
    if allow_empty and not actual:
        return
    if actual == set(expected):
        return
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {', '.join(unexpected)}")
    raise EvaluationInputError(
        f"{location} must contain the exact fixed class set ({'; '.join(details)})"
    )


def _scores_equal(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-6,
        abs_tol=SCORE_ABS_TOLERANCE,
    )


def _validate_stage2_evidence(
    value: Any,
    location: str,
    expected_classes: frozenset[str],
    legal_statuses: frozenset[str],
    *,
    candidate_must_be_max: bool,
) -> None:
    section = _require_mapping(value, location)
    _require_fields(
        section,
        (
            "status",
            "candidate_class",
            "predicted_class",
            "score",
            "score_kind",
            "second_best_class",
            "second_best_score",
            "margin",
            "class_scores",
        ),
        location,
    )
    status = _require_nonempty_string(section["status"], f"{location}.status")
    if status not in legal_statuses:
        raise EvaluationInputError(
            f"{location}.status must be one of {', '.join(sorted(legal_statuses))}"
        )

    candidate = _require_optional_class(
        section["candidate_class"], f"{location}.candidate_class"
    )
    predicted = _require_optional_class(
        section["predicted_class"], f"{location}.predicted_class"
    )
    second_best = _require_optional_class(
        section["second_best_class"], f"{location}.second_best_class"
    )
    score = _require_optional_number(section["score"], f"{location}.score")
    second_best_score = _require_optional_number(
        section["second_best_score"], f"{location}.second_best_score"
    )
    margin = _require_optional_number(section["margin"], f"{location}.margin")
    score_kind = _require_optional_class(
        section["score_kind"], f"{location}.score_kind"
    )

    for field, label in (
        ("candidate_class", candidate),
        ("predicted_class", predicted),
        ("second_best_class", second_best),
    ):
        if label is not None and label not in expected_classes:
            raise EvaluationInputError(
                f"{location}.{field} is outside the fixed stage-2 class set"
            )

    scores = _validate_scores(section["class_scores"], f"{location}.class_scores")
    _validate_score_domain(
        scores,
        expected_classes,
        f"{location}.class_scores",
        allow_empty=True,
    )

    if status == "matched":
        if predicted is None or predicted != candidate:
            raise EvaluationInputError(
                f"{location} matched status requires equal non-null "
                "candidate_class and predicted_class"
            )
    elif predicted is not None:
        raise EvaluationInputError(
            f"{location} status {status!r} requires predicted_class to be null"
        )

    empty_status = status in {"no_face", "error"}
    if not scores:
        if not empty_status:
            raise EvaluationInputError(
                f"{location} status {status!r} requires a populated class_scores map"
            )
        evidence = {
            "candidate_class": candidate,
            "score": score,
            "score_kind": score_kind,
            "second_best_class": second_best,
            "second_best_score": second_best_score,
            "margin": margin,
        }
        populated = sorted(field for field, item in evidence.items() if item is not None)
        if populated:
            raise EvaluationInputError(
                f"{location} has empty class_scores but non-null evidence: "
                f"{', '.join(populated)}"
            )
        return
    if empty_status:
        raise EvaluationInputError(
            f"{location} status {status!r} requires an empty class_scores map"
        )

    required_evidence = {
        "candidate_class": candidate,
        "score": score,
        "score_kind": score_kind,
        "second_best_class": second_best,
        "second_best_score": second_best_score,
        "margin": margin,
    }
    missing = sorted(field for field, item in required_evidence.items() if item is None)
    if missing:
        raise EvaluationInputError(
            f"{location} has class_scores but null evidence: {', '.join(missing)}"
        )
    assert candidate is not None
    assert score is not None
    assert second_best is not None
    assert second_best_score is not None
    assert margin is not None

    candidate_score = float(scores[candidate])
    if not _scores_equal(score, candidate_score):
        raise EvaluationInputError(
            f"{location}.score does not match candidate_class in class_scores"
        )
    if candidate_must_be_max and not _scores_equal(
        candidate_score, max(float(item) for item in scores.values())
    ):
        raise EvaluationInputError(
            f"{location}.candidate_class is not a maximum-scoring class"
        )
    if second_best == candidate:
        raise EvaluationInputError(
            f"{location}.second_best_class must differ from candidate_class"
        )
    mapped_second_score = float(scores[second_best])
    if not _scores_equal(second_best_score, mapped_second_score):
        raise EvaluationInputError(
            f"{location}.second_best_score does not match second_best_class "
            "in class_scores"
        )
    expected_second_score = max(
        float(item_score)
        for label, item_score in scores.items()
        if label != candidate
    )
    if not _scores_equal(mapped_second_score, expected_second_score):
        raise EvaluationInputError(
            f"{location}.second_best_class is not the highest-scoring "
            "class other than candidate_class"
        )
    if not _scores_equal(margin, score - second_best_score):
        raise EvaluationInputError(
            f"{location}.margin does not equal score minus second_best_score"
        )


def _require_matching_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Iterable[str],
    location: str,
) -> None:
    mismatched = sorted(field for field in fields if left[field] != right[field])
    if mismatched:
        raise EvaluationInputError(
            f"{location} differs from stage2_raw in: {', '.join(mismatched)}"
        )


def _validate_temporal_evidence(
    raw: Mapping[str, Any],
    decision: Mapping[str, Any],
    expected_classes: frozenset[str],
    location: str,
) -> None:
    temporal = _require_mapping(decision["temporal"], location)
    _require_exact_fields(temporal, TEMPORAL_FIELDS, location)

    method = _require_nonempty_string(temporal["method"], f"{location}.method")
    if method not in {"mean", "vote"}:
        raise EvaluationInputError(f"{location}.method must be mean or vote")

    integer_fields = {
        "window_size": temporal["window_size"],
        "samples": temporal["samples"],
        "switch_confirmations": temporal["switch_confirmations"],
        "pending_count": temporal["pending_count"],
    }
    for field, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise EvaluationInputError(f"{location}.{field} must be an integer")
    window_size = integer_fields["window_size"]
    samples = integer_fields["samples"]
    switch_confirmations = integer_fields["switch_confirmations"]
    pending_count = integer_fields["pending_count"]
    if window_size <= 0:
        raise EvaluationInputError(f"{location}.window_size must be positive")
    if samples < 0 or samples > window_size:
        raise EvaluationInputError(
            f"{location}.samples must be between zero and window_size"
        )
    if switch_confirmations <= 0:
        raise EvaluationInputError(
            f"{location}.switch_confirmations must be positive"
        )
    if pending_count < 0 or pending_count >= switch_confirmations:
        raise EvaluationInputError(
            f"{location}.pending_count must be non-negative and less than "
            "switch_confirmations"
        )

    labels: dict[str, str | None] = {}
    for field in ("stable_class", "proposed_class", "pending_class"):
        label = _require_optional_class(temporal[field], f"{location}.{field}")
        if label is not None and label not in expected_classes:
            raise EvaluationInputError(
                f"{location}.{field} is outside the fixed stage-2 class set"
            )
        labels[field] = label

    pending = labels["pending_class"]
    if (pending is None) != (pending_count == 0):
        raise EvaluationInputError(
            f"{location}.pending_class and pending_count are inconsistent"
        )
    if pending is not None and pending != labels["proposed_class"]:
        raise EvaluationInputError(
            f"{location}.pending_class must equal proposed_class"
        )

    raw_summary = _require_mapping(temporal["raw"], f"{location}.raw")
    _require_exact_fields(raw_summary, TEMPORAL_RAW_FIELDS, f"{location}.raw")
    _require_matching_fields(
        raw,
        raw_summary,
        TEMPORAL_RAW_FIELDS,
        f"{location}.raw",
    )

    raw_candidate = raw["candidate_class"]
    if raw["score_kind"] is not None and decision["score_kind"] != raw["score_kind"]:
        raise EvaluationInputError(
            f"{location}.score_kind must match stage2_raw when raw evidence exists"
        )
    if raw_candidate is not None and samples == 0:
        raise EvaluationInputError(
            f"{location}.samples must be positive when stage2_raw has a candidate"
        )
    if raw_candidate is None and labels["proposed_class"] is not None:
        raise EvaluationInputError(
            f"{location}.proposed_class must be null when stage2_raw has no candidate"
        )

    stable = labels["stable_class"]
    if stable is not None:
        if (
            decision["status"] != "matched"
            or decision["predicted_class"] != stable
            or decision["candidate_class"] != stable
        ):
            raise EvaluationInputError(
                f"{location}.stable_class must match the accepted decision"
            )
        if samples == 0:
            raise EvaluationInputError(
                f"{location}.stable_class requires at least one history sample"
            )
    else:
        if decision["status"] == "matched":
            raise EvaluationInputError(
                f"{location}.matched decision requires stable_class"
            )
        if decision["candidate_class"] != labels["proposed_class"]:
            raise EvaluationInputError(
                f"{location}.proposed_class must match decision.candidate_class "
                "when no stable class exists"
            )
        if raw_candidate is None:
            _require_matching_fields(
                raw,
                decision,
                STAGE2_CORE_FIELDS,
                f"{location} without a stable class",
            )

    if decision["status"] == "confirming":
        if pending is None or pending != decision["candidate_class"]:
            raise EvaluationInputError(
                f"{location}.confirming decision requires its candidate to be pending"
            )


def _validate_shadow_record(record: Any, line_number: int) -> dict[str, Any]:
    location = f"shadow line {line_number}"
    item = _require_mapping(record, location)
    _require_fields(
        item,
        (
            "schema",
            "timestamp",
            "input",
            "source",
            "stage2_raw",
            "decision",
            "route",
            "timing_ms",
        ),
        location,
    )
    if item["schema"] != SHADOW_SCHEMA:
        raise EvaluationInputError(
            f"{location} has schema {item['schema']!r}; expected {SHADOW_SCHEMA!r}"
        )

    _require_nonempty_string(item["timestamp"], f"{location}.timestamp")
    _validate_input(item["input"], f"{location}.input")
    source = _require_mapping(item["source"], f"{location}.source")
    _require_fields(
        source,
        (
            "id",
            "frame_index",
            "major_class",
            "confidence",
            "bbox_xyxy",
            "raw_yolo_class_id",
            "raw_yolo_class",
            "class_scores",
        ),
        f"{location}.source",
    )
    _require_nonempty_string(source["id"], f"{location}.source.id")
    if (
        not isinstance(source["frame_index"], int)
        or isinstance(source["frame_index"], bool)
        or source["frame_index"] < 0
    ):
        raise EvaluationInputError(
            f"{location}.source.frame_index must be a non-negative integer"
        )
    if source["major_class"] not in {"f", "k"}:
        raise EvaluationInputError(
            f"{location}.source.major_class must be f or k"
        )
    confidence = source["confidence"]
    if (
        not _is_number(confidence)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise EvaluationInputError(
            f"{location}.source.confidence must be a finite number between 0 and 1"
        )
    _validate_bbox(source["bbox_xyxy"], f"{location}.source.bbox_xyxy")
    raw_class_id = source["raw_yolo_class_id"]
    if not isinstance(raw_class_id, int) or isinstance(raw_class_id, bool):
        raise EvaluationInputError(
            f"{location}.source.raw_yolo_class_id must be an integer from 0 to 20"
        )
    raw_class = _require_nonempty_string(
        source["raw_yolo_class"], f"{location}.source.raw_yolo_class"
    )
    if raw_class not in YOLO_CLASS_IDS:
        raise EvaluationInputError(
            f"{location}.source.raw_yolo_class is outside the fixed YOLO class set"
        )
    if YOLO_CLASS_IDS[raw_class] != raw_class_id:
        raise EvaluationInputError(
            f"{location}.source raw_yolo_class_id does not match raw_yolo_class"
        )
    if not raw_class.startswith(source["major_class"]):
        raise EvaluationInputError(
            f"{location}.source.major_class does not match raw_yolo_class"
        )
    source_scores = _validate_scores(
        source["class_scores"], f"{location}.source.class_scores"
    )
    _validate_score_domain(
        source_scores,
        frozenset(YOLO_CLASSES),
        f"{location}.source.class_scores",
    )
    raw_score = float(source_scores[raw_class])
    if not _scores_equal(raw_score, max(float(score) for score in source_scores.values())):
        raise EvaluationInputError(
            f"{location}.source.raw_yolo_class is not a maximum-scoring class"
        )
    if not _scores_equal(float(confidence), raw_score):
        raise EvaluationInputError(
            f"{location}.source.confidence does not match raw_yolo_class "
            "in class_scores"
        )

    expected_stage2 = STAGE2_CLASSES[source["major_class"]]
    _validate_stage2_evidence(
        item["stage2_raw"],
        f"{location}.stage2_raw",
        expected_stage2,
        RAW_STATUSES[source["major_class"]],
        candidate_must_be_max=True,
    )
    _validate_stage2_evidence(
        item["decision"],
        f"{location}.decision",
        expected_stage2,
        FINAL_STATUSES[source["major_class"]],
        candidate_must_be_max=False,
    )
    legal_score_kinds = STAGE2_SCORE_KINDS[source["major_class"]]
    for section_name in ("stage2_raw", "decision"):
        score_kind = item[section_name]["score_kind"]
        if score_kind is not None and score_kind not in legal_score_kinds:
            raise EvaluationInputError(
                f"{location}.{section_name}.score_kind is outside the fixed "
                f"{source['major_class']} score kind set"
            )

    route = _require_mapping(item["route"], f"{location}.route")
    _require_fields(route, ("track_id", "execution"), f"{location}.route")
    track_id = route["track_id"]
    if track_id is not None and (
        not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0
    ):
        raise EvaluationInputError(
            f"{location}.route.track_id must be null or a non-negative integer"
        )
    execution = _require_nonempty_string(
        route["execution"], f"{location}.route.execution"
    )
    if execution not in {"direct", "fresh"}:
        raise EvaluationInputError(
            f"{location}.route.execution must be direct or fresh"
        )

    raw_stage2 = _require_mapping(item["stage2_raw"], f"{location}.stage2_raw")
    decision = _require_mapping(item["decision"], f"{location}.decision")
    has_temporal = "temporal" in decision
    if execution == "direct":
        if track_id is not None:
            raise EvaluationInputError(
                f"{location}.route direct execution requires track_id to be null"
            )
        if has_temporal:
            raise EvaluationInputError(
                f"{location}.decision direct execution must not contain temporal"
            )
        _require_matching_fields(
            raw_stage2,
            decision,
            STAGE2_CORE_FIELDS,
            f"{location}.decision",
        )
    else:
        if (
            not isinstance(track_id, int)
            or isinstance(track_id, bool)
            or track_id <= 0
        ):
            raise EvaluationInputError(
                f"{location}.route fresh execution requires a positive track_id"
            )
        if has_temporal:
            _validate_temporal_evidence(
                raw_stage2,
                decision,
                expected_stage2,
                f"{location}.decision.temporal",
            )
        else:
            _require_matching_fields(
                raw_stage2,
                decision,
                STAGE2_CORE_FIELDS,
                f"{location}.decision without temporal smoothing",
            )

    timing = _require_mapping(item["timing_ms"], f"{location}.timing_ms")
    _require_fields(timing, ("stage2",), f"{location}.timing_ms")
    stage2_ms = timing["stage2"]
    if (
        not _is_number(stage2_ms)
        or not math.isfinite(float(stage2_ms))
        or float(stage2_ms) < 0.0
    ):
        raise EvaluationInputError(
            f"{location}.timing_ms.stage2 must be a non-negative finite number"
        )
    return dict(item)


def load_shadow(path: Path) -> ShadowLog:
    """Load and validate a recognition_shadow.v1 JSONL file by source ID."""

    records: dict[str, dict[str, Any]] = {}
    input_provenance: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(
                    f"shadow line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            record = _validate_shadow_record(decoded, line_number)
            record_input = dict(record["input"])
            if input_provenance is None:
                input_provenance = record_input
            elif record_input != input_provenance:
                raise EvaluationInputError(
                    f"shadow line {line_number}.input differs from earlier records"
                )
            source_id = record["source"]["id"]
            if source_id in records:
                raise EvaluationInputError(
                    f"duplicate shadow source id {source_id!r} at line {line_number}"
                )
            records[source_id] = record
    if not records:
        raise EvaluationInputError("shadow JSONL contains no records")
    assert input_provenance is not None
    return ShadowLog(input=input_provenance, records=records)


def _validate_golden_item(item: Any, index: int) -> dict[str, Any]:
    location = f"golden.items[{index}]"
    value = _require_mapping(item, location)
    _require_fields(value, ("id", "expected_class"), location)
    _require_nonempty_string(value["id"], f"{location}.id")
    expected = _require_optional_class(
        value["expected_class"], f"{location}.expected_class"
    )
    if expected is not None and expected not in GOLDEN_CLASSES:
        raise EvaluationInputError(
            f"{location}.expected_class is outside the fixed golden class set"
        )
    if "bbox_xyxy" in value and value["bbox_xyxy"] is not None:
        _validate_bbox(value["bbox_xyxy"], f"{location}.bbox_xyxy")
    return dict(value)


def load_golden(path: Path) -> GoldenSet:
    """Load a golden JSON object containing an ``items`` list."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationInputError(f"golden file is not valid JSON: {exc.msg}") from exc
    document = _require_mapping(decoded, "golden document")
    _require_fields(document, ("schema", "input", "items"), "golden document")
    if document["schema"] != GOLDEN_SCHEMA:
        raise EvaluationInputError(
            f"golden document has schema {document['schema']!r}; "
            f"expected {GOLDEN_SCHEMA!r}"
        )
    input_provenance = _validate_input(document["input"], "golden document.input")
    items = document["items"]
    if not isinstance(items, list):
        raise EvaluationInputError("golden document.items must be a list")
    if not items:
        raise EvaluationInputError("golden document.items must not be empty")

    indexed: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(items):
        item = _validate_golden_item(raw_item, index)
        item_id = item["id"]
        if item_id in indexed:
            raise EvaluationInputError(f"duplicate golden id {item_id!r}")
        indexed[item_id] = item
    return GoldenSet(input=input_provenance, items=indexed)


def _normalized_class(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == UNKNOWN:
        return None
    return str(value)


def _is_known_truth(golden: Mapping[str, Any]) -> bool:
    return golden["expected_class"] not in {None, UNKNOWN}


def _is_explicit_unknown(golden: Mapping[str, Any]) -> bool:
    return golden["expected_class"] == UNKNOWN


def _is_accepted(record: Mapping[str, Any]) -> bool:
    decision = record["decision"]
    return (
        decision["status"] in ACCEPTED_STATUSES
        and _normalized_class(decision["predicted_class"]) is not None
    )


def _is_algorithm_rejection(record: Mapping[str, Any]) -> bool:
    decision = record["decision"]
    return (
        decision["status"] in REJECTION_STATUSES
        and _normalized_class(decision["predicted_class"]) is None
    )


def _display_class(value: Any) -> str:
    return _normalized_class(value) or UNKNOWN


def _major_class(value: Any) -> str:
    label = _normalized_class(value)
    if label is None:
        return UNKNOWN
    normalized = label.lower()
    if normalized == "b0":
        return "b0"
    if normalized.startswith("f"):
        return "f"
    if normalized.startswith("k"):
        return "k"
    return "other"


def _prediction(record: Mapping[str, Any], stage: str) -> str | None:
    if stage == "yolo":
        return _normalized_class(record["source"]["raw_yolo_class"])
    if stage == "stage2":
        return _normalized_class(record["stage2_raw"]["predicted_class"])
    if stage == "stage2_candidate":
        return _normalized_class(record["stage2_raw"]["candidate_class"])
    if stage == "decision":
        return _normalized_class(record["decision"]["predicted_class"])
    raise AssertionError(f"unsupported stage: {stage}")


def _accuracy(
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], stage: str
) -> dict[str, Any]:
    known = [pair for pair in aligned if _is_known_truth(pair[1])]
    predictions = [_prediction(record, stage) for record, _ in known]
    correct = sum(
        predicted == _normalized_class(golden["expected_class"])
        for predicted, (_, golden) in zip(predictions, known)
    )
    assigned = sum(predicted is not None for predicted in predictions)
    return {
        "known_truth": len(known),
        "assigned": assigned,
        "correct": correct,
        "accuracy": _percent(correct, len(known)),
    }


def _confusion(
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], stage: str
) -> dict[str, Any]:
    fine = Counter(
        (
            str(golden["expected_class"]),
            _display_class(_prediction(record, stage)),
        )
        for record, golden in aligned
        if golden["expected_class"] is not None
    )
    major: Counter[tuple[str, str]] = Counter()
    for (expected, predicted), count in fine.items():
        major[(_major_class(expected), _major_class(predicted))] += count
    return {
        "per_class": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(fine.items())
        ],
        "by_major": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(major.items())
        ],
    }


def _agreement(
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    stages: Sequence[str],
) -> dict[str, Any]:
    exact = 0
    major = 0
    for record, _ in aligned:
        predictions = [_prediction(record, stage) for stage in stages]
        exact += int(len(set(predictions)) == 1)
        major += int(len({_major_class(value) for value in predictions}) == 1)
    return {
        "compared": len(aligned),
        "exact": exact,
        "exact_rate": _percent(exact, len(aligned)),
        "same_major": major,
        "same_major_rate": _percent(major, len(aligned)),
    }


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union else 0.0


def _bbox_metrics(
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    values = [
        _bbox_iou(record["source"]["bbox_xyxy"], golden["bbox_xyxy"])
        for record, golden in aligned
        if golden["expected_class"] is not None
        and golden.get("bbox_xyxy") is not None
    ]
    passing = sum(value >= 0.5 for value in values)
    return {
        "compared": len(values),
        "mean_iou": round(sum(values) / len(values), 6) if values else None,
        "min_iou": round(min(values), 6) if values else None,
        "iou_at_0_5": passing,
        "iou_at_0_5_rate": _percent(passing, len(values)),
    }


def _invalid_predictions(
    records: Mapping[str, Mapping[str, Any]],
    golden: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invalid_ids: set[str] = set()
    fields = {
        "yolo": ("source", "raw_yolo_class"),
        "stage2_candidate": ("stage2_raw", "candidate_class"),
        "stage2": ("stage2_raw", "predicted_class"),
        "decision_candidate": ("decision", "candidate_class"),
        "decision": ("decision", "predicted_class"),
    }
    status_error: Counter[str] = Counter()
    status_error_ids: set[str] = set()
    status_mismatch: Counter[str] = Counter()
    status_mismatch_ids: set[str] = set()
    for record in records.values():
        source_id = record["source"]["id"]
        expected_major = record["source"]["major_class"]
        for name, (section, field) in fields.items():
            value = _normalized_class(record[section][field])
            if value is not None and _major_class(value) != expected_major:
                counts[name] += 1
                invalid_ids.add(source_id)
        for stage in ("stage2_raw", "decision"):
            status = record[stage]["status"]
            predicted = _normalized_class(record[stage]["predicted_class"])
            if status == "error":
                status_error[stage] += 1
                status_error_ids.add(source_id)
                invalid_ids.add(source_id)
            accepted_status = status in ACCEPTED_STATUSES
            if accepted_status != (predicted is not None):
                status_mismatch[stage] += 1
                status_mismatch_ids.add(source_id)
                invalid_ids.add(source_id)

    truth_mismatch_ids: set[str] = set()
    for source_id in records.keys() & golden.keys():
        expected = golden[source_id]["expected_class"]
        if expected in {None, UNKNOWN}:
            continue
        if _major_class(expected) != records[source_id]["source"]["major_class"]:
            truth_mismatch_ids.add(source_id)
            invalid_ids.add(source_id)
    return {
        "records": len(invalid_ids),
        "ids": sorted(invalid_ids),
        "no_scorable_candidate": 0,
        "major_mismatch": {name: counts[name] for name in fields},
        "status_error": {
            "records": len(status_error_ids),
            "ids": sorted(status_error_ids),
            "by_stage": {stage: status_error[stage] for stage in ("stage2_raw", "decision")},
        },
        "status_prediction_mismatch": {
            "records": len(status_mismatch_ids),
            "ids": sorted(status_mismatch_ids),
            "by_stage": {stage: status_mismatch[stage] for stage in ("stage2_raw", "decision")},
        },
        "truth_source_major_mismatch": {
            "records": len(truth_mismatch_ids),
            "ids": sorted(truth_mismatch_ids),
        },
    }


def _subset_metrics(
    aligned: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    known = [pair for pair in aligned if _is_known_truth(pair[1])]
    matched = [pair for pair in aligned if _is_accepted(pair[0])]
    matched_known = [pair for pair in matched if _is_known_truth(pair[1])]
    matched_correct = sum(
        _prediction(record, "decision") == _normalized_class(golden["expected_class"])
        for record, golden in matched_known
    )
    return {
        "items": len(aligned),
        "known_truth": len(known),
        "matched": len(matched),
        "coverage": _percent(len(matched), len(aligned)),
        "matched_known_truth": len(matched_known),
        "matched_correct": matched_correct,
        "matched_accuracy": _percent(matched_correct, len(matched_known)),
        "accuracy": {
            stage: _accuracy(aligned, stage)
            for stage in ("yolo", "stage2", "decision")
        },
    }


def evaluate(
    shadow: ShadowLog,
    golden: GoldenSet,
) -> dict[str, Any]:
    """Compare validated shadow records and golden items aligned by source ID."""

    if shadow.input != golden.input:
        raise EvaluationInputError(
            "shadow input provenance does not match golden input provenance"
        )
    shadow_ids = set(shadow.records)
    golden_ids = set(golden.items)
    aligned_ids = sorted(shadow_ids & golden_ids)
    aligned = [
        (shadow.records[item_id], golden.items[item_id]) for item_id in aligned_ids
    ]
    missing_shadow = sorted(golden_ids - shadow_ids)
    missing_golden = sorted(shadow_ids - golden_ids)
    invalid = _invalid_predictions(shadow.records, golden.items)

    subset = _subset_metrics(aligned)
    unknown = [pair for pair in aligned if _is_explicit_unknown(pair[1])]
    unknown_rejected = sum(
        _is_algorithm_rejection(record) for record, _ in unknown
    )
    unlabeled = [pair for pair in aligned if pair[1]["expected_class"] is None]
    known = [pair for pair in aligned if _is_known_truth(pair[1])]
    by_major = {
        major: _subset_metrics(
            [pair for pair in aligned if pair[0]["source"]["major_class"] == major]
        )
        for major in ("b0", "f", "k")
        if any(pair[0]["source"]["major_class"] == major for pair in aligned)
    }

    return {
        "schema": OUTPUT_SCHEMA,
        "input": dict(shadow.input),
        "counts": {
            "shadow_records": len(shadow.records),
            "golden_items": len(golden.items),
            "aligned": len(aligned),
            "missing_shadow": len(missing_shadow),
            "missing_golden": len(missing_golden),
            "invalid_records": invalid["records"],
        },
        "missing": {
            "shadow_for_golden": missing_shadow,
            "golden_for_shadow": missing_golden,
        },
        "invalid": invalid,
        "coverage": {
            "aligned": len(aligned),
            "golden_items": len(golden.items),
            "alignment_rate": _percent(len(aligned), len(golden.items)),
            "matched": subset["matched"],
            "matched_rate": subset["coverage"],
        },
        "truth": {
            "labeled": len(known) + len(unknown),
            "known": len(known),
            "unknown": len(unknown),
            "unlabeled": len(unlabeled),
        },
        "matched_accuracy": {
            "known_truth": subset["matched_known_truth"],
            "correct": subset["matched_correct"],
            "accuracy": subset["matched_accuracy"],
        },
        "accuracy": {
            "yolo": _accuracy(aligned, "yolo"),
            "stage2": _accuracy(aligned, "stage2"),
            "stage2_candidate": _accuracy(aligned, "stage2_candidate"),
            "decision": _accuracy(aligned, "decision"),
        },
        "unknown_truth": {
            "items": len(unknown),
            "decision_rejected": unknown_rejected,
            "decision_rejection_rate": _percent(unknown_rejected, len(unknown)),
        },
        "agreement": {
            "yolo_stage2": _agreement(aligned, ("yolo", "stage2")),
            "stage2_decision": _agreement(aligned, ("stage2", "decision")),
            "yolo_decision": _agreement(aligned, ("yolo", "decision")),
            "all_three": _agreement(aligned, ("yolo", "stage2", "decision")),
        },
        "bbox": _bbox_metrics(aligned),
        "confusion": {
            stage: _confusion(aligned, stage)
            for stage in ("yolo", "stage2", "decision")
        },
        "by_major": by_major,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recognition_shadow.v1 JSONL against a golden set."
    )
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        shadow = load_shadow(args.shadow.expanduser().resolve())
        golden = load_golden(args.golden.expanduser().resolve())
        result = evaluate(shadow, golden)
    except (OSError, EvaluationInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    coverage = result["coverage"]
    matched = result["matched_accuracy"]
    print(
        f"aligned: {coverage['aligned']}/{coverage['golden_items']} "
        f"({coverage['alignment_rate'] or 0.0:.2%})"
    )
    print(
        f"matched: {coverage['matched']}/{coverage['aligned']} "
        f"({coverage['matched_rate'] or 0.0:.2%})"
    )
    matched_accuracy = matched["accuracy"]
    print(
        "matched accuracy: "
        + (f"{matched_accuracy:.2%}" if matched_accuracy is not None else "n/a")
    )
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
