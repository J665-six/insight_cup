#!/usr/bin/env python3
"""Offline-only evaluation against stage-1 debug subclass labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stage-2 face output against stage1_debug.json. "
            "Never use this tool in the production handoff path."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--debug", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def evaluate(predictions: dict[str, Any], debug: dict[str, Any]) -> dict[str, Any]:
    truth = {
        detection["id"]: detection["raw_yolo_class"]
        for frame in debug.get("frames", [])
        for detection in frame.get("detections", [])
        if detection.get("major_class") == "f"
    }
    rows = [
        row
        for row in predictions.get("detections", [])
        if row.get("id") in truth
    ]
    detected = [
        row
        for row in rows
        if row.get("status") not in {"no_face", "missing_crop", "read_error"}
    ]
    assigned = [row for row in rows if row.get("predicted_class") is not None]
    top1_correct = sum(
        row.get("candidate_class") == truth[row["id"]] for row in detected
    )
    assigned_correct = sum(
        row.get("predicted_class") == truth[row["id"]] for row in assigned
    )
    confusion = Counter(
        (truth[row["id"]], str(row["predicted_class"])) for row in assigned
    )
    return {
        "schema": "stage2_face_offline_evaluation.v1",
        "warning": (
            "Debug-only evaluation. stage1 raw subclasses are not inputs to "
            "the face-recognition runtime."
        ),
        "evaluated_detections": len(rows),
        "face_detected": len(detected),
        "face_detection_rate": percent(len(detected), len(rows)),
        "top1_correct_on_detected": top1_correct,
        "top1_accuracy_on_detected": percent(top1_correct, len(detected)),
        "assigned": len(assigned),
        "assignment_rate": percent(len(assigned), len(rows)),
        "assigned_correct": assigned_correct,
        "assigned_precision": percent(assigned_correct, len(assigned)),
        "confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in confusion.most_common()
        ],
    }


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser().resolve()
    debug_path = Path(args.debug).expanduser().resolve()
    result = evaluate(
        json.loads(predictions_path.read_text(encoding="utf-8")),
        json.loads(debug_path.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Output: {output_path}")
    print(rendered)


if __name__ == "__main__":
    main()
