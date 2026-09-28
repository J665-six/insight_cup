#!/usr/bin/env python3
"""Evaluate the exported knife classifier on a PaddleClas list file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.knife.engine import DEFAULT_MODEL_DIR, PaddleClasONNXEngine
from insight_cup.knife.service import (
    DEFAULT_KNIFE_MIN_MARGIN,
    DEFAULT_KNIFE_THRESHOLD,
)


DEFAULT_DATASET = PROJECT_ROOT / "data" / "knife_dataset" / "trainv5_grouped"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "knife" / "evaluation" / "test_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the PaddleClas-derived ONNX knife classifier."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--list",
        dest="list_path",
        type=Path,
        default=None,
        help="PaddleClas list file (default: <dataset>/<split>_list.txt)",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=DEFAULT_KNIFE_THRESHOLD)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_KNIFE_MIN_MARGIN)
    return parser.parse_args()


def load_samples(dataset: Path, list_path: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        relative_path, separator, class_id = value.rpartition(" ")
        if not separator or not relative_path or not class_id.isdigit():
            raise ValueError(f"Invalid dataset line {line_number}: {line!r}")
        image_path = dataset / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Dataset image not found: {image_path}")
        samples.append((image_path, int(class_id)))
    if not samples:
        raise ValueError(f"Dataset list is empty: {list_path}")
    return samples


def percentile(values: list[float], value: float) -> float:
    return round(float(np.percentile(np.asarray(values), value)), 6)


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 6),
        "p10": percentile(values, 10),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": round(max(values), 6),
        "mean": round(float(np.mean(values)), 6),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    list_path = (
        args.list_path.expanduser().resolve()
        if args.list_path
        else dataset / f"{args.split}_list.txt"
    )
    samples = load_samples(dataset, list_path)
    engine = PaddleClasONNXEngine(args.model_dir, provider=args.provider)
    labels = engine.labels
    if any(class_id >= len(labels) for _, class_id in samples):
        raise ValueError("Dataset contains a class ID absent from the model label map")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    top1_correct = 0
    top2_correct = 0
    accepted_correct = 0
    status_counts: Counter[str] = Counter()
    confidences: list[float] = []
    margins: list[float] = []
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    inference_seconds = 0.0

    for offset in range(0, len(samples), args.batch_size):
        batch_samples = samples[offset : offset + args.batch_size]
        images = []
        for image_path, _ in batch_samples:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            images.append(image)
        started = time.perf_counter()
        predictions = engine.predict_batch(images)
        inference_seconds += time.perf_counter() - started

        for (_, expected), prediction in zip(batch_samples, predictions):
            predicted = prediction.class_ids[0]
            confidence = prediction.scores[0]
            second_confidence = prediction.scores[1]
            margin = confidence - second_confidence
            is_correct = predicted == expected
            is_accepted = confidence >= args.threshold and margin >= args.min_margin
            confusion[expected, predicted] += 1
            top1_correct += int(is_correct)
            top2_correct += int(expected in prediction.class_ids[:2])
            accepted_correct += int(is_correct and is_accepted)
            status_counts["matched" if is_accepted else "rejected"] += 1
            confidences.append(confidence)
            margins.append(margin)
            (correct_confidences if is_correct else incorrect_confidences).append(confidence)
            (correct_margins if is_correct else incorrect_margins).append(margin)

    total = len(samples)
    accepted = status_counts["matched"]
    per_class = {}
    for class_id, label in enumerate(labels):
        support = int(confusion[class_id].sum())
        correct = int(confusion[class_id, class_id])
        per_class[label] = {
            "class_id": class_id,
            "support": support,
            "correct": correct,
            "accuracy": round(correct / support, 6) if support else None,
        }
    timing = {
        "batch_size": args.batch_size,
        "measured_images": total,
        "preprocess_and_inference_seconds": round(inference_seconds, 6),
        "milliseconds_per_image": round(inference_seconds * 1000.0 / total, 3),
        "images_per_second": round(total / inference_seconds, 3),
    }
    return {
        "schema": "knife_classifier_evaluation.v1",
        "model": engine.metadata,
        "dataset": {
            "root": str(dataset),
            "list": str(list_path),
            "split": args.split,
            "samples": total,
        },
        "metrics": {
            "top1": round(top1_correct / total, 6),
            "top1_correct": top1_correct,
            "top2": round(top2_correct / total, 6),
            "top2_correct": top2_correct,
            "per_class": per_class,
            "confusion_matrix": {
                "rows_expected_columns_predicted": list(labels),
                "values": confusion.tolist(),
            },
        },
        "decision": {
            "threshold": args.threshold,
            "min_margin": args.min_margin,
            "matched": accepted,
            "rejected": total - accepted,
            "coverage": round(accepted / total, 6),
            "matched_correct": accepted_correct,
            "matched_precision": round(accepted_correct / accepted, 6) if accepted else None,
            "note": "Known-class data only; this does not measure unknown-class rejection.",
        },
        "score_distributions": {
            "all_confidence": summarize_values(confidences),
            "all_margin": summarize_values(margins),
            "correct_confidence": summarize_values(correct_confidences),
            "incorrect_confidence": summarize_values(incorrect_confidences),
            "correct_margin": summarize_values(correct_margins),
            "incorrect_margin": summarize_values(incorrect_margins),
        },
        "timing": timing,
    }


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    decision = result["decision"]
    timing = result["timing"]
    print(f"samples: {result['dataset']['samples']}")
    print(f"top1: {metrics['top1']:.2%} ({metrics['top1_correct']})")
    print(f"top2: {metrics['top2']:.2%} ({metrics['top2_correct']})")
    print(
        f"matched: {decision['coverage']:.2%}, "
        f"precision: {decision['matched_precision']:.2%}"
    )
    print(f"speed: {timing['milliseconds_per_image']:.3f} ms/image")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
