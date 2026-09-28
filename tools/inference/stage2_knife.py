#!/usr/bin/env python3
"""Run PaddleClas knife recognition for k-crops from the YOLO gateway."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.knife.engine import DEFAULT_MODEL_DIR, PaddleClasONNXEngine
from insight_cup.knife.pipeline import process_handoff
from insight_cup.knife.retrieval import (
    DEFAULT_RETRIEVAL_GALLERY,
    DEFAULT_RETRIEVAL_MODEL,
    PaddleClasRetrievalEngine,
)
from insight_cup.knife.service import (
    DEFAULT_KNIFE_MIN_MARGIN,
    DEFAULT_KNIFE_THRESHOLD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize k1...k10 from stage1_yolo_handoff.v1 crops."
    )
    parser.add_argument("--handoff", required=True, help="stage1_handoff.json")
    parser.add_argument(
        "--mode",
        choices=("classification", "retrieval"),
        default="classification",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--retrieval-model", default=str(DEFAULT_RETRIEVAL_MODEL))
    parser.add_argument("--gallery", default=str(DEFAULT_RETRIEVAL_GALLERY))
    parser.add_argument(
        "--out",
        default="outputs/stage2_knife/stage2_knife_handoff.json",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_detections is not None and args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")

    if args.mode == "retrieval":
        engine = PaddleClasRetrievalEngine(
            model_path=args.retrieval_model,
            gallery_path=args.gallery,
            provider=args.provider,
            cpu_threads=args.cpu_threads,
        )
        threshold = 0.50 if args.threshold is None else args.threshold
        min_margin = 0.02 if args.min_margin is None else args.min_margin
    else:
        engine = PaddleClasONNXEngine(
            model_dir=args.model_dir,
            provider=args.provider,
            cpu_threads=args.cpu_threads,
        )
        threshold = (
            DEFAULT_KNIFE_THRESHOLD if args.threshold is None else args.threshold
        )
        min_margin = (
            DEFAULT_KNIFE_MIN_MARGIN if args.min_margin is None else args.min_margin
        )
    last_reported = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed == total or completed - last_reported >= 100:
            print(f"Processed {completed}/{total} k detections", flush=True)
            last_reported = completed

    result = process_handoff(
        handoff_path=args.handoff,
        engine=engine,
        output_path=args.out,
        threshold=threshold,
        min_margin=min_margin,
        max_detections=args.max_detections,
        progress=report_progress,
    )
    print(f"Output: {Path(args.out).expanduser().resolve()}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
