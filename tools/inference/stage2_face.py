#!/usr/bin/env python3
"""Run face identity matching for f-crops from the YOLO stage-1 gateway."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.face.engine import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_ROOT,
    InsightFaceEmbeddingEngine,
)
from insight_cup.face.gallery import FaceGallery
from insight_cup.face.pipeline import process_handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize f identities from stage1_yolo_handoff.v1 crops."
    )
    parser.add_argument("--handoff", required=True, help="stage1_handoff.json")
    parser.add_argument(
        "--gallery", default="data/face_gallery/trainv5_gallery_r50.npz"
    )
    parser.add_argument("--out", default="outputs/stage2_face/stage2_face_handoff.json")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--det-thresh", type=float, default=0.18)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--upsample-min-side", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_detections is not None and args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")

    gallery = FaceGallery.load(args.gallery)
    engine = InsightFaceEmbeddingEngine(
        model_name=args.model_name,
        model_root=args.model_root,
        provider=args.provider,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
        upsample_min_side=args.upsample_min_side,
    )

    last_reported = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed == total or completed - last_reported >= 100:
            print(f"Processed {completed}/{total} f detections", flush=True)
            last_reported = completed

    result = process_handoff(
        handoff_path=args.handoff,
        gallery=gallery,
        engine=engine,
        output_path=args.out,
        threshold=args.threshold,
        min_margin=args.min_margin,
        max_detections=args.max_detections,
        progress=report_progress,
    )
    print(f"Output: {Path(args.out).expanduser().resolve()}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
