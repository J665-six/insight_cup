#!/usr/bin/env python3
"""Build an f-class reference gallery with InsightFace embeddings."""

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
from insight_cup.face.gallery import build_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an InsightFace gallery from references/fN/*.jpg."
    )
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", default="data/face_gallery/gallery.npz")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--det-thresh", type=float, default=0.25)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--upsample-min-side", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InsightFaceEmbeddingEngine(
        model_name=args.model_name,
        model_root=args.model_root,
        provider=args.provider,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
        upsample_min_side=args.upsample_min_side,
    )
    metadata = build_gallery(args.references, args.output, engine)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
