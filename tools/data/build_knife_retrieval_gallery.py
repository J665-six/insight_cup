#!/usr/bin/env python3
"""Build and evaluate a PP-ShiTuV2 feature gallery for k1...k10."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.knife.engine import DEFAULT_LABELS
from insight_cup.knife.retrieval import (
    DEFAULT_RETRIEVAL_GALLERY,
    DEFAULT_RETRIEVAL_MODEL,
    KnifeFeatureGallery,
    PaddleClasFeatureExtractor,
)
from insight_cup.vendor.paddleclas_core import FlatInnerProductIndex


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "knife_dataset" / "trainv5_grouped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a k1...k10 gallery with the official PP-ShiTuV2 model."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", default=str(DEFAULT_RETRIEVAL_MODEL))
    parser.add_argument("--gallery", default=str(DEFAULT_RETRIEVAL_GALLERY))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(dataset_root: Path, split: str) -> list[tuple[Path, str, str]]:
    list_path = dataset_root / f"{split}_list.txt"
    records: list[tuple[Path, str, str]] = []
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        relative, separator, class_id = line.rpartition(" ")
        if not separator or not class_id.isdigit():
            raise ValueError(f"Invalid {list_path.name} line {line_number}: {raw_line!r}")
        label_index = int(class_id)
        if not 0 <= label_index < len(DEFAULT_LABELS):
            raise ValueError(f"Invalid knife class ID on line {line_number}: {class_id}")
        image_path = dataset_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Gallery image not found: {image_path}")
        records.append((image_path, DEFAULT_LABELS[label_index], relative))
    if not records:
        raise RuntimeError(f"No images found in {list_path}")
    return records


def _batches(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _extract_records(
    extractor: PaddleClasFeatureExtractor,
    records: list[tuple[Path, str, str]],
    batch_size: int,
    title: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    completed = 0
    for batch in _batches(records, batch_size):
        images = [cv2.imread(str(record[0])) for record in batch]
        unreadable = [str(batch[i][0]) for i, image in enumerate(images) if image is None]
        if unreadable:
            raise RuntimeError(f"Could not read images: {unreadable[:3]}")
        features.append(extractor.extract_batch(images))
        completed += len(batch)
        print(f"{title}: {completed}/{len(records)}", end="\r", flush=True)
    print()
    return np.concatenate(features, axis=0)


def _rank_labels(
    index: FlatInnerProductIndex,
    gallery_labels: np.ndarray,
    queries: np.ndarray,
) -> list[tuple[str, float, str, float]]:
    search = index.search(queries, index.count)
    ranked: list[tuple[str, float, str, float]] = []
    for scores, indices in zip(search.scores, search.indices):
        candidates: list[tuple[str, float]] = []
        for score, gallery_index in zip(scores, indices):
            label = str(gallery_labels[int(gallery_index)])
            if any(existing == label for existing, _ in candidates):
                continue
            candidates.append((label, float(score)))
            if len(candidates) == 2:
                break
        ranked.append(
            (candidates[0][0], candidates[0][1], candidates[1][0], candidates[1][1])
        )
    return ranked


def _evaluate(
    extractor: PaddleClasFeatureExtractor,
    index: FlatInnerProductIndex,
    gallery_labels: np.ndarray,
    records: list[tuple[Path, str, str]],
    batch_size: int,
    split: str,
) -> dict:
    features = _extract_records(extractor, records, batch_size, f"evaluate {split}")
    ranked = _rank_labels(index, gallery_labels, features)
    correct = np.asarray(
        [
            candidate == record[1]
            for (candidate, _, _, _), record in zip(ranked, records)
        ],
        dtype=np.bool_,
    )
    scores = np.asarray([row[1] for row in ranked], dtype=np.float32)
    margins = np.asarray([row[1] - row[3] for row in ranked], dtype=np.float32)
    operating_points = []
    for threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        for min_margin in (0.00, 0.01, 0.02, 0.03, 0.05, 0.10):
            accepted = (scores >= threshold) & (margins >= min_margin)
            count = int(accepted.sum())
            operating_points.append(
                {
                    "threshold": threshold,
                    "min_margin": min_margin,
                    "coverage": round(count / len(records), 6),
                    "accepted_precision": (
                        round(float(correct[accepted].mean()), 6) if count else None
                    ),
                }
            )
    return {
        "images": len(records),
        "top1_accuracy": round(float(correct.mean()), 6),
        "candidate_counts": dict(Counter(row[0] for row in ranked)),
        "score_quantiles": {
            str(q): round(float(np.quantile(scores, q)), 6)
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "margin_quantiles": {
            str(q): round(float(np.quantile(margins, q)), 6)
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "operating_points": operating_points,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("batch size and CPU threads must be positive")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    gallery_path = Path(args.gallery).expanduser().resolve()
    train_records = _read_records(dataset_root, "train")
    extractor = PaddleClasFeatureExtractor(
        model_path=model_path,
        provider=args.provider,
        cpu_threads=args.cpu_threads,
    )

    started = time.perf_counter()
    embeddings = _extract_records(
        extractor, train_records, args.batch_size, "build gallery"
    )
    gallery = KnifeFeatureGallery(
        labels=np.asarray([record[1] for record in train_records], dtype=np.str_),
        embeddings=embeddings,
        references=np.asarray([record[2] for record in train_records], dtype=np.str_),
    )
    gallery.save(gallery_path)
    index = FlatInnerProductIndex(gallery.embeddings)
    evaluations = {
        split: _evaluate(
            extractor,
            index,
            gallery.labels,
            _read_records(dataset_root, split),
            args.batch_size,
            split,
        )
        for split in ("val", "test")
    }
    metadata = {
        "schema": "knife_feature_gallery.v1",
        "backend": "PaddleClas PP-ShiTuV2 + Faiss IndexFlatIP",
        "upstream_commit": "f1233c18455b8acde4fc42ab0bea575fa06daa8e",
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "dataset_root": str(dataset_root),
        "gallery": str(gallery_path),
        "references": int(gallery.labels.size),
        "embedding_size": int(gallery.embeddings.shape[1]),
        "label_counts": dict(Counter(gallery.labels.tolist())),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "evaluations": evaluations,
    }
    metadata_path = gallery_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Gallery: {gallery_path}")
    print(f"Metadata: {metadata_path}")
    print(
        json.dumps(
            {
                split: {"top1_accuracy": value["top1_accuracy"]}
                for split, value in evaluations.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
