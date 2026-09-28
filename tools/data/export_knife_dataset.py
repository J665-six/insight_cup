#!/usr/bin/env python3
"""Export grouped PaddleClas crops from YOLO k1...k10 annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.common.crops import KNIFE_CROP_CONTEXT_SCALE, square_context_crop


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
KNIFE_PATTERN = re.compile(r"k([1-9]|10)", re.IGNORECASE)
RECORDING_PATTERN = re.compile(r"d455_color_(\d{8}_\d{6}_\d+)")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class KnifeBox:
    source_image: Path
    source_label: Path
    source_split: str
    recording_id: str
    line_number: int
    class_id: int
    class_name: str
    normalized_xywh: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a recording-grouped PaddleClas knife dataset."
    )
    parser.add_argument("--data", default="/home/j/trainv5/data.yaml")
    parser.add_argument("--output", default="data/knife_dataset/trainv5_grouped")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-trials", type=int, default=20000)
    parser.add_argument(
        "--context-scale",
        type=float,
        default=KNIFE_CROP_CONTEXT_SCALE,
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def _class_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(name).strip().lower() for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name).strip().lower() for index, name in enumerate(value)}
    raise ValueError("YOLO data config must contain a names mapping or list")


def _dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("path", config_path.parent))).expanduser()
    if not configured.is_absolute():
        configured = config_path.parent / configured
    return configured.resolve()


def _split_paths(root: Path, configured: Any) -> list[Path]:
    values = configured if isinstance(configured, list) else [configured]
    result = []
    for value in values:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        result.append(path.resolve())
    return result


def _label_path(root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.relative_to(root / "images")
    except ValueError as exc:
        raise ValueError(f"Image is not below {root / 'images'}: {image_path}") from exc
    return (root / "labels" / relative).with_suffix(".txt")


def _recording_id(image_path: Path) -> str:
    match = RECORDING_PATTERN.search(image_path.stem)
    if match:
        return match.group(1)
    prefix, separator, _ = image_path.stem.partition("_frame_")
    if separator:
        return prefix
    raise ValueError(f"Cannot determine source recording from {image_path.name}")


def collect_boxes(config_path: Path) -> tuple[list[KnifeBox], dict[int, str]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _dataset_root(config_path, config)
    names = _class_names(config.get("names"))
    knife_source_ids = {
        class_id: class_name
        for class_id, class_name in names.items()
        if KNIFE_PATTERN.fullmatch(class_name)
    }
    expected = {f"k{index}" for index in range(1, 11)}
    if set(knife_source_ids.values()) != expected:
        raise ValueError(
            f"Expected exactly k1...k10 in YOLO config, got {sorted(knife_source_ids.values())}"
        )
    output_ids = {
        source_id: int(class_name[1:]) - 1
        for source_id, class_name in knife_source_ids.items()
    }

    boxes: list[KnifeBox] = []
    seen_images: set[Path] = set()
    for source_split in SPLIT_NAMES:
        configured = config.get(source_split)
        if configured is None:
            continue
        for image_root in _split_paths(root, configured):
            if not image_root.is_dir():
                raise FileNotFoundError(f"Image split directory not found: {image_root}")
            for image_path in sorted(image_root.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image_path = image_path.resolve()
                if image_path in seen_images:
                    raise ValueError(f"Duplicate image across configured splits: {image_path}")
                seen_images.add(image_path)
                label_path = _label_path(root, image_path)
                if not label_path.is_file():
                    raise FileNotFoundError(f"YOLO label not found: {label_path}")
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(),
                    1,
                ):
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(
                            f"Invalid YOLO row in {label_path}:{line_number}: {line!r}"
                        )
                    source_class_id = int(fields[0])
                    if source_class_id not in output_ids:
                        continue
                    coordinates = tuple(float(value) for value in fields[1:])
                    if any(not np.isfinite(value) for value in coordinates):
                        raise ValueError(f"Non-finite box in {label_path}:{line_number}")
                    boxes.append(
                        KnifeBox(
                            source_image=image_path,
                            source_label=label_path,
                            source_split=source_split,
                            recording_id=_recording_id(image_path),
                            line_number=line_number,
                            class_id=output_ids[source_class_id],
                            class_name=knife_source_ids[source_class_id],
                            normalized_xywh=coordinates,
                        )
                    )
    if not boxes:
        raise RuntimeError("No k1...k10 boxes were found")
    return boxes, {index: f"k{index + 1}" for index in range(10)}


def assign_recording_splits(
    boxes: list[KnifeBox],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    trials: int,
) -> dict[str, str]:
    train_fraction = 1.0 - val_fraction - test_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0.0:
        raise ValueError("train, val, and test fractions must all be positive")
    if trials <= 0:
        raise ValueError("split-trials must be positive")

    group_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(10, dtype=np.int64)
    )
    for box in boxes:
        group_counts[box.recording_id][box.class_id] += 1
    groups = sorted(group_counts)
    if len(groups) < 3:
        raise ValueError("At least three recording groups are required")

    total_counts = np.sum([group_counts[group] for group in groups], axis=0)
    target_fractions = np.asarray(
        [train_fraction, val_fraction, test_fraction],
        dtype=np.float64,
    )
    number_val = max(1, int(round(len(groups) * val_fraction)))
    number_test = max(1, int(round(len(groups) * test_fraction)))
    if number_val + number_test >= len(groups):
        raise ValueError("Requested validation/test fractions leave no training groups")

    generator = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(trials):
        shuffled = groups.copy()
        generator.shuffle(shuffled)
        test_groups = set(shuffled[:number_test])
        val_groups = set(shuffled[number_test : number_test + number_val])
        assignment = {
            group: (
                "test"
                if group in test_groups
                else "val"
                if group in val_groups
                else "train"
            )
            for group in groups
        }
        split_counts = np.stack(
            [
                np.sum(
                    [
                        group_counts[group]
                        for group in groups
                        if assignment[group] == split_name
                    ],
                    axis=0,
                )
                for split_name in SPLIT_NAMES
            ]
        )
        if np.any(split_counts == 0):
            continue
        actual = split_counts / total_counts[np.newaxis, :]
        class_error = float(np.mean((actual - target_fractions[:, np.newaxis]) ** 2))
        total_by_split = split_counts.sum(axis=1) / split_counts.sum()
        total_error = float(np.mean((total_by_split - target_fractions) ** 2))
        score = class_error + total_error
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not find a grouped split containing every knife class")
    return best_assignment


def _pixel_box(
    normalized_xywh: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    center_x, center_y, box_width, box_height = normalized_xywh
    x1 = (center_x - box_width * 0.5) * width
    y1 = (center_y - box_height * 0.5) * height
    x2 = (center_x + box_width * 0.5) * width
    y2 = (center_y + box_height * 0.5) * height
    return x1, y1, x2, y2


def export_dataset(
    boxes: list[KnifeBox],
    labels: dict[int, str],
    assignments: dict[str, str],
    output_dir: Path,
    context_scale: float,
    jpeg_quality: int,
    source_config: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Move it aside before rebuilding."
        )
    if context_scale < 1.0:
        raise ValueError("context-scale must be at least 1.0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)

    list_rows: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    records: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    image_cache: dict[Path, np.ndarray] = {}
    try:
        for box_index, box in enumerate(boxes):
            split_name = assignments[box.recording_id]
            image = image_cache.get(box.source_image)
            if image is None:
                image = cv2.imread(str(box.source_image))
                if image is None:
                    raise RuntimeError(f"Failed to read source image: {box.source_image}")
                image_cache[box.source_image] = image
            height, width = image.shape[:2]
            pixel_box = _pixel_box(box.normalized_xywh, width, height)
            crop = square_context_crop(
                image,
                pixel_box,
                context_scale=context_scale,
            )
            relative_crop = (
                Path("images")
                / split_name
                / box.class_name
                / f"{box.source_split}_{box.source_image.stem}__box{box.line_number:02d}.jpg"
            )
            destination = temporary / relative_crop
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(destination),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Failed to write crop: {destination}")
            list_rows[split_name].append(f"{relative_crop.as_posix()} {box.class_id}")
            counts[split_name][box.class_name] += 1
            records.append(
                {
                    "crop": relative_crop.as_posix(),
                    "split": split_name,
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    "recording_id": box.recording_id,
                    "source_split": box.source_split,
                    "source_image": str(box.source_image),
                    "source_label": str(box.source_label),
                    "source_label_line": box.line_number,
                    "normalized_xywh": list(box.normalized_xywh),
                    "pixel_xyxy": [round(value, 3) for value in pixel_box],
                    "crop_shape": [int(crop.shape[0]), int(crop.shape[1])],
                }
            )
            if len(image_cache) > 32:
                image_cache.pop(next(iter(image_cache)))

        for split_name in SPLIT_NAMES:
            rows = sorted(list_rows[split_name])
            (temporary / f"{split_name}_list.txt").write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )
        (temporary / "labels.txt").write_text(
            "".join(f"{class_id} {labels[class_id]}\n" for class_id in sorted(labels)),
            encoding="utf-8",
        )

        groups_by_split = {
            split_name: sorted(
                group for group, assigned in assignments.items() if assigned == split_name
            )
            for split_name in SPLIT_NAMES
        }
        manifest = {
            "schema": "knife_classification_dataset.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_config": str(source_config),
            "seed": seed,
            "context_scale": context_scale,
            "labels": {str(key): value for key, value in labels.items()},
            "recording_groups": groups_by_split,
            "counts": {
                split_name: {
                    "total": sum(counts[split_name].values()),
                    "by_class": dict(sorted(counts[split_name].items())),
                }
                for split_name in SPLIT_NAMES
            },
            "records": records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    config_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    boxes, labels = collect_boxes(config_path)
    assignments = assign_recording_splits(
        boxes,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        trials=args.split_trials,
    )
    manifest = export_dataset(
        boxes=boxes,
        labels=labels,
        assignments=assignments,
        output_dir=output_dir,
        context_scale=args.context_scale,
        jpeg_quality=args.jpeg_quality,
        source_config=config_path,
        seed=args.seed,
    )
    print(f"Output: {output_dir}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
