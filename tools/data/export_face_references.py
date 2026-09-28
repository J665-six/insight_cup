#!/usr/bin/env python3
"""Export high-resolution f-class crops from an existing YOLO dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
FACE_CLASS = re.compile(r"f\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    identity: str
    image_path: Path
    label_index: int
    bbox_xyxy: tuple[int, int, int, int]
    quality: float
    recording_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create references/fN/*.jpg from YOLO labels."
    )
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/face_gallery/references")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--min-side", type=int, default=24)
    return parser.parse_args()


def _class_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    raise ValueError("data.yaml must contain a names mapping or list")


def _dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", config_path.parent))).expanduser()
    if not raw_root.is_absolute():
        raw_root = config_path.parent / raw_root
    return raw_root.resolve()


def _split_roots(root: Path, split_value: str) -> tuple[Path, Path]:
    image_root = Path(split_value).expanduser()
    if not image_root.is_absolute():
        image_root = root / image_root
    image_root = image_root.resolve()
    parts = list(image_root.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Cannot derive labels directory from {image_root}") from exc
    parts[image_index] = "labels"
    return image_root, Path(*parts)


def _recording_key(stem: str) -> str:
    source = stem.split("__", 1)[-1]
    return re.sub(r"_frame_.*$", "", source)


def collect_candidates(
    image_root: Path,
    label_root: Path,
    names: dict[int, str],
    min_side: int,
) -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    for image_path in image_paths:
        relative = image_path.relative_to(image_root).with_suffix(".txt")
        label_path = label_root / relative
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for label_index, line in enumerate(label_path.read_text().splitlines()):
            fields = line.split()
            if len(fields) < 5:
                continue
            class_id = int(float(fields[0]))
            identity = names.get(class_id, "").lower()
            if not FACE_CLASS.fullmatch(identity):
                continue
            center_x, center_y, box_width, box_height = map(float, fields[1:5])
            x1 = max(0, int(round((center_x - box_width * 0.5) * width)))
            y1 = max(0, int(round((center_y - box_height * 0.5) * height)))
            x2 = min(width, int(round((center_x + box_width * 0.5) * width)))
            y2 = min(height, int(round((center_y + box_height * 0.5) * height)))
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            if min(pixel_width, pixel_height) < min_side:
                continue
            quality = min(pixel_width, pixel_height) * math.sqrt(pixel_width * pixel_height)
            candidates[identity].append(
                Candidate(
                    identity=identity,
                    image_path=image_path,
                    label_index=label_index,
                    bbox_xyxy=(x1, y1, x2, y2),
                    quality=quality,
                    recording_key=_recording_key(image_path.stem),
                )
            )
    return candidates


def select_diverse(candidates: list[Candidate], limit: int) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.recording_key].append(candidate)
    for values in groups.values():
        values.sort(key=lambda item: item.quality, reverse=True)

    selected: list[Candidate] = []
    ordered_groups = sorted(groups, key=lambda key: groups[key][0].quality, reverse=True)
    while len(selected) < limit:
        added = False
        for key in ordered_groups:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
                added = True
        if not added:
            break
    return selected


def expanded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    add_x = int(round((x2 - x1) * margin))
    add_y = int(round((y2 - y1) * margin))
    return (
        max(0, x1 - add_x),
        max(0, y1 - add_y),
        min(width, x2 + add_x),
        min(height, y2 + add_y),
    )


def main() -> None:
    args = parse_args()
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")
    if not 0.0 <= args.margin <= 1.0:
        raise ValueError("--margin must be between 0 and 1")

    config_path = Path(args.data).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _dataset_root(config_path, config)
    split_value = config.get(args.split)
    if not isinstance(split_value, str):
        raise ValueError(f"Split {args.split!r} must be a single directory path")
    image_root, label_root = _split_roots(root, split_value)
    names = _class_names(config.get("names"))

    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = collect_candidates(image_root, label_root, names, args.min_side)
    manifest_records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for identity in sorted(all_candidates, key=lambda value: int(value[1:])):
        selected = select_diverse(all_candidates[identity], args.per_class)
        identity_dir = output_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for index, candidate in enumerate(selected, start=1):
            image = cv2.imread(str(candidate.image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            box = expanded_box(candidate.bbox_xyxy, width, height, args.margin)
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            output_path = identity_dir / f"{index:03d}_{candidate.image_path.stem}.jpg"
            if not cv2.imwrite(str(output_path), crop):
                continue
            saved += 1
            manifest_records.append(
                {
                    "identity": identity,
                    "reference": str(output_path.relative_to(output_dir)),
                    "source_image": str(candidate.image_path),
                    "bbox_xyxy": list(box),
                    "quality": round(candidate.quality, 3),
                }
            )
        counts[identity] = saved

    manifest = {
        "schema": "yolo_face_references.v1",
        "data_yaml": str(config_path),
        "split": args.split,
        "output_dir": str(output_dir),
        "per_class_limit": args.per_class,
        "identity_counts": counts,
        "references": manifest_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "references"}, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
