#!/usr/bin/env python3
"""Run inference with the latest trained YOLO model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(
    "/home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.pt"
)
DEFAULT_SOURCE = Path("/home/j/trainv5/images/test")
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the trained trainv5_continue_latest-2 YOLO model."
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Model weights path. Defaults to the latest trainv5 best.pt.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Image, directory, video path, camera index such as 0, or stream URL.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, 0, 0,1, etc.",
    )
    parser.add_argument(
        "--out",
        default="outputs/latest_model",
        help="Output directory for annotated media and JSON results.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="For image directories, test only the first N images.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not save YOLO txt label files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window while running inference.",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth video frame. Use 2 or 3 for faster live checks.",
    )
    return parser.parse_args()


def resolve_source(source: str, max_images: int | None) -> str | list[str]:
    path = Path(source).expanduser()
    if max_images and path.is_dir():
        images = [
            p
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        return [str(p) for p in images[:max_images]]
    return source


def names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names or [])}


def box_records(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()

    records = []
    for coords, conf, cls_id in zip(xyxy, confs, classes):
        cls_int = int(cls_id)
        records.append(
            {
                "class_id": cls_int,
                "class_name": names.get(cls_int, str(cls_int)),
                "confidence": round(float(conf), 6),
                "xyxy": [round(float(v), 2) for v in coords],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    names = names_map(model.names)
    source = resolve_source(args.source, args.max_images)

    predict_kwargs: dict[str, Any] = {
        "source": source,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save": True,
        "save_txt": not args.no_labels,
        "save_conf": not args.no_labels,
        "project": str(out_dir.parent),
        "name": out_dir.name,
        "exist_ok": True,
        "stream": True,
        "verbose": False,
        "show": args.show,
        "vid_stride": args.vid_stride,
    }
    if args.device != "auto":
        predict_kwargs["device"] = args.device

    started_at = datetime.now().isoformat(timespec="seconds")
    class_counts: Counter[str] = Counter()
    frames: list[dict[str, Any]] = []
    save_dir = out_dir

    for result in model.predict(**predict_kwargs):
        save_dir = Path(result.save_dir)
        detections = box_records(result, names)
        class_counts.update(d["class_name"] for d in detections)
        frames.append(
            {
                "source": str(result.path),
                "image_shape": list(result.orig_shape),
                "detections": detections,
            }
        )

    payload = {
        "model": str(weights),
        "source": args.source,
        "started_at": started_at,
        "parameters": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "max_images": args.max_images,
            "show": args.show,
            "vid_stride": args.vid_stride,
        },
        "classes": names,
        "frames": frames,
    }

    predictions_path = save_dir / "predictions.json"
    summary_path = save_dir / "summary.json"
    predictions_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "output_dir": str(save_dir),
        "frames": len(frames),
        "detections": sum(class_counts.values()),
        "detections_by_class": dict(class_counts.most_common()),
        "predictions_json": str(predictions_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Model: {weights}")
    print(f"Source: {args.source}")
    print(f"Output: {save_dir}")
    print(f"Frames/images processed: {summary['frames']}")
    print(f"Detections: {summary['detections']}")
    if class_counts:
        print("Detections by class:")
        for name, count in class_counts.most_common():
            print(f"  {name}: {count}")
    print(f"JSON: {predictions_path}")


if __name__ == "__main__":
    main()
