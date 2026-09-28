#!/usr/bin/env python3
"""YOLO stage-1 gateway: output boxes plus b0/f/k major classes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.common.crops import KNIFE_CROP_CONTEXT_SCALE, square_context_crop


DEFAULT_WEIGHTS = Path(
    "/home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.pt"
)
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
COLORS = {
    "b0": (64, 180, 255),
    "f": (24, 180, 90),
    "k": (220, 80, 80),
    "unknown": (150, 150, 150),
}
ROUTES = {
    "b0": "b0_region",
    "f": "face_recognition_region",
    "k": "knife_recognition_region",
    "unknown": "review_region",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO as a first-stage detector and hand off b0/f/k regions."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", required=True, help="Image, directory, video, or camera index.")
    parser.add_argument("--out", default="outputs/stage1_gateway")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show", action="store_true", help="Show live preview.")
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame for video/camera input.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames for video/camera input.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not save per-detection crops for downstream modules.",
    )
    parser.add_argument(
        "--show-raw-label",
        action="store_true",
        help="Draw raw YOLO subclass next to the major class for debugging.",
    )
    return parser.parse_args()


def names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names or [])}


def major_class(raw_name: str) -> str:
    name = raw_name.strip().lower()
    if name == "b0":
        return "b0"
    if re.fullmatch(r"f\d+", name):
        return "f"
    if re.fullmatch(r"k\d+", name):
        return "k"
    return "unknown"


def predict_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device != "auto":
        kwargs["device"] = args.device
    return kwargs


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_detection(
    frame: Any,
    box: tuple[int, int, int, int],
    major: str,
    confidence: float,
    raw_name: str,
    show_raw_label: bool,
) -> None:
    x1, y1, x2, y2 = box
    color = COLORS.get(major, COLORS["unknown"])
    label = f"{major} {confidence:.2f}"
    if show_raw_label:
        label = f"{label} ({raw_name})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    y_text = max(th + baseline + 4, y1)
    cv2.rectangle(
        frame,
        (x1, y_text - th - baseline - 6),
        (x1 + tw + 8, y_text + baseline - 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def crop_path_for(major: str, frame_index: int, det_index: int) -> Path:
    return Path("crops") / major / f"frame{frame_index:06d}_det{det_index:03d}.jpg"


def build_detections(
    result: Any,
    names: dict[int, str],
    frame: Any,
    out_dir: Path,
    frame_index: int,
    save_crops: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = frame.shape[:2]
    boxes = result.boxes
    handoff: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    if boxes is None or len(boxes) == 0:
        return handoff, debug

    for det_index, (xyxy, conf, cls_id) in enumerate(
        zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
    ):
        class_id = int(cls_id)
        raw_name = names.get(class_id, str(class_id))
        major = major_class(raw_name)
        box = clamp_box(xyxy, width, height)
        x1, y1, x2, y2 = box
        crop_rel_path: Path | None = None
        if save_crops and x2 > x1 and y2 > y1:
            crop_rel_path = crop_path_for(major, frame_index, det_index)
            crop_abs_path = out_dir / crop_rel_path
            crop_abs_path.parent.mkdir(parents=True, exist_ok=True)
            crop = frame[y1:y2, x1:x2]
            if major == "k":
                crop = square_context_crop(
                    frame,
                    box,
                    context_scale=KNIFE_CROP_CONTEXT_SCALE,
                )
            cv2.imwrite(str(crop_abs_path), crop)

        stage_record = {
            "id": f"frame{frame_index:06d}:det{det_index:03d}",
            "major_class": major,
            "target_module": ROUTES.get(major, ROUTES["unknown"]),
            "confidence": round(float(conf), 6),
            "bbox_xyxy": [x1, y1, x2, y2],
            "crop_path": str(crop_rel_path) if crop_rel_path else None,
        }
        handoff.append(stage_record)
        debug.append(
            {
                **stage_record,
                "raw_yolo_class_id": class_id,
                "raw_yolo_class": raw_name,
            }
        )
    return handoff, debug


def process_frame(
    model: YOLO,
    names: dict[int, str],
    frame: Any,
    args: argparse.Namespace,
    out_dir: Path,
    frame_index: int,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    result = model.predict(frame, **predict_kwargs(args))[0]
    handoff, debug = build_detections(
        result=result,
        names=names,
        frame=frame,
        out_dir=out_dir,
        frame_index=frame_index,
        save_crops=not args.no_crops,
    )
    annotated = frame.copy()
    for record in debug:
        draw_detection(
            annotated,
            tuple(record["bbox_xyxy"]),
            record["major_class"],
            record["confidence"],
            record["raw_yolo_class"],
            args.show_raw_label,
        )
    return annotated, handoff, debug


def image_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return [source]


def write_json_outputs(
    out_dir: Path,
    weights: Path,
    source: str,
    started_at: str,
    frames_handoff: list[dict[str, Any]],
    frames_debug: list[dict[str, Any]],
    output_media: list[str],
) -> None:
    major_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    for frame in frames_handoff:
        major_counts.update(d["major_class"] for d in frame["detections"])
    for frame in frames_debug:
        raw_counts.update(d["raw_yolo_class"] for d in frame["detections"])

    handoff_payload = {
        "schema": "stage1_yolo_handoff.v1",
        "model": str(weights),
        "started_at": started_at,
        "base_dir": str(out_dir),
        "major_classes": ["b0", "f", "k"],
        "routes": ROUTES,
        "crop_policies": {
            "b0": {"shape": "bbox"},
            "f": {"shape": "bbox"},
            "k": {
                "shape": "square_context",
                "context_scale": KNIFE_CROP_CONTEXT_SCALE,
            },
        },
        "frames": frames_handoff,
    }
    debug_payload = {
        "schema": "stage1_yolo_debug.v1",
        "model": str(weights),
        "source": source,
        "started_at": started_at,
        "frames": frames_debug,
    }
    summary = {
        "output_dir": str(out_dir),
        "frames": len(frames_handoff),
        "detections": sum(major_counts.values()),
        "major_counts": dict(major_counts.most_common()),
        "raw_yolo_counts": dict(raw_counts.most_common()),
        "output_media": output_media,
        "handoff_json": str(out_dir / "stage1_handoff.json"),
        "debug_json": str(out_dir / "stage1_debug.json"),
    }
    (out_dir / "stage1_handoff.json").write_text(
        json.dumps(handoff_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "stage1_debug.json").write_text(
        json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_images(
    model: YOLO,
    names: dict[int, str],
    source: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    frame_handoff: list[dict[str, Any]] = []
    frame_debug: list[dict[str, Any]] = []
    output_media: list[str] = []
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, image_path in enumerate(image_sources(source)):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        annotated, handoff, debug = process_frame(
            model, names, frame, args, out_dir, frame_index
        )
        out_path = annotated_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)
        output_media.append(str(out_path))
        frame_handoff.append(
            {
                "frame_index": frame_index,
                "image_shape": list(frame.shape[:2]),
                "detections": handoff,
            }
        )
        frame_debug.append(
            {
                "frame_index": frame_index,
                "source": str(image_path),
                "image_shape": list(frame.shape[:2]),
                "detections": debug,
            }
        )
    return frame_handoff, frame_debug, output_media


def process_video_or_camera(
    model: YOLO,
    names: dict[int, str],
    source: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    capture_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = out_dir / "stage1_macro.mp4"
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video: {out_video}")

    frame_handoff: list[dict[str, Any]] = []
    frame_debug: list[dict[str, Any]] = []
    raw_index = 0
    processed_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_index % max(args.vid_stride, 1) != 0:
                raw_index += 1
                continue
            annotated, handoff, debug = process_frame(
                model, names, frame, args, out_dir, raw_index
            )
            writer.write(annotated)
            frame_handoff.append(
                {
                    "frame_index": raw_index,
                    "processed_index": processed_index,
                    "image_shape": list(frame.shape[:2]),
                    "detections": handoff,
                }
            )
            frame_debug.append(
                {
                    "frame_index": raw_index,
                    "processed_index": processed_index,
                    "source": source,
                    "image_shape": list(frame.shape[:2]),
                    "detections": debug,
                }
            )
            if args.show:
                cv2.imshow("stage1 yolo gateway", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed_index += 1
            raw_index += 1
            if args.max_frames and processed_index >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    return frame_handoff, frame_debug, [str(out_video)]


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    model = YOLO(str(weights))
    names = names_map(model.names)
    source_path = Path(args.source).expanduser()
    source_suffix = source_path.suffix.lower()

    if source_path.exists() and (source_path.is_dir() or source_suffix in IMAGE_EXTS):
        frames_handoff, frames_debug, output_media = process_images(
            model, names, source_path, args, out_dir
        )
    elif args.source.isdigit() or source_suffix in VIDEO_EXTS:
        frames_handoff, frames_debug, output_media = process_video_or_camera(
            model, names, args.source, args, out_dir
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    write_json_outputs(
        out_dir=out_dir,
        weights=weights,
        source=args.source,
        started_at=started_at,
        frames_handoff=frames_handoff,
        frames_debug=frames_debug,
        output_media=output_media,
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Model: {weights}")
    print(f"Source: {args.source}")
    print(f"Output: {out_dir}")
    print(f"Frames processed: {summary['frames']}")
    print(f"Detections: {summary['detections']}")
    print("Major counts:")
    for name, count in summary["major_counts"].items():
        print(f"  {name}: {count}")
    print(f"Handoff JSON: {summary['handoff_json']}")
    print(f"Debug JSON: {summary['debug_json']}")
    for media in output_media:
        print(f"Media: {media}")


if __name__ == "__main__":
    main()
