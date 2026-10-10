#!/usr/bin/env python3
"""Draw polygon regions on a D455 view and summarize pixel sizes/ranges."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pyrealsense2 as rs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_NAME = "D455 distance and region recorder"
PREVIEW_WIDTH = 960
SIDEBAR_WIDTH = 430
DEFAULT_COLOR_WIDTH = 1280
DEFAULT_COLOR_HEIGHT = 720
DEFAULT_DEPTH_WIDTH = 848
DEFAULT_DEPTH_HEIGHT = 480
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "best.pt"
TABLE_TOP = 190
TABLE_ROW_HEIGHT = 25
TABLE_COLUMNS = (
    ("id", 34), ("x1", 40), ("y1", 40), ("x2", 40), ("y2", 40),
    ("W", 40), ("H", 40), ("A", 54), ("R", 65),
)
POINT_CLOSE_RADIUS_PX = 10


@dataclass
class BoxRecord:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width_px: int
    height_px: int
    area_px2: int
    depth_z_m: float | None
    range_m: float | None
    camera_x_m: float | None
    camera_y_m: float | None
    valid_depth_samples: int
    sampled_depth_pixels: int
    # Polygon vertices in aligned color/depth pixel coordinates.  The name of
    # the record remains BoxRecord for compatibility with the existing JSON
    # and sidebar, while area_px2 now describes the selected polygon pixels.
    polygon: list[tuple[int, int]] = field(default_factory=list)
    # Filled automatically from the YOLO box that overlaps this selection.
    # It remains null when the selection does not overlap an f/k/b0 detection.
    kind: str | None = None
    yolo_label: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw polygon regions in a D455 color view and average their size/range."
    )
    parser.add_argument("--serial", help="RealSense serial number, if multiple cameras are connected")
    parser.add_argument("--fps", type=int, default=15, help="Camera stream FPS (default: 15)")
    parser.add_argument(
        "--width", type=int, default=DEFAULT_COLOR_WIDTH,
        help=f"Color stream width (default: {DEFAULT_COLOR_WIDTH})",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_COLOR_HEIGHT,
        help=f"Color stream height (default: {DEFAULT_COLOR_HEIGHT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: outputs/distance/distance_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--yolo", action="store_true",
        help="Draw first-stage YOLO boxes over the camera view while marking regions",
    )
    parser.add_argument(
        "--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS,
        help=f"YOLO .pt weights (default: {DEFAULT_YOLO_WEIGHTS})",
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=0.25,
        help="YOLO confidence threshold when --yolo is enabled (default: 0.25)",
    )
    parser.add_argument(
        "--yolo-iou", type=float, default=0.45,
        help="YOLO NMS IoU threshold when --yolo is enabled (default: 0.45)",
    )
    parser.add_argument(
        "--yolo-imgsz", type=int, default=640,
        help="YOLO inference image size when --yolo is enabled (default: 640)",
    )
    parser.add_argument(
        "--yolo-device", default="auto",
        help="YOLO device, for example auto, cpu, or 0 (default: auto)",
    )
    parser.add_argument(
        "--yolo-hold-frames", type=int, default=6,
        help="Keep a briefly missed b0 box for this many frames (default: 6)",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--fps, --width, and --height must be positive")
    if not 0.0 <= args.yolo_conf <= 1.0 or not 0.0 <= args.yolo_iou <= 1.0:
        parser.error("--yolo-conf and --yolo-iou must be between 0 and 1")
    if args.yolo_imgsz <= 0:
        parser.error("--yolo-imgsz must be positive")
    if args.yolo_hold_frames < 0:
        parser.error("--yolo-hold-frames must be non-negative")
    return args


def load_yolo(args: argparse.Namespace) -> Any | None:
    if not args.yolo:
        return None
    weights = args.yolo_weights.expanduser().resolve()
    if not weights.is_file():
        raise RuntimeError(f"YOLO weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "--yolo requires the Python ultralytics package"
        ) from error
    print(f"Loading YOLO overlay model: {weights}")
    return YOLO(str(weights))


def draw_yolo_overlay(
    color: np.ndarray,
    model: Any,
    args: argparse.Namespace,
    previous_detections: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw first-stage YOLO boxes without changing the source pixel geometry."""

    kwargs: dict[str, Any] = {
        "conf": args.yolo_conf,
        "iou": args.yolo_iou,
        "imgsz": args.yolo_imgsz,
        "verbose": False,
    }
    if args.yolo_device != "auto":
        kwargs["device"] = args.yolo_device
    result = model.predict(color, **kwargs)[0]
    annotated = color.copy()
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    names = result.names
    if boxes is not None and len(boxes) > 0:
      for xyxy, confidence, class_id in zip(
          boxes.xyxy.cpu().tolist(),
          boxes.conf.cpu().tolist(),
          boxes.cls.cpu().tolist(),
      ):
          x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
          x1 = max(0, min(annotated.shape[1] - 1, x1))
          y1 = max(0, min(annotated.shape[0] - 1, y1))
          x2 = max(0, min(annotated.shape[1] - 1, x2))
          y2 = max(0, min(annotated.shape[0] - 1, y2))
          if x2 < x1:
              x1, x2 = x2, x1
          if y2 < y1:
              y1, y2 = y2, y1
          raw_name = str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(class_id)
          lowered = raw_name.lower()
          major = "b0" if lowered == "b0" else "f" if lowered.startswith("f") else "k" if lowered.startswith("k") else "unknown"
          detections.append({
              "bbox": (x1, y1, x2, y2),
              "major": major,
              "label": raw_name,
              "confidence": float(confidence),
          })
    # A close obstacle can briefly fall below the confidence threshold as the
    # camera or object moves.  Hold and smooth only b0 boxes so calibration
    # remains usable without hiding genuine person/knife changes.
    previous = previous_detections or []
    used_previous: set[int] = set()
    smoothed: list[dict[str, Any]] = []
    for detection in detections:
        if detection["major"] != "b0":
            smoothed.append(detection)
            continue
        best_index = -1
        best_iou = 0.0
        x1, y1, x2, y2 = detection["bbox"]
        for index, old in enumerate(previous):
            if index in used_previous or old.get("major") != "b0":
                continue
            ox1, oy1, ox2, oy2 = old["bbox"]
            intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(
                0, min(y2, oy2) - max(y1, oy1)
            )
            union = max(1, (x2 - x1) * (y2 - y1) +
                        (ox2 - ox1) * (oy2 - oy1) - intersection)
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= 0.15:
            old = previous[best_index]
            used_previous.add(best_index)
            old_box = old["bbox"]
            detection["bbox"] = tuple(
                round(0.7 * current + 0.3 * prior)
                for current, prior in zip(detection["bbox"], old_box)
            )
        detection["missing_frames"] = 0
        detection["held"] = False
        smoothed.append(detection)
    for index, old in enumerate(previous):
        if index in used_previous or old.get("major") != "b0":
            continue
        missing = int(old.get("missing_frames", 0)) + 1
        if missing <= args.yolo_hold_frames:
            held = dict(old)
            held["missing_frames"] = missing
            held["held"] = True
            held["confidence"] = float(old.get("confidence", 0.0))
            smoothed.append(held)
    detections = smoothed

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        major = detection["major"]
        raw_name = detection["label"]
        confidence = float(detection["confidence"])
        color_code = {
            "b0": (64, 180, 255),
            "f": (24, 180, 90),
            "k": (220, 80, 80),
            "unknown": (150, 150, 150),
        }[major]
        label = f"{raw_name} {confidence:.2f}"
        if detection.get("held"):
            label += " hold"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_code, 2)
        cv2.putText(
            annotated, label, (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_code, 1, cv2.LINE_AA,
        )
    return annotated, detections


def measure_box(
    depth_image: np.ndarray,
    box: tuple[int, int, int, int],
    depth_scale: float,
    intrinsics: Any,
) -> dict[str, float | int | None]:
    """Use a central ROI median and calibrated intrinsics for a range sample."""
    x1, y1, x2, y2 = box
    image_height, image_width = depth_image.shape
    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))
    if x2 <= x1 or y2 <= y1:
        return {
            "depth_z_m": None,
            "range_m": None,
            "camera_x_m": None,
            "camera_y_m": None,
            "valid_depth_samples": 0,
            "sampled_depth_pixels": 0,
        }

    inset_x = int((x2 - x1) * 0.2)
    inset_y = int((y2 - y1) * 0.2)
    left = min(x2 - 1, x1 + inset_x)
    right = max(left + 1, x2 - inset_x)
    top = min(y2 - 1, y1 + inset_y)
    bottom = max(top + 1, y2 - inset_y)
    step_x = max(1, (right - left) // 25)
    step_y = max(1, (bottom - top) // 25)
    sampled = depth_image[top:bottom:step_y, left:right:step_x]
    depths = sampled.astype(np.float32) * depth_scale
    valid = depths[np.isfinite(depths) & (depths >= 0.2) & (depths <= 10.0)]
    valid_count = int(valid.size)
    sampled_count = int(sampled.size)
    if valid_count < 5 or valid_count / max(sampled_count, 1) < 0.05:
        return {
            "depth_z_m": None,
            "range_m": None,
            "camera_x_m": None,
            "camera_y_m": None,
            "valid_depth_samples": valid_count,
            "sampled_depth_pixels": sampled_count,
        }

    depth_z = float(np.median(valid))
    center = [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)]
    point = rs.rs2_deproject_pixel_to_point(intrinsics, center, depth_z)
    camera_x, camera_y, depth_z = (float(value) for value in point)
    return {
        "depth_z_m": depth_z,
        "range_m": math.sqrt(camera_x**2 + camera_y**2 + depth_z**2),
        "camera_x_m": camera_x,
        "camera_y_m": camera_y,
        "valid_depth_samples": valid_count,
        "sampled_depth_pixels": sampled_count,
    }


def empty_measurement() -> dict[str, float | int | None]:
    return {
        "depth_z_m": None,
        "range_m": None,
        "camera_x_m": None,
        "camera_y_m": None,
        "valid_depth_samples": 0,
        "sampled_depth_pixels": 0,
    }


def polygon_bounds(
    polygon: list[tuple[int, int]], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Return an exclusive pixel bounding box for a polygon."""

    if len(polygon) < 3:
        return None
    xs = [max(0, min(width - 1, int(point[0]))) for point in polygon]
    ys = [max(0, min(height - 1, int(point[1]))) for point in polygon]
    x1, x2 = min(xs), max(xs) + 1
    y1, y2 = min(ys), max(ys) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def polygon_pixel_area(polygon: list[tuple[int, int]], width: int, height: int) -> int:
    """Count the pixels inside a closed polygon in image coordinates."""

    bounds = polygon_bounds(polygon, width, height)
    if bounds is None:
        return 0
    x1, y1, x2, y2 = bounds
    local = np.asarray(
        [(int(x) - x1, int(y) - y1) for x, y in polygon], dtype=np.int32
    )
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    return int(cv2.countNonZero(mask))


def polygon_center(polygon: list[tuple[int, int]]) -> tuple[float, float]:
    """Return the geometric centroid used for calibrated deprojection."""

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(points)
    if abs(float(moments["m00"])) > 1e-6:
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    values = np.asarray(polygon, dtype=np.float32)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def measure_polygon(
    depth_image: np.ndarray,
    polygon: list[tuple[int, int]],
    depth_scale: float,
    intrinsics: Any,
) -> dict[str, float | int | None]:
    """Measure valid aligned depth samples inside an arbitrary polygon.

    The depth median is taken only from pixels covered by the polygon.  A
    bounded, evenly spaced sample keeps a large polygon responsive while
    preserving the same valid-depth filtering used by ``measure_box``.
    """

    if len(polygon) < 3 or depth_image.ndim != 2:
        return empty_measurement()
    image_height, image_width = depth_image.shape
    bounds = polygon_bounds(polygon, image_width, image_height)
    if bounds is None:
        return empty_measurement()
    x1, y1, x2, y2 = bounds
    local = np.asarray(
        [(int(x) - x1, int(y) - y1) for x, y in polygon], dtype=np.int32
    )
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return empty_measurement()
    # The raster centroid is guaranteed to lie in the selected pixel region,
    # including for concave polygons.  It is therefore safer for deprojection
    # than assuming the mathematical polygon centroid is inside the region.
    center_x = float(x1 + np.mean(xs))
    center_y = float(y1 + np.mean(ys))

    # A maximum sample count bounds work for a full-frame polygon without
    # changing the selected pixel area or the polygon shown to the user.
    max_samples = 12000
    if xs.size > max_samples:
        indices = np.linspace(0, xs.size - 1, max_samples, dtype=np.int64)
        xs = xs[indices]
        ys = ys[indices]
    sampled = depth_image[y1 + ys, x1 + xs]
    depths = sampled.astype(np.float32) * depth_scale
    valid = depths[np.isfinite(depths) & (depths >= 0.2) & (depths <= 10.0)]
    valid_count = int(valid.size)
    sampled_count = int(sampled.size)
    if valid_count < 5 or valid_count / max(sampled_count, 1) < 0.05:
        return {
            **empty_measurement(),
            "valid_depth_samples": valid_count,
            "sampled_depth_pixels": sampled_count,
        }

    depth_z = float(np.median(valid))
    center = [
        max(0.0, min(float(image_width - 1), center_x)),
        max(0.0, min(float(image_height - 1), center_y)),
    ]
    point = rs.rs2_deproject_pixel_to_point(intrinsics, center, depth_z)
    camera_x, camera_y, depth_z = (float(value) for value in point)
    return {
        "depth_z_m": depth_z,
        "range_m": math.sqrt(camera_x**2 + camera_y**2 + depth_z**2),
        "camera_x_m": camera_x,
        "camera_y_m": camera_y,
        "valid_depth_samples": valid_count,
        "sampled_depth_pixels": sampled_count,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[BoxRecord]) -> dict[str, float | int | None]:
    ranges = [record.range_m for record in records if record.range_m is not None]
    depths = [record.depth_z_m for record in records if record.depth_z_m is not None]
    return {
        "box_count": len(records),
        "average_width_px": mean([float(record.width_px) for record in records]),
        "average_height_px": mean([float(record.height_px) for record in records]),
        "average_area_px2": mean([float(record.area_px2) for record in records]),
        "valid_range_count": len(ranges),
        "average_range_m": mean(ranges),
        "average_depth_z_m": mean(depths),
    }


def format_value(value: float | int | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value}{unit}"
    return f"{value:.2f}{unit}"


class DistanceRecorder:
    def __init__(
        self,
        width: int,
        height: int,
        depth_stream_size: tuple[int, int] =
        (DEFAULT_DEPTH_WIDTH, DEFAULT_DEPTH_HEIGHT),
    ) -> None:
        self.width = width
        self.height = height
        self.depth_stream_size = depth_stream_size
        self.aligned_depth_size = (width, height)
        self.records: list[BoxRecord] = []
        self.next_id = 1
        self.selected_id: int | None = None
        self.pending_points: list[tuple[int, int]] = []
        self.pending_cursor: tuple[int, int] | None = None
        self.last_message = "Right click points; click the first point to close a region"
        self.editing_id: int | None = None
        self.editing_field: str | None = None
        self.edit_buffer = ""
        self.confirmed_average: dict[str, float | int | None] | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_depth_scale = 0.001
        self.latest_intrinsics: Any = None
        self.latest_yolo_detections: list[dict[str, Any]] = []
        # Installed by main() once camera metadata and the output path exist.
        # It persists edits, including an unfinished polygon, immediately.
        self.on_change: Callable[[], None] | None = None

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def to_source_point(self, x: int, y: int, preview_width: int, preview_height: int) -> tuple[int, int]:
        source_x = round(x * self.width / max(preview_width, 1))
        source_y = round(y * self.height / max(preview_height, 1))
        return (
            max(0, min(self.width - 1, source_x)),
            max(0, min(self.height - 1, source_y)),
        )

    def _record_at_point(self, point: tuple[int, int]) -> BoxRecord | None:
        """Select the smallest completed polygon containing a pixel."""

        hits: list[BoxRecord] = []
        for record in self.records:
            if record.polygon:
                contour = np.asarray(record.polygon, dtype=np.float32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                    hits.append(record)
            elif record.x1 <= point[0] < record.x2 and record.y1 <= point[1] < record.y2:
                hits.append(record)
        return min(hits, key=lambda record: record.area_px2) if hits else None

    def set_yolo_detections(self, detections: list[dict[str, Any]]) -> None:
        self.latest_yolo_detections = detections

    def _yolo_match(
        self, polygon: list[tuple[int, int]]
    ) -> tuple[str | None, str | None]:
        """Match a drawn region to the current YOLO box by rectangle overlap."""

        bounds = polygon_bounds(polygon, self.width, self.height)
        if bounds is None:
            return None, None
        x1, y1, x2, y2 = bounds
        selected_area = max(1, (x2 - x1) * (y2 - y1))
        best: tuple[float, str, str] | None = None
        for detection in self.latest_yolo_detections:
            major = str(detection.get("major", ""))
            if major not in {"f", "k", "b0"}:
                continue
            box = detection.get("bbox")
            if not box or len(box) != 4:
                continue
            dx1, dy1, dx2, dy2 = (int(value) for value in box)
            detection_area = max(1, (dx2 - dx1) * (dy2 - dy1))
            intersection = max(0, min(x2, dx2) - max(x1, dx1)) * max(
                0, min(y2, dy2) - max(y1, dy1)
            )
            if intersection <= 0:
                continue
            union = selected_area + detection_area - intersection
            iou = intersection / max(union, 1)
            detection_coverage = intersection / detection_area
            # A user-drawn box may be slightly larger than YOLO's box; require
            # at least a quarter of the YOLO box to be inside the selection.
            if detection_coverage < 0.25:
                continue
            score = iou + 0.5 * detection_coverage
            candidate = (score, major, str(detection.get("label", major)))
            if best is None or candidate[0] > best[0]:
                best = candidate
        return (best[1], best[2]) if best else (None, None)

    def _close_polygon(
        self, depth_image: np.ndarray, depth_scale: float, intrinsics: Any
    ) -> None:
        if len(self.pending_points) < 3:
            self.last_message = "A region needs at least 3 points"
            return
        polygon = list(self.pending_points)
        bounds = polygon_bounds(polygon, self.width, self.height)
        if bounds is None:
            self.last_message = "The selected points do not enclose a region"
            return
        x1, y1, x2, y2 = bounds
        area = polygon_pixel_area(polygon, self.width, self.height)
        if area <= 0:
            self.last_message = "The selected points do not enclose any pixels"
            return
        dimensions = measure_polygon(depth_image, polygon, depth_scale, intrinsics)
        kind, yolo_label = self._yolo_match(polygon)
        record = BoxRecord(
            id=self.next_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width_px=x2 - x1,
            height_px=y2 - y1,
            area_px2=area,
            polygon=polygon,
            kind=kind,
            yolo_label=yolo_label,
            **dimensions,
        )
        self.records.append(record)
        self.selected_id = record.id
        self.next_id += 1
        self.pending_points.clear()
        self.pending_cursor = None
        tag = f" {kind}/{yolo_label}" if kind else " (no YOLO match)"
        self.last_message = f"Added region #{record.id}{tag}; right click to start another"
        self._changed()
        print(
            f"Added #{record.id}: bbox={record.width_px}x{record.height_px} px, "
            f"polygon_area={record.area_px2} px^2, "
            f"kind={record.kind or 'unmatched'} "
            f"label={record.yolo_label or 'n/a'}, "
            f"range={format_value(record.range_m, ' m')}"
        )

    def mouse(
        self,
        event: int,
        x: int,
        y: int,
        preview_size: tuple[int, int],
        depth_image: np.ndarray,
        depth_scale: float,
        intrinsics: Any,
    ) -> None:
        point = self.to_source_point(x, y, preview_size[0], preview_size[1])
        if event == cv2.EVENT_MOUSEMOVE:
            if self.pending_points:
                self.pending_cursor = point
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pending_points:
                removed = self.pending_points.pop()
                self.pending_cursor = point
                self.last_message = f"Undid point ({removed[0]}, {removed[1]})"
                if not self.pending_points:
                    self.pending_cursor = None
                self._changed()
                return
            selected = self._record_at_point(point)
            self.selected_id = selected.id if selected else None
            self.last_message = (
                f"Selected region #{selected.id}"
                if selected
                else "Right click to add the first point"
            )
            return
        if event != cv2.EVENT_RBUTTONDOWN:
            return

        if self.pending_points:
            first = self.pending_points[0]
            close_radius = POINT_CLOSE_RADIUS_PX
            distance = math.hypot(point[0] - first[0], point[1] - first[1])
            if distance <= close_radius:
                if len(self.pending_points) >= 3:
                    self._close_polygon(depth_image, depth_scale, intrinsics)
                else:
                    self.last_message = "A region needs at least 3 points before closing"
                return
            if point == self.pending_points[-1]:
                self.last_message = "Duplicate point ignored"
                return
        self.pending_points.append(point)
        self.pending_cursor = point
        self.last_message = f"Point {len(self.pending_points)} added; left click undoes"
        self._changed()

    def set_latest_depth(
        self, depth_image: np.ndarray, depth_scale: float, intrinsics: Any
    ) -> None:
        self.latest_depth = depth_image
        self.latest_depth_scale = depth_scale
        self.latest_intrinsics = intrinsics
        self.aligned_depth_size = (int(depth_image.shape[1]), int(depth_image.shape[0]))

    def _record(self, record_id: int | None) -> BoxRecord | None:
        if record_id is None:
            return None
        return next((record for record in self.records if record.id == record_id), None)

    def _clear_confirmation(self) -> None:
        self.confirmed_average = None

    def _remeasure(self, record: BoxRecord) -> None:
        record.width_px = record.x2 - record.x1
        record.height_px = record.y2 - record.y1
        record.area_px2 = (
            polygon_pixel_area(record.polygon, self.width, self.height)
            if record.polygon
            else record.width_px * record.height_px
        )
        if self.latest_depth is None or self.latest_intrinsics is None:
            record.depth_z_m = None
            record.range_m = None
            record.camera_x_m = None
            record.camera_y_m = None
            record.valid_depth_samples = 0
            record.sampled_depth_pixels = 0
            return
        dimensions = (
            measure_polygon(
                self.latest_depth,
                record.polygon,
                self.latest_depth_scale,
                self.latest_intrinsics,
            )
            if record.polygon
            else measure_box(
                self.latest_depth,
                (record.x1, record.y1, record.x2, record.y2),
                self.latest_depth_scale,
                self.latest_intrinsics,
            )
        )
        for key, value in dimensions.items():
            setattr(record, key, value)

    def begin_edit(self, field: str) -> None:
        if field not in {"x1", "y1", "x2", "y2", "W", "H"} or self.selected_id is None:
            return
        if self.editing_field is not None:
            self.commit_edit()
        self.editing_id = self.selected_id
        self.editing_field = field
        self.edit_buffer = ""

    def cancel_edit(self) -> None:
        self.editing_id = None
        self.editing_field = None
        self.edit_buffer = ""

    def commit_edit(self) -> bool:
        record = self._record(self.editing_id)
        field = self.editing_field
        if record is None or field is None:
            self.cancel_edit()
            return False
        if not self.edit_buffer:
            self.cancel_edit()
            return False
        try:
            value = int(self.edit_buffer)
        except ValueError:
            return False
        candidate = {"x1": record.x1, "y1": record.y1,
                     "x2": record.x2, "y2": record.y2}
        if field in {"x1", "y1", "x2", "y2"}:
            candidate[field] = value
        elif field == "W":
            candidate["x2"] = candidate["x1"] + value
        elif field == "H":
            candidate["y2"] = candidate["y1"] + value
        if not (0 <= candidate["x1"] < candidate["x2"] <= self.width and
                0 <= candidate["y1"] < candidate["y2"] <= self.height):
            return False
        if record.polygon:
            # Map inclusive polygon vertices to the new inclusive bounds so
            # the exclusive x2/y2 sidebar values remain geometrically exact.
            old_width = max(record.x2 - record.x1 - 1, 1)
            old_height = max(record.y2 - record.y1 - 1, 1)
            new_width = candidate["x2"] - candidate["x1"]
            new_height = candidate["y2"] - candidate["y1"]
            new_polygon = [
                (
                    max(
                        0,
                        min(
                            self.width - 1,
                            round(
                                candidate["x1"]
                                + (x - record.x1) * max(new_width - 1, 0) / old_width
                            ),
                        ),
                    ),
                    max(
                        0,
                        min(
                            self.height - 1,
                            round(
                                candidate["y1"]
                                + (y - record.y1) * max(new_height - 1, 0) / old_height
                            ),
                        ),
                    ),
                )
                for x, y in record.polygon
            ]
            if polygon_pixel_area(new_polygon, self.width, self.height) <= 0:
                return False
            record.polygon = new_polygon
        record.x1 = candidate["x1"]
        record.y1 = candidate["y1"]
        record.x2 = candidate["x2"]
        record.y2 = candidate["y2"]
        self._remeasure(record)
        self._clear_confirmation()
        self.cancel_edit()
        self._changed()
        return True

    def handle_edit_key(self, key: int) -> bool:
        if self.editing_field is None:
            return False
        if key in (13, 10):
            self.commit_edit()
        elif key == 27:
            self.cancel_edit()
        elif key in (8, 127):
            self.edit_buffer = self.edit_buffer[:-1]
        elif key == ord("-") and not self.edit_buffer:
            self.edit_buffer = "-"
        elif ord("0") <= key <= ord("9"):
            self.edit_buffer += chr(key)
        return True

    def confirm_average(self) -> None:
        if self.editing_field is not None:
            self.commit_edit()
        self.confirmed_average = summarize(self.records)
        print("Confirmed average of current boxes:")
        print(json.dumps(self.confirmed_average, ensure_ascii=False, indent=2))
        self._changed()

    def panel_mouse(self, event: int, x: int, y: int) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or x < PREVIEW_WIDTH:
            return
        panel_x = x - PREVIEW_WIDTH
        if 12 <= panel_x < 210 and 58 <= y < 94:
            self.confirm_average()
            return
        if 220 <= panel_x < 405 and 58 <= y < 94:
            self.clear()
            return
        row_index = (y - TABLE_TOP) // TABLE_ROW_HEIGHT
        if row_index < 0 or row_index >= len(self.records):
            return
        record = self.records[row_index]
        self.selected_id = record.id
        cell_x = 8
        for field, cell_width in TABLE_COLUMNS:
            if cell_x <= panel_x < cell_x + cell_width:
                self.begin_edit(field)
                return
            cell_x += cell_width

    def delete_selected(self) -> None:
        if self.selected_id is None:
            return
        old_count = len(self.records)
        self.records = [record for record in self.records if record.id != self.selected_id]
        if len(self.records) != old_count:
            print(f"Deleted box #{self.selected_id}")
        self.selected_id = None
        self.cancel_edit()
        self._clear_confirmation()
        if len(self.records) != old_count:
            self._changed()

    def undo(self) -> None:
        if self.pending_points:
            removed = self.pending_points.pop()
            self.pending_cursor = self.pending_points[-1] if self.pending_points else None
            self.last_message = f"Undid point ({removed[0]}, {removed[1]})"
            self._changed()
            return
        if self.records:
            record = self.records.pop()
            print(f"Undid box #{record.id}")
            self.last_message = f"Undid region #{record.id}"
        self.selected_id = None
        self.cancel_edit()
        self._clear_confirmation()
        self._changed()

    def clear(self) -> None:
        count = len(self.records)
        self.records.clear()
        self.pending_points.clear()
        self.pending_cursor = None
        self.selected_id = None
        self.cancel_edit()
        self._clear_confirmation()
        self.last_message = f"Cleared {count} completed region(s)"
        print(f"Cleared {count} box record(s)")
        self._changed()

    def draw(self, color: np.ndarray, fps: float) -> np.ndarray:
        preview_height = max(1, round(color.shape[0] * PREVIEW_WIDTH / color.shape[1]))
        preview = cv2.resize(color, (PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)
        scale_x = PREVIEW_WIDTH / self.width
        scale_y = preview_height / self.height
        for record in self.records:
            box = (
                round(record.x1 * scale_x),
                round(record.y1 * scale_y),
                round(record.x2 * scale_x),
                round(record.y2 * scale_y),
            )
            selected = record.id == self.selected_id
            color_code = (0, 220, 255) if selected else (50, 220, 90)
            if record.polygon:
                points = np.asarray(
                    [
                        (round(x * scale_x), round(y * scale_y))
                        for x, y in record.polygon
                    ],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    preview,
                    [points],
                    True,
                    color_code,
                    2 if selected else 1,
                    cv2.LINE_AA,
                )
                for point in points.reshape(-1, 2):
                    cv2.circle(
                        preview,
                        (int(point[0]), int(point[1])),
                        3,
                        color_code,
                        -1,
                        cv2.LINE_AA,
                    )
            else:
                cv2.rectangle(preview, box[:2], box[2:], color_code, 2 if selected else 1)
            range_text = format_value(record.range_m, "m")
            kind_text = record.yolo_label or record.kind or "?"
            label = f"#{record.id} {kind_text} A:{record.area_px2}px2 R:{range_text}"
            text_y = max(20, box[1] - 7)
            cv2.putText(preview, label, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color_code, 1, cv2.LINE_AA)

        if self.pending_points:
            pending = [
                (round(x * scale_x), round(y * scale_y))
                for x, y in self.pending_points
            ]
            for start, end in zip(pending, pending[1:]):
                cv2.line(preview, start, end, (255, 200, 0), 2, cv2.LINE_AA)
            for index, point in enumerate(pending):
                point_color = (0, 255, 255) if index == 0 else (255, 200, 0)
                cv2.circle(preview, point, 5, point_color, -1, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    str(index + 1),
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    point_color,
                    1,
                    cv2.LINE_AA,
                )
            if self.pending_cursor is not None:
                cursor = (
                    round(self.pending_cursor[0] * scale_x),
                    round(self.pending_cursor[1] * scale_y),
                )
                cv2.line(preview, pending[-1], cursor, (255, 180, 0), 1, cv2.LINE_AA)
                if len(self.pending_points) >= 3:
                    first = pending[0]
                    if math.hypot(cursor[0] - first[0], cursor[1] - first[1]) <= max(8, round(POINT_CLOSE_RADIUS_PX * scale_x)):
                        cv2.circle(preview, first, 9, (0, 255, 0), 2, cv2.LINE_AA)

        stats = summarize(self.records)
        rows = [
            f"D455 {fps:.1f} FPS | regions: {stats['box_count']}",
            f"average bbox: {format_value(stats['average_width_px'])} x "
            f"{format_value(stats['average_height_px'])} px | "
            f"R {format_value(stats['average_range_m'], ' m')}",
            "Right: add point/close | Left: undo/select",
        ]
        image_panel_height = preview.shape[0]
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 70), (20, 20, 20), -1)
        for index, text in enumerate(rows):
            cv2.putText(preview, text, (10, 19 + index * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (80, 230, 255) if index < 2 else (240, 240, 240),
                        1, cv2.LINE_AA)

        panel = np.full((image_panel_height, SIDEBAR_WIDTH, 3), 30, dtype=np.uint8)
        cv2.putText(panel, "DISTANCE / REGION DATA", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"color {self.width}x{self.height}  @ {fps:.1f} FPS",
                    (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (12, 58), (210, 94), (45, 125, 65), -1)
        cv2.putText(panel, "CONFIRM AVERAGE", (25, 81),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (220, 58), (405, 94), (100, 75, 55), -1)
        cv2.putText(panel, "CLEAR ALL", (270, 81),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"raw depth {self.depth_stream_size[0]}x{self.depth_stream_size[1]}  "
                    f"aligned {self.aligned_depth_size[0]}x{self.aligned_depth_size[1]}",
                    (12, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                    (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(panel, f"current avg: W {format_value(stats['average_width_px'])}  "
                    f"H {format_value(stats['average_height_px'])}  "
                    f"A {format_value(stats['average_area_px2'])}",
                    (12, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (120, 240, 150), 1, cv2.LINE_AA)
        cv2.putText(panel, f"R {format_value(stats['average_range_m'], 'm')}  "
                    f"Z {format_value(stats['average_depth_z_m'], 'm')}",
                    (12, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (120, 240, 150), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"pending points: {len(self.pending_points)} | {self.last_message}",
            (12, 177),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 220, 130),
            1,
            cv2.LINE_AA,
        )
        header_x = 8
        for field, cell_width in TABLE_COLUMNS:
            cv2.putText(panel, field, (header_x + 3, TABLE_TOP - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 220), 1, cv2.LINE_AA)
            header_x += cell_width
        for row_index, record in enumerate(self.records):
            y = TABLE_TOP + row_index * TABLE_ROW_HEIGHT
            selected = record.id == self.selected_id
            if selected:
                cv2.rectangle(panel, (4, y - 18), (SIDEBAR_WIDTH - 8, y + 7),
                              (65, 80, 105), -1)
            values = [
                str(record.id), str(record.x1), str(record.y1), str(record.x2),
                str(record.y2), str(record.width_px), str(record.height_px),
                str(record.area_px2), format_value(record.range_m, "m"),
            ]
            cell_x = 8
            for (field, cell_width), value in zip(TABLE_COLUMNS, values):
                if record.id == self.editing_id and field == self.editing_field:
                    value = self.edit_buffer or "_"
                cv2.putText(panel, value, (cell_x + 3, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (255, 230, 100) if selected else (230, 230, 230),
                            1, cv2.LINE_AA)
                cell_x += cell_width
        detail_y = min(image_panel_height - 105,
                       TABLE_TOP + max(len(self.records), 1) * TABLE_ROW_HEIGHT + 12)
        selected = self._record(self.selected_id)
        if self.editing_field is not None:
            detail = f"Editing #{self.editing_id} {self.editing_field}: " \
                     f"{self.edit_buffer or '(type number)'}"
            cv2.putText(panel, detail, (12, detail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 100), 1, cv2.LINE_AA)
            cv2.putText(panel, "Enter=apply  Esc=cancel  (edit coords or W/H)",
                        (12, detail_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (230, 230, 230), 1, cv2.LINE_AA)
        elif selected is not None:
            cv2.putText(panel, f"selected #{selected.id}: Z "
                        f"{format_value(selected.depth_z_m, 'm')}  "
                        f"R {format_value(selected.range_m, 'm')}",
                        (12, detail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 220, 100), 1, cv2.LINE_AA)
            cv2.putText(panel, "Click x1/y1/x2/y2/W/H cell to edit", (12, detail_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
        if self.confirmed_average is not None:
            confirmed = self.confirmed_average
            confirm_y = image_panel_height - 57
            cv2.rectangle(panel, (8, confirm_y - 18), (SIDEBAR_WIDTH - 8, confirm_y + 32),
                          (40, 100, 65), -1)
            cv2.putText(panel, "CONFIRMED AVERAGE", (15, confirm_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 170), 1, cv2.LINE_AA)
            cv2.putText(panel, f"W {format_value(confirmed['average_width_px'])} px  "
                        f"H {format_value(confirmed['average_height_px'])} px  "
                        f"R {format_value(confirmed['average_range_m'], 'm')}",
                        (15, confirm_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                        (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Click CONFIRM AVERAGE or press Enter", (12, image_panel_height - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 220, 200), 1, cv2.LINE_AA)

        canvas = np.zeros((image_panel_height, PREVIEW_WIDTH + SIDEBAR_WIDTH, 3), dtype=np.uint8)
        canvas[:, :PREVIEW_WIDTH] = preview
        canvas[:, PREVIEW_WIDTH:] = panel
        return canvas


def print_summary(records: list[BoxRecord]) -> None:
    print("Average of current boxes:")
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print("Current box records:")
    print(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2))


def save_report(
    output_path: Path,
    records: list[BoxRecord],
    camera_info: dict[str, Any],
    fps: int,
    color_size: tuple[int, int],
    confirmed_average: dict[str, float | int | None] | None = None,
    pending_points: list[tuple[int, int]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "insight.distance_boxes.v1",
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera": camera_info,
        "configured_fps": fps,
        "color_size": {"width": color_size[0], "height": color_size[1]},
        "average": summarize(records),
        "confirmed_average": confirmed_average,
        "boxes": [asdict(record) for record in records],
        # Raw source-image pixels clicked for the region currently being
        # drawn.  This is intentionally retained until the polygon closes.
        "pending_points": [list(point) for point in (pending_points or [])],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} box record(s) to {output_path}")
    return output_path


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "distance" / f"distance_{stamp}.json"


def main() -> int:
    args = parse_args()
    yolo_model = load_yolo(args)
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense camera found. Connect the D455 and retry.")
    if args.serial:
        serials = {
            device.get_info(rs.camera_info.serial_number) for device in devices
        }
        if args.serial not in serials:
            raise RuntimeError(f"RealSense serial not found: {args.serial}")
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, DEFAULT_DEPTH_WIDTH, DEFAULT_DEPTH_HEIGHT,
        rs.format.z16, args.fps
    )
    profile = pipeline.start(config)
    device = profile.get_device()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    actual_color_size = (color_profile.width(), color_profile.height())
    actual_depth_size = (depth_profile.width(), depth_profile.height())
    if actual_color_size != (args.width, args.height):
        raise RuntimeError(
            f"RealSense color stream is {actual_color_size[0]}x{actual_color_size[1]}, "
            f"but distance requested {args.width}x{args.height}"
        )
    camera_info = {
        "name": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "color_stream": {
            "width": actual_color_size[0], "height": actual_color_size[1],
            "fps": args.fps, "format": "bgr8",
        },
        "depth_stream": {
            "width": actual_depth_size[0], "height": actual_depth_size[1],
            "fps": args.fps, "format": "z16",
        },
    }
    depth_scale = device.first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    recorder = DistanceRecorder(args.width, args.height, actual_depth_size)
    window_size = (PREVIEW_WIDTH, max(1, round(args.height * PREVIEW_WIDTH / args.width)))
    preview_height = window_size[1]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH + SIDEBAR_WIDTH, preview_height)
    latest_depth: np.ndarray | None = None
    latest_depth_scale = depth_scale
    latest_intrinsics: Any = None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        if x >= PREVIEW_WIDTH:
            recorder.panel_mouse(event, x, y)
            return
        if latest_depth is None or latest_intrinsics is None:
            return
        recorder.mouse(event, x, y, window_size, latest_depth,
                       latest_depth_scale, latest_intrinsics)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(
        f"Camera: {camera_info['name']} serial={camera_info['serial']} "
        f"color={actual_color_size[0]}x{actual_color_size[1]}@{args.fps} FPS "
        f"raw_depth={actual_depth_size[0]}x{actual_depth_size[1]} "
        f"aligned_depth={actual_color_size[0]}x{actual_color_size[1]}"
    )
    print(
        "Right click points; right click the first point to close a region; "
        "left click undoes the latest pending point. Click a completed region "
        "or sidebar row to select/edit. Confirm Average computes the current set."
    )
    previous_time = cv2.getTickCount()
    fps_average = float(args.fps)
    output_path = args.output.expanduser().resolve() if args.output else default_output_path()

    def autosave() -> None:
        save_report(
            output_path,
            recorder.records,
            camera_info,
            args.fps,
            (recorder.width, recorder.height),
            recorder.confirmed_average,
            recorder.pending_points,
        )

    recorder.on_change = autosave
    should_save = False
    frame_index = 0
    yolo_memory: list[dict[str, Any]] = []
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            latest_depth = np.asanyarray(depth_frame.get_data())
            latest_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            latest_depth_scale = depth_scale
            recorder.set_latest_depth(latest_depth, latest_depth_scale, latest_intrinsics)

            ticks = cv2.getTickCount()
            elapsed = (ticks - previous_time) / cv2.getTickFrequency()
            previous_time = ticks
            if elapsed > 0:
                fps_average = 0.85 * fps_average + 0.15 / elapsed
            if yolo_model is not None:
                display_color, yolo_detections = draw_yolo_overlay(
                    color, yolo_model, args, yolo_memory
                )
                yolo_memory = yolo_detections
                recorder.set_yolo_detections(yolo_detections)
            else:
                display_color = color
                yolo_memory = []
                recorder.set_yolo_detections([])
            preview = recorder.draw(display_color, fps_average)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if recorder.handle_edit_key(key):
                frame_index += 1
                continue
            if key in (ord("q"), 27):
                should_save = True
                break
            if key in (ord("x"), 8, 127):
                recorder.delete_selected()
            elif key == ord("u"):
                recorder.undo()
            elif key == ord("c"):
                recorder.clear()
            elif key == ord("a"):
                print_summary(recorder.records)
            elif key in (13, 10):
                recorder.confirm_average()
            elif key == ord("s"):
                save_report(output_path, recorder.records, camera_info, args.fps,
                            (recorder.width, recorder.height),
                            recorder.confirmed_average, recorder.pending_points)
            elif key == ord("h"):
                print(
                    "Controls: right click add/close polygon; left click undo point "
                    "or select; x/Delete remove; u undo; c clear; a average; "
                    "s save; q quit"
                )
            frame_index += 1
    except KeyboardInterrupt:
        should_save = True
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    if should_save:
        print_summary(recorder.records)
        save_report(output_path, recorder.records, camera_info, args.fps,
                    (recorder.width, recorder.height), recorder.confirmed_average,
                    recorder.pending_points)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"distance: {error}", file=sys.stderr)
        raise SystemExit(1)
