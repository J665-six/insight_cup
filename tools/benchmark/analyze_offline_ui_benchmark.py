#!/usr/bin/env python3
"""Analyze offline UI/no-UI benchmark logs and render resource charts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


FRAME_PATTERN = re.compile(
    r"FRAME frame=(?P<frame>\d+) processed=(?P<processed>\d+) "
    r"detections=(?P<detections>\d+) yolo_ms=(?P<yolo>[0-9.]+) "
    r"stage2_ms=(?P<stage2>[0-9.]+) total_ms=(?P<total>[0-9.]+) "
    r"fps=(?P<fps>[0-9.]+)"
)
FILENAME_TOKEN_CLASSES = {
    "f1_f4": {"f1", "f4"},
    "k4_k5_k10": {"k4", "k5", "k10"},
}
EXPECTED_MAJOR = {"f1_f4": "f", "k4_k5_k10": "k"}
RECORDING_IDS = {
    "all": "20260917_230330_750162",
    "f1_f4": "20260918_093215_640581",
    "k4_k5_k10": "20260918_081653_413685",
}
EXPECTED_RUN_IDS = {
    f"{video_key}-{mode}"
    for video_key in RECORDING_IDS
    for mode in ("no-ui", "ui")
}
SEMANTIC_REJECTIONS = {"unknown", "ambiguous", "no_face"}
STAT_KEYS = ("mean", "p50", "p95", "p99", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an offline UI/no-UI benchmark directory"
    )
    parser.add_argument("benchmark_root", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_resources(path: Path) -> list[dict[str, Any]]:
    string_fields = {"phase", "npu_runtime_status"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    key: value if key in string_fields else optional_float(value)
                    for key, value in raw.items()
                }
            )
    return rows


def numeric(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, **{key: None for key in STAT_KEYS}}
    data = np.asarray(values, dtype=float)
    return {
        "samples": int(data.size),
        "mean": round(float(np.mean(data)), 6),
        "p50": round(float(np.percentile(data, 50)), 6),
        "p95": round(float(np.percentile(data, 95)), 6),
        "p99": round(float(np.percentile(data, 99)), 6),
        "max": round(float(np.max(data)), 6),
    }


def integrated_process_cost(
    rows: Sequence[Mapping[str, Any]],
    process: str,
    start_t_s: float,
    end_t_s: float,
    frames: int,
) -> dict[str, Any]:
    """Integrate sampled /proc CPU ticks over the media-processing window."""
    core_seconds = 0.0
    covered_seconds = 0.0
    delta_key = f"{process}_cpu_core_seconds_delta"
    window_start_key = f"{process}_cpu_window_start_s"
    window_end_key = f"{process}_cpu_window_end_s"
    exact_windows = [
        row
        for row in rows
        if row.get(delta_key) is not None
        and row.get(window_start_key) is not None
        and row.get(window_end_key) is not None
    ]
    for row in exact_windows:
        window_start = float(row[window_start_key])
        window_end = float(row[window_end_key])
        window_duration = window_end - window_start
        overlap_start = max(window_start, start_t_s)
        overlap_end = min(window_end, end_t_s)
        if window_duration <= 0 or overlap_end <= overlap_start:
            continue
        overlap = overlap_end - overlap_start
        core_seconds += float(row[delta_key]) * overlap / window_duration
        covered_seconds += overlap

    # Backward compatibility for benchmark captures made before exact tick deltas
    # were added. Formal runs use the exact-window branch above.
    previous_t: float | None = None
    if not exact_windows:
        key = f"{process}_cpu_core_pct"
        for row in rows:
            current_t_value = row.get("t_s")
            if current_t_value is None:
                continue
            current_t = float(current_t_value)
            if previous_t is not None:
                overlap_start = max(previous_t, start_t_s)
                overlap_end = min(current_t, end_t_s)
                value = row.get(key)
                if (
                    overlap_end > overlap_start
                    and value is not None
                    and math.isfinite(float(value))
                ):
                    duration = overlap_end - overlap_start
                    core_seconds += float(value) / 100.0 * duration
                    covered_seconds += duration
            previous_t = current_t
    if covered_seconds <= 0:
        return {
            "core_seconds": None,
            "core_ms_per_frame": None,
            "covered_seconds": 0.0,
            "media_window_coverage": 0.0,
            "source": "proc_tick_delta" if exact_windows else "unavailable",
        }
    media_duration = max(0.0, end_t_s - start_t_s)
    return {
        "core_seconds": round(core_seconds, 6),
        "core_ms_per_frame": (
            round(core_seconds * 1000.0 / frames, 6) if frames else None
        ),
        "covered_seconds": round(covered_seconds, 6),
        "media_window_coverage": (
            round(min(covered_seconds / media_duration, 1.0), 6)
            if media_duration > 0
            else None
        ),
        "source": "proc_tick_delta" if exact_windows else "legacy_percent_window",
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def parse_frame_log(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = FRAME_PATTERN.search(line)
        if not match:
            continue
        values = match.groupdict()
        frames.append(
            {
                "frame": int(values["frame"]),
                "processed": int(values["processed"]),
                "detections": int(values["detections"]),
                "yolo_ms": float(values["yolo"]),
                "stage2_ms": float(values["stage2"]),
                "total_ms": float(values["total"]),
                "fps": float(values["fps"]),
            }
        )
    return frames


def decision_metrics(video_key: str, shadow: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    filename_tokens = FILENAME_TOKEN_CLASSES.get(video_key)
    major = EXPECTED_MAJOR.get(video_key)
    related = [record for record in shadow if major is None or record["source"]["major_class"] == major]
    final_status = Counter(record["decision"]["status"] for record in related)
    raw_status = Counter(record["stage2_raw"]["status"] for record in related)
    final_accepted = [
        record
        for record in related
        if record["decision"]["status"] == "matched"
        and record["decision"].get("predicted_class") is not None
    ]
    raw_accepted = [
        record
        for record in related
        if record["stage2_raw"]["status"] == "matched"
        and record["stage2_raw"].get("predicted_class") is not None
    ]
    final_predictions = Counter(
        record["decision"].get("predicted_class") or record["decision"]["status"]
        for record in related
    )
    raw_predictions = Counter(
        record["stage2_raw"].get("predicted_class")
        or record["stage2_raw"]["status"]
        for record in related
    )
    yolo_predictions = Counter(record["source"]["raw_yolo_class"] for record in related)
    final_token_matches = (
        sum(
            record["decision"]["predicted_class"] in filename_tokens
            for record in final_accepted
        )
        if filename_tokens is not None
        else None
    )
    raw_token_matches = (
        sum(
            record["stage2_raw"]["predicted_class"] in filename_tokens
            for record in raw_accepted
        )
        if filename_tokens is not None
        else None
    )
    yolo_token_matches = (
        sum(
            record["source"]["raw_yolo_class"] in filename_tokens
            for record in related
        )
        if filename_tokens is not None
        else None
    )
    semantic_rejections = sum(final_status[item] for item in SEMANTIC_REJECTIONS)
    return {
        "scope_major_class": major,
        "records": len(related),
        "filename_token_classes": (
            sorted(filename_tokens) if filename_tokens is not None else None
        ),
        "final_status_counts": dict(sorted(final_status.items())),
        "raw_status_counts": dict(sorted(raw_status.items())),
        "final_prediction_counts": dict(sorted(final_predictions.items())),
        "raw_prediction_counts": dict(sorted(raw_predictions.items())),
        "yolo_raw_class_counts": dict(sorted(yolo_predictions.items())),
        "accepted": len(final_accepted),
        "semantic_rejections": semantic_rejections,
        "confirming": final_status["confirming"],
        "errors": final_status["error"],
        "accepted_rate": rate(len(final_accepted), len(related)),
        "semantic_rejection_rate": rate(semantic_rejections, len(related)),
        "confirming_rate": rate(final_status["confirming"], len(related)),
        "filename_token_accepted": final_token_matches,
        "filename_token_consistency": (
            rate(final_token_matches, len(final_accepted))
            if final_token_matches is not None
            else None
        ),
        "target_record_token_yield": (
            rate(final_token_matches, len(related))
            if final_token_matches is not None
            else None
        ),
        "raw_accepted": len(raw_accepted),
        "raw_filename_token_accepted": raw_token_matches,
        "raw_filename_token_consistency": (
            rate(raw_token_matches, len(raw_accepted))
            if raw_token_matches is not None
            else None
        ),
        "yolo_filename_token_records": yolo_token_matches,
        "yolo_filename_token_consistency": (
            rate(yolo_token_matches, len(related))
            if yolo_token_matches is not None
            else None
        ),
    }


def source_frame_index(path: str) -> int:
    match = re.search(r"_src_(\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse raw source frame from {path}")
    return int(match.group(1))


def load_sparse_ground_truth(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    ground_truth: dict[str, list[dict[str, Any]]] = {
        key: [] for key in RECORDING_IDS
    }
    knife_manifest = read_json(
        project_root / "data/knife_dataset/trainv5_grouped/manifest.json"
    )
    for record in knife_manifest["records"]:
        for video_key, recording_id in RECORDING_IDS.items():
            if record["recording_id"] != recording_id:
                continue
            ground_truth[video_key].append(
                {
                    "key": (
                        f"knife:{record['source_image']}:"
                        f"{record['source_label_line']}"
                    ),
                    "major_class": "k",
                    "expected_class": record["class_name"],
                    "frame_index": source_frame_index(record["source_image"]),
                    "bbox_xyxy": record["pixel_xyxy"],
                    "derived_split": record["split"],
                    "source_split": record["source_split"],
                    "provenance": "knife_dataset_manifest",
                }
            )
    face_manifest = read_json(
        project_root / "data/face_gallery/references_trainv5/manifest.json"
    )
    for record in face_manifest["references"]:
        for video_key, recording_id in RECORDING_IDS.items():
            if recording_id not in Path(record["source_image"]).name:
                continue
            ground_truth[video_key].append(
                {
                    "key": f"face:{record['reference']}",
                    "major_class": "f",
                    "expected_class": record["identity"],
                    "frame_index": source_frame_index(record["source_image"]),
                    "bbox_xyxy": record["bbox_xyxy"],
                    "derived_split": "gallery_reference",
                    "source_split": "train",
                    "provenance": "face_gallery_reference_manifest",
                }
            )
    return ground_truth


def sparse_reference_metrics(
    items: Sequence[Mapping[str, Any]], shadow: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for major in ("f", "k"):
        expected = [item for item in items if item["major_class"] == major]
        if not expected:
            continue
        predictions = [
            record for record in shadow if record["source"]["major_class"] == major
        ]
        expected_groups: dict[int, list[Mapping[str, Any]]] = {}
        prediction_groups: dict[int, list[Mapping[str, Any]]] = {}
        for item in expected:
            expected_groups.setdefault(int(item["frame_index"]), []).append(item)
        for record in predictions:
            prediction_groups.setdefault(
                int(record["source"]["frame_index"]), []
            ).append(record)
        observed_frames = sum(frame in prediction_groups for frame in expected_groups)
        matches: list[dict[str, Any]] = []
        for frame, frame_expected in expected_groups.items():
            frame_predictions = prediction_groups.get(frame, [])
            if not frame_predictions:
                continue
            overlaps = np.asarray(
                [
                    [
                        iou(item["bbox_xyxy"], record["source"]["bbox_xyxy"])
                        for record in frame_predictions
                    ]
                    for item in frame_expected
                ],
                dtype=float,
            )
            expected_indexes, prediction_indexes = linear_sum_assignment(
                1.0 - overlaps
            )
            for expected_index, prediction_index in zip(
                expected_indexes, prediction_indexes
            ):
                overlap = float(overlaps[expected_index, prediction_index])
                if overlap < 0.5:
                    continue
                item = frame_expected[int(expected_index)]
                record = frame_predictions[int(prediction_index)]
                raw = record["stage2_raw"]
                final = record["decision"]
                raw_correct = (
                    raw["status"] == "matched"
                    and raw.get("predicted_class") == item["expected_class"]
                )
                final_correct = (
                    final["status"] == "matched"
                    and final.get("predicted_class") == item["expected_class"]
                )
                matches.append(
                    {
                        "ground_truth_key": item["key"],
                        "source_id": record["source"]["id"],
                        "frame_index": frame,
                        "expected_class": item["expected_class"],
                        "iou": round(overlap, 6),
                        "raw_status": raw["status"],
                        "raw_predicted_class": raw.get("predicted_class"),
                        "raw_correct": raw_correct,
                        "final_status": final["status"],
                        "final_predicted_class": final.get("predicted_class"),
                        "final_correct": final_correct,
                    }
                )
        raw_correct_count = sum(item["raw_correct"] for item in matches)
        final_correct_count = sum(item["final_correct"] for item in matches)
        final_accepted_count = sum(
            item["final_status"] == "matched"
            and item["final_predicted_class"] is not None
            for item in matches
        )
        provenance = Counter(item["provenance"] for item in expected)
        derived_splits = Counter(item["derived_split"] for item in expected)
        source_splits = Counter(item["source_split"] for item in expected)
        results[major] = {
            "ground_truth_frames": len(expected_groups),
            "ground_truth_boxes": len(expected),
            "observable_frames": observed_frames,
            "frame_observability": rate(observed_frames, len(expected_groups)),
            "iou_aligned_boxes": len(matches),
            "iou_alignment_coverage": rate(len(matches), len(expected)),
            "raw_correct": raw_correct_count,
            "raw_conditional_accuracy": rate(raw_correct_count, len(matches)),
            "final_correct": final_correct_count,
            "final_conditional_accuracy": rate(final_correct_count, len(matches)),
            "final_accepted": final_accepted_count,
            "final_accepted_coverage": rate(final_accepted_count, len(matches)),
            "accepted_precision": rate(final_correct_count, final_accepted_count),
            "provenance_counts": dict(sorted(provenance.items())),
            "derived_split_counts": dict(sorted(derived_splits.items())),
            "source_split_counts": dict(sorted(source_splits.items())),
            "matches": matches,
            "warning": (
                "Conditional on fresh-shadow observability and IoU alignment; "
                "unaligned GT is not a detection miss."
            ),
        }
    return results


def rknn_performance_metrics(
    records: Sequence[Mapping[str, Any]], inference_wall_s: float
) -> dict[str, Any]:
    core_duration_us: Counter[int] = Counter()
    core_runs: Counter[int] = Counter()
    models: list[dict[str, Any]] = []
    for record in records:
        core = int(record["core"])
        duration = int(record["total_run_duration_us"])
        runs = int(record["runs"])
        core_duration_us[core] += duration
        core_runs[core] += runs
        models.append(
            {
                "model": Path(record["model"]).name,
                "core": core,
                "runs": runs,
                "total_run_duration_us": duration,
                "mean_run_duration_ms": (
                    round(duration / runs / 1000.0, 6) if runs else None
                ),
                "min_run_duration_ms": round(
                    int(record["min_run_duration_us"]) / 1000.0, 6
                ),
                "max_run_duration_ms": round(
                    int(record["max_run_duration_us"]) / 1000.0, 6
                ),
                "query_errors": int(record.get("query_errors", 0)),
            }
        )
    denominator_us = inference_wall_s * 1_000_000.0
    core_proxy = {
        str(core): (
            round(core_duration_us[core] / denominator_us * 100.0, 6)
            if denominator_us > 0
            else None
        )
        for core in range(3)
    }
    total_duration = sum(core_duration_us.values())
    return {
        "available": bool(records),
        "definition": (
            "sum(RKNN_QUERY_PERF_RUN duration) / inference wall time; app-only "
            "compute-time proxy, not hardware utilization"
        ),
        "core_compute_time_proxy_pct": core_proxy,
        "aggregate_capacity_proxy_pct": (
            round(total_duration / (denominator_us * 3.0) * 100.0, 6)
            if denominator_us > 0 and records
            else None
        ),
        "core_total_run_duration_us": {
            str(core): core_duration_us[core] for core in range(3)
        },
        "core_runs": {str(core): core_runs[core] for core in range(3)},
        "models": sorted(models, key=lambda item: (item["core"], item["model"])),
    }


def analyze_run(
    meta: Mapping[str, Any], sparse_items: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = str(meta["run_id"])
    run_root = Path(meta["resources_csv"]).parent
    session_root = Path(meta["session_root"])
    resources = read_resources(Path(meta["resources_csv"]))
    first_t = float(meta["first_frame_t_s"])
    end_t = float(meta["runtime_end_t_s"])
    inference = [
        row
        for row in resources
        if row["phase"] == "inference"
        and first_t <= float(row["t_s"]) <= end_t
        and (row.get("app_threads") or 0) > 0
    ]
    baseline = [row for row in resources if row["phase"] == "baseline"]
    frames = parse_frame_log(Path(meta["application_log"]))
    shadow = read_jsonl(session_root / "shadow.jsonl")
    events = read_jsonl(session_root / "events.jsonl")
    manifest = read_json(session_root / "manifest.json")
    summary = read_json(session_root / "summary.json")
    rknn_performance = read_jsonl(Path(meta["rknn_performance_jsonl"]))
    first_frame_duration = frames[0]["total_ms"] / 1000.0 if frames else 0.0
    media_start_t = max(0.0, first_t - first_frame_duration)
    inference_wall = max(0.0, end_t - first_t + first_frame_duration)
    face_stage2 = [
        float(record["timing_ms"]["stage2"])
        for record in shadow
        if record["source"]["major_class"] == "f"
    ]
    knife_stage2 = [
        float(record["timing_ms"]["stage2"])
        for record in shadow
        if record["source"]["major_class"] == "k"
    ]
    app_cpu_cost = integrated_process_cost(
        resources,
        "app",
        media_start_t,
        end_t,
        len(frames),
    )
    browser_cpu_cost = integrated_process_cost(
        resources,
        "browser",
        media_start_t,
        end_t,
        len(frames),
    )
    chromium_log = run_root / "chromium.log"
    graphics_warning_lines = (
        [
            line
            for line in chromium_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if "libEGL warning" in line or "gpu_init.cc" in line
        ]
        if chromium_log.exists()
        else []
    )
    metrics = {
        "run_id": run_id,
        "video_key": meta["video_key"],
        "ui": bool(meta["ui"]),
        "visible_browser": bool(meta["visible_browser"]),
        "processed_frames": summary["processed_frames"],
        "expected_frames": meta["expected_frames"],
        "complete_frame_log": len(frames) == int(meta["expected_frames"]),
        "frame_log_records": len(frames),
        "inference_wall_s": round(inference_wall, 6),
        "effective_fps": (
            round(len(frames) / inference_wall, 6) if inference_wall > 0 else None
        ),
        "latency_ms": {
            "frame_total": distribution([record["total_ms"] for record in frames]),
            "yolo": distribution([record["yolo_ms"] for record in frames]),
            "face_fresh_stage2": distribution(face_stage2),
            "knife_fresh_stage2": distribution(knife_stage2),
        },
        "cpu_pct": {
            "app_core_equivalent": distribution(numeric(inference, "app_cpu_core_pct")),
            "app_machine_share": distribution(numeric(inference, "app_cpu_machine_pct")),
            "browser_core_equivalent": distribution(
                numeric(inference, "browser_cpu_core_pct")
            ),
            "browser_machine_share": distribution(
                numeric(inference, "browser_cpu_machine_pct")
            ),
            "system_total": distribution(numeric(inference, "system_cpu_pct")),
            "baseline_system_total": distribution(numeric(baseline, "system_cpu_pct")),
        },
        "cpu_cost": {
            "app": app_cpu_cost,
            "browser": browser_cpu_cost,
        },
        "browser_graphics": {
            "acceleration_warning": bool(graphics_warning_lines),
            "warning_lines": graphics_warning_lines,
        },
        "memory_kib": {
            "app_rss": distribution(numeric(inference, "app_rss_kib")),
            "browser_rss": distribution(numeric(inference, "browser_rss_kib")),
        },
        "npu": {
            "debugfs_available": any(
                value > 0 for value in numeric(resources, "npu_debug_available")
            ),
            "debug_core0_pct": distribution(numeric(inference, "npu_core0_pct")),
            "debug_core1_pct": distribution(numeric(inference, "npu_core1_pct")),
            "debug_core2_pct": distribution(numeric(inference, "npu_core2_pct")),
            "debug_capacity_mean_pct": distribution(
                numeric(inference, "npu_debug_mean_pct")
            ),
            "power_active_residency_pct": distribution(
                numeric(inference, "npu_power_active_pct")
            ),
            "frequency_hz": distribution(numeric(inference, "npu_freq_hz")),
            "devfreq_load_excluded": True,
            "app_compute_time_proxy": rknn_performance_metrics(
                rknn_performance, inference_wall
            ),
        },
        "temperature_c": {
            "soc": distribution(numeric(inference, "soc_temp_c")),
            "npu": distribution(numeric(inference, "npu_temp_c")),
        },
        "recognition_proxy": decision_metrics(str(meta["video_key"]), shadow),
        "sparse_reference": sparse_reference_metrics(sparse_items, shadow),
        "shadow_records": len(shadow),
        "events": len(events),
        "input_provenance": shadow[0]["input"] if shadow else None,
        "manifest_config": manifest["config"],
        "model_metadata": manifest["models"],
        "strict_accuracy": {
            "available": False,
            "reason": "No exhaustive per-frame human ground truth exists for this video.",
        },
        "run_root": str(run_root),
        "session_root": str(session_root),
    }
    return metrics, resources


def equivalent(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            equivalent(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(
            equivalent(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def subset(record: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: record.get(field) for field in fields}


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_fallback_pairs(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    left_ids: set[str],
    right_ids: set[str],
) -> list[tuple[str, str]]:
    candidates: list[tuple[float, str, str]] = []
    for left_id in left_ids:
        left_source = left[left_id]["source"]
        for right_id in right_ids:
            right_source = right[right_id]["source"]
            if (
                left_source["frame_index"] != right_source["frame_index"]
                or left_source["major_class"] != right_source["major_class"]
            ):
                continue
            overlap = iou(left_source["bbox_xyxy"], right_source["bbox_xyxy"])
            if overlap >= 0.9:
                candidates.append((overlap, left_id, right_id))
    pairs: list[tuple[str, str]] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    for _, left_id, right_id in sorted(candidates, reverse=True):
        if left_id not in used_left and right_id not in used_right:
            pairs.append((left_id, right_id))
            used_left.add(left_id)
            used_right.add(right_id)
    return pairs


def compare_pair(no_ui: Mapping[str, Any], ui: Mapping[str, Any]) -> dict[str, Any]:
    no_records = read_jsonl(Path(no_ui["session_root"]) / "shadow.jsonl")
    ui_records = read_jsonl(Path(ui["session_root"]) / "shadow.jsonl")
    no_map = {record["source"]["id"]: record for record in no_records}
    ui_map = {record["source"]["id"]: record for record in ui_records}
    common = set(no_map) & set(ui_map)
    only_no = set(no_map) - set(ui_map)
    only_ui = set(ui_map) - set(no_map)
    fallback = bbox_fallback_pairs(no_map, ui_map, only_no, only_ui)
    pairs = [(source_id, source_id) for source_id in sorted(common)] + fallback
    source_fields = (
        "frame_index",
        "major_class",
        "confidence",
        "bbox_xyxy",
        "raw_yolo_class_id",
        "raw_yolo_class",
        "class_scores",
    )
    stage2_fields = (
        "status",
        "candidate_class",
        "predicted_class",
        "score",
        "score_kind",
        "second_best_class",
        "second_best_score",
        "margin",
        "class_scores",
    )
    decision_fields = stage2_fields + ("temporal",)
    source_equal = sum(
        equivalent(
            subset(no_map[left]["source"], source_fields),
            subset(ui_map[right]["source"], source_fields),
        )
        for left, right in pairs
    )
    raw_equal = sum(
        equivalent(
            subset(no_map[left]["stage2_raw"], stage2_fields),
            subset(ui_map[right]["stage2_raw"], stage2_fields),
        )
        for left, right in pairs
    )
    decision_equal = sum(
        equivalent(
            subset(no_map[left]["decision"], decision_fields),
            subset(ui_map[right]["decision"], decision_fields),
        )
        for left, right in pairs
    )
    fallback_left = {left for left, _ in fallback}
    fallback_right = {right for _, right in fallback}
    union = set(no_map) | set(ui_map)
    sparse_paired: dict[str, Any] = {}
    for major in ("f", "k"):
        no_sparse = no_ui.get("sparse_reference", {}).get(major)
        ui_sparse = ui.get("sparse_reference", {}).get(major)
        if no_sparse is None or ui_sparse is None:
            continue
        no_matches = {
            item["ground_truth_key"]: item for item in no_sparse["matches"]
        }
        ui_matches = {
            item["ground_truth_key"]: item for item in ui_sparse["matches"]
        }
        common_ground_truth = set(no_matches) & set(ui_matches)
        sparse_paired[major] = {
            "common_ground_truth_boxes": len(common_ground_truth),
            "no_ui_final_conditional_accuracy": rate(
                sum(no_matches[key]["final_correct"] for key in common_ground_truth),
                len(common_ground_truth),
            ),
            "ui_final_conditional_accuracy": rate(
                sum(ui_matches[key]["final_correct"] for key in common_ground_truth),
                len(common_ground_truth),
            ),
            "raw_output_equality_rate": rate(
                sum(
                    (
                        no_matches[key]["raw_status"],
                        no_matches[key]["raw_predicted_class"],
                    )
                    == (
                        ui_matches[key]["raw_status"],
                        ui_matches[key]["raw_predicted_class"],
                    )
                    for key in common_ground_truth
                ),
                len(common_ground_truth),
            ),
            "final_output_equality_rate": rate(
                sum(
                    (
                        no_matches[key]["final_status"],
                        no_matches[key]["final_predicted_class"],
                    )
                    == (
                        ui_matches[key]["final_status"],
                        ui_matches[key]["final_predicted_class"],
                    )
                    for key in common_ground_truth
                ),
                len(common_ground_truth),
            ),
        }
    return {
        "video_key": no_ui["video_key"],
        "same_input_provenance": no_ui["input_provenance"] == ui["input_provenance"],
        "no_ui_records": len(no_records),
        "ui_records": len(ui_records),
        "exact_id_common": len(common),
        "bbox_fallback_pairs": len(fallback),
        "paired_records": len(pairs),
        "id_jaccard": rate(len(common), len(union)),
        "unpaired_no_ui": len(only_no - fallback_left),
        "unpaired_ui": len(only_ui - fallback_right),
        "source_equality_rate": rate(source_equal, len(pairs)),
        "raw_stage2_equality_rate": rate(raw_equal, len(pairs)),
        "final_decision_equality_rate": rate(decision_equal, len(pairs)),
        "paired_sparse_reference": sparse_paired,
    }


def rolling(values: Sequence[float], window: int = 5) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(data, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def series(
    resources: Sequence[Mapping[str, Any]], first_frame_t_s: float, key: str
) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["t_s"]) - first_frame_t_s, float(row[key]))
        for row in resources
        if row.get("phase") == "inference" and row.get(key) is not None
    ]
    if not points:
        return np.asarray([]), np.asarray([])
    x, y = zip(*points)
    return np.asarray(x), np.asarray(y)


def plot_cpu_pair(
    output: Path,
    video_key: str,
    no_ui: Mapping[str, Any],
    ui: Mapping[str, Any],
    resources: Mapping[str, Sequence[Mapping[str, Any]]],
    metas: Mapping[str, Mapping[str, Any]],
) -> None:
    colors = {"no-ui": "#16835d", "ui": "#c64636", "browser": "#d18b16"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"CPU usage: {video_key} (8-core machine share)", fontsize=14)
    for mode, run in (("no-ui", no_ui), ("ui", ui)):
        run_id = str(run["run_id"])
        x, y = series(
            resources[run_id], float(metas[run_id]["first_frame_t_s"]), "app_cpu_machine_pct"
        )
        if len(x):
            axes[0].plot(x, y, color=colors[mode], alpha=0.18, linewidth=0.8)
            axes[0].plot(x, rolling(y), color=colors[mode], linewidth=1.8, label=f"App {mode}")
    ui_id = str(ui["run_id"])
    x, y = series(
        resources[ui_id],
        float(metas[ui_id]["first_frame_t_s"]),
        "browser_cpu_machine_pct",
    )
    if len(x):
        axes[0].plot(x, y, color=colors["browser"], alpha=0.18, linewidth=0.8)
        axes[0].plot(x, rolling(y), color=colors["browser"], linewidth=1.8, label="UI Chromium")
    axes[0].set_ylabel("Process CPU (% machine)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    for mode, run in (("no-ui", no_ui), ("ui", ui)):
        run_id = str(run["run_id"])
        x, y = series(
            resources[run_id], float(metas[run_id]["first_frame_t_s"]), "system_cpu_pct"
        )
        if len(x):
            axes[1].plot(x, y, color=colors[mode], alpha=0.18, linewidth=0.8)
            axes[1].plot(x, rolling(y), color=colors[mode], linewidth=1.8, label=f"System {mode}")
    axes[1].set_xlabel("Seconds since first processed frame")
    axes[1].set_ylabel("Whole-system CPU (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_summary(output: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(run["run_id"]) for run in runs]
    app = [run["cpu_pct"]["app_machine_share"]["mean"] or 0.0 for run in runs]
    browser = [run["cpu_pct"]["browser_machine_share"]["mean"] or 0.0 for run in runs]
    system = [run["cpu_pct"]["system_total"]["mean"] or 0.0 for run in runs]
    positions = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(positions - width, app, width, label="App", color="#16835d")
    axis.bar(positions, browser, width, label="UI Chromium", color="#d18b16")
    axis.bar(positions + width, system, width, label="Whole system", color="#53606d")
    axis.set_ylabel("Mean CPU (% of 8-core machine)")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def number(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def relative_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current / baseline - 1.0) * 100.0


def difference(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def ui_impact(by_video: Mapping[str, Mapping[bool, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for video_key, modes in sorted(by_video.items()):
        if False not in modes or True not in modes:
            continue
        no_ui = modes[False]
        ui = modes[True]
        no_ui_models = {
            item["model"]: item
            for item in no_ui["npu"]["app_compute_time_proxy"]["models"]
        }
        ui_models = {
            item["model"]: item
            for item in ui["npu"]["app_compute_time_proxy"]["models"]
        }
        no_ui_yolo = no_ui_models.get("yolo_best.rknn", {})
        ui_yolo = ui_models.get("yolo_best.rknn", {})
        no_ui_system_above_baseline = difference(
            no_ui["cpu_pct"]["system_total"]["mean"],
            no_ui["cpu_pct"]["baseline_system_total"]["mean"],
        )
        ui_system_above_baseline = difference(
            ui["cpu_pct"]["system_total"]["mean"],
            ui["cpu_pct"]["baseline_system_total"]["mean"],
        )
        output.append(
            {
                "video_key": video_key,
                "frame_total_mean_change_pct": relative_change(
                    ui["latency_ms"]["frame_total"]["mean"],
                    no_ui["latency_ms"]["frame_total"]["mean"],
                ),
                "frame_total_p95_change_pct": relative_change(
                    ui["latency_ms"]["frame_total"]["p95"],
                    no_ui["latency_ms"]["frame_total"]["p95"],
                ),
                "effective_fps_change_pct": relative_change(
                    ui["effective_fps"], no_ui["effective_fps"]
                ),
                "app_cpu_machine_delta_points": difference(
                    ui["cpu_pct"]["app_machine_share"]["mean"],
                    no_ui["cpu_pct"]["app_machine_share"]["mean"],
                ),
                "browser_cpu_machine_pct": ui["cpu_pct"][
                    "browser_machine_share"
                ]["mean"],
                "system_cpu_raw_delta_points": difference(
                    ui["cpu_pct"]["system_total"]["mean"],
                    no_ui["cpu_pct"]["system_total"]["mean"],
                ),
                "system_cpu_baseline_adjusted_delta_points": difference(
                    ui_system_above_baseline,
                    no_ui_system_above_baseline,
                ),
                "app_cpu_core_ms_per_frame_no_ui": no_ui["cpu_cost"]["app"][
                    "core_ms_per_frame"
                ],
                "app_cpu_core_ms_per_frame_ui": ui["cpu_cost"]["app"][
                    "core_ms_per_frame"
                ],
                "app_cpu_core_ms_per_frame_change_pct": relative_change(
                    ui["cpu_cost"]["app"]["core_ms_per_frame"],
                    no_ui["cpu_cost"]["app"]["core_ms_per_frame"],
                ),
                "browser_cpu_core_ms_per_frame_ui": ui["cpu_cost"]["browser"][
                    "core_ms_per_frame"
                ],
                "yolo_mean_change_pct": relative_change(
                    ui["latency_ms"]["yolo"]["mean"],
                    no_ui["latency_ms"]["yolo"]["mean"],
                ),
                "npu_yolo_runs_no_ui": no_ui_yolo.get("runs"),
                "npu_yolo_runs_ui": ui_yolo.get("runs"),
                "npu_yolo_mean_ms_no_ui": no_ui_yolo.get(
                    "mean_run_duration_ms"
                ),
                "npu_yolo_mean_ms_ui": ui_yolo.get("mean_run_duration_ms"),
            }
        )
    return output


def markdown_report(
    benchmark: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    chart_names: Mapping[str, str],
    impacts: Sequence[Mapping[str, Any]],
) -> str:
    baseline_means = [
        float(run["cpu_pct"]["baseline_system_total"]["mean"])
        for run in runs
        if run["cpu_pct"]["baseline_system_total"]["mean"] is not None
    ]
    baseline_range = (
        f"{min(baseline_means):.1f}% 到 {max(baseline_means):.1f}%"
        if baseline_means
        else "N/A"
    )
    browser_graphics_warning = any(
        run["browser_graphics"]["acceleration_warning"] for run in runs
    )
    cpu_coverages = [
        float(run["cpu_cost"][process]["media_window_coverage"])
        for run in runs
        for process in ("app", "browser")
        if process == "app" or bool(run["ui"])
    ]
    minimum_cpu_coverage = min(cpu_coverages) if cpu_coverages else None
    logical_cpus = int(benchmark.get("host", {}).get("logical_cpus", 1))
    lines = [
        "# 三段离线视频 UI / 无 UI 基准报告",
        "",
        f"> 生成时间：{benchmark.get('completed_at', benchmark['created_at'])}",
        "> 口径：RKNN、vid-stride=1、源视频节流、正常调度优先级、shadow 开启、录像/crop 关闭；每种场景单次观测。",
        "",
        "## 结论边界",
        "",
        "三段视频没有完整逐帧人工真值，所以本报告不能给出严格端到端 accuracy。下文的“文件名 token 一致率”只检查最终接受预测是否落在文件名列出的 token 集合中；它不是正确率，无法区分 f1 与 f4 互相误判，也无法统计 YOLO 漏框。`k4_k5_k10` 的现有稀疏标注中还包含一个真实 k2，因此该指标尤其不能作为 ground truth。`all` 不计算此项。",
        "",
        "这些视频还与训练集/人脸图库有来源重叠，指标只能描述本机离线回放和样本内表现，不能代表泛化精度。",
        "",
        "## UI 开销摘要",
        "",
        "| 视频 | 平均总延迟变化 | P95 总延迟变化 | 有效 FPS 变化 | 应用 CPU core-ms/帧 no-UI / UI（变化） | Chromium core-ms/帧 | 系统 CPU 基线校正差（百分点） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for impact in impacts:
        lines.append(
            f"| {impact['video_key']} | {number(impact['frame_total_mean_change_pct'])}% | "
            f"{number(impact['frame_total_p95_change_pct'])}% | "
            f"{number(impact['effective_fps_change_pct'])}% | "
            f"{number(impact['app_cpu_core_ms_per_frame_no_ui'])} / "
            f"{number(impact['app_cpu_core_ms_per_frame_ui'])} "
            f"({number(impact['app_cpu_core_ms_per_frame_change_pct'])}%) | "
            f"{number(impact['browser_cpu_core_ms_per_frame_ui'])} | "
            f"{number(impact['system_cpu_baseline_adjusted_delta_points'])} |"
        )
    lines += [
        "",
        f"这里的 UI 包含应用端逐帧 resize/JPEG、MJPEG/JSON 服务，以及一个实际可见的 Chromium 窗口。系统 CPU 运行前基线为 {baseline_range}；表中系统差值先分别减去各轮基线，但进程 CPU 和每帧 CPU 成本仍更适合归因。",
        "",
        "## 运行结果",
        "",
        "| 视频 / 模式 | 帧数 | 有效 FPS | 总延迟 mean / P95 / P99 (ms) | YOLO mean / P95 (ms) | Face fresh P95 (ms) | Knife fresh P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        latency = run["latency_ms"]
        lines.append(
            "| {run_id} | {frames} | {fps} | {total_mean} / {total_p95} / {total_p99} | "
            "{yolo_mean} / {yolo_p95} | {face_p95} | {knife_p95} |".format(
                run_id=run["run_id"],
                frames=run["processed_frames"],
                fps=number(run["effective_fps"]),
                total_mean=number(latency["frame_total"]["mean"]),
                total_p95=number(latency["frame_total"]["p95"]),
                total_p99=number(latency["frame_total"]["p99"]),
                yolo_mean=number(latency["yolo"]["mean"]),
                yolo_p95=number(latency["yolo"]["p95"]),
                face_p95=number(latency["face_fresh_stage2"]["p95"]),
                knife_p95=number(latency["knife_fresh_stage2"]["p95"]),
            )
        )
    lines += [
        "",
        "`frame_total` 包含 YOLO、调度、画框，以及 UI 模式的缩放/JPEG 编码；不含源视频等待。fresh stage2 是实际新推理耗时。当前日志没有“提交检测到确认结果显示”的单调时间戳，因此没有伪造该端到端确认延迟。",
        "",
        "## CPU 占用",
        "",
        "| 视频 / 模式 | 应用 mean / P95 (% 单核) | 应用 mean (% 整机) | 应用 core-s / core-ms每帧 | 浏览器 mean (% 整机) / core-ms每帧 | 系统 mean / P95 (%) | 运行前基线 mean (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        cpu = run["cpu_pct"]
        cost = run["cpu_cost"]
        lines.append(
            "| {run_id} | {app_mean} / {app_p95} | {app_machine} | "
            "{app_seconds} / {app_per_frame} | {browser} / {browser_per_frame} | "
            "{system_mean} / {system_p95} | {baseline} |".format(
                run_id=run["run_id"],
                app_mean=number(cpu["app_core_equivalent"]["mean"]),
                app_p95=number(cpu["app_core_equivalent"]["p95"]),
                app_machine=number(cpu["app_machine_share"]["mean"]),
                app_seconds=number(cost["app"]["core_seconds"]),
                app_per_frame=number(cost["app"]["core_ms_per_frame"]),
                browser=number(cpu["browser_machine_share"]["mean"]),
                browser_per_frame=number(cost["browser"]["core_ms_per_frame"]),
                system_mean=number(cpu["system_total"]["mean"]),
                system_p95=number(cpu["system_total"]["p95"]),
                baseline=number(cpu["baseline_system_total"]["mean"]),
            )
        )
    lines += [
        "",
        f"进程 CPU 的单核口径允许超过 100%；整机口径已除以本机 {logical_cpus} 个逻辑 CPU。core-ms/帧按 `/proc` CPU tick 增量和精确采样窗口在媒体区间积分，六轮所需进程的最低窗口覆盖率为 {percent(minimum_cpu_coverage)}。系统数值包含 GNOME、NoMachine、VS Code 等背景负载，不能全部归因于识别程序。曲线淡线为 200 ms 原始采样，实线为约 1 秒滚动均值；短周期原始点受内核 tick 量化影响，不用于解释单点峰值。",
        "",
    ]
    if browser_graphics_warning:
        lines += [
            "UI 轮次出现 Chromium EGL/DRI 初始化告警，当前桌面环境可能没有正常使用图形加速。因此浏览器 CPU 是本机当前环境的实测值，不能直接外推到 GPU 加速正常的部署环境。",
            "",
        ]
    for key in ("all", "f1_f4", "k4_k5_k10"):
        if key in chart_names:
            lines += [f"![{key} CPU]({chart_names[key]})", ""]
    lines += ["![CPU summary](cpu_summary.png)", "", "## NPU 占用", ""]
    debug_available = any(run["npu"]["debugfs_available"] for run in runs)
    if debug_available:
        lines += [
            "本次可读取 RK3588 驱动的三核 1 秒窗口负载，以下是三核容量平均值。",
            "",
            "| 视频 / 模式 | NPU mean / P95 / max (%) | 频率 mean (MHz) |",
            "|---|---:|---:|",
        ]
        for run in runs:
            npu = run["npu"]
            debug = npu["debug_capacity_mean_pct"]
            frequency = npu["frequency_hz"]["mean"]
            lines.append(
                f"| {run['run_id']} | {number(debug['mean'])} / {number(debug['p95'])} / "
                f"{number(debug['max'])} | {number(frequency / 1e6 if frequency else None)} |"
            )
    else:
        lines += [
            "本次普通用户无法读取 `/sys/kernel/debug/rknpu/load`，因此没有把任何数值冒充真实 NPU 利用率。`/sys/class/devfreq/fdab0000.npu/load` 在设备挂起时仍固定显示 100%，已明确排除。",
            "",
            "下表仅给出 NPU **上电驻留率**，它包含上下文持有、等待和延迟关电，不等于计算占用，也不能归属到单个进程。",
            "",
            "| 视频 / 模式 | 上电驻留 mean / P95 (%) | 频率 mean (MHz) |",
            "|---|---:|---:|",
        ]
        for run in runs:
            npu = run["npu"]
            residency = npu["power_active_residency_pct"]
            frequency = npu["frequency_hz"]["mean"]
            lines.append(
                f"| {run['run_id']} | {number(residency['mean'])} / {number(residency['p95'])} | "
                f"{number(frequency / 1e6 if frequency else None)} |"
            )
        lines += [
            "",
            "要补齐真实 NPU 三核占用，可在重跑前临时执行 `sudo chmod o+x /sys/kernel/debug`；重启后权限会恢复。该操作不影响本次 CPU、延迟和识别输出结果。",
        ]
    lines += [
        "",
        "RKNN 还提供每次硬件推理的实际运行微秒数。下表将每核累计 `PERF_RUN` 除以媒体处理时长，作为本应用的 **NPU 计算时间占比代理**；它不是系统级硬件计数器，多上下文排队也会影响解释。Aggregate 再除以三核总容量。",
        "",
        "| 视频 / 模式 | Core0 YOLO (%) | Core1 Face (%) | Core2 Knife (%) | Aggregate capacity (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        proxy = run["npu"]["app_compute_time_proxy"]
        cores = proxy["core_compute_time_proxy_pct"]
        lines.append(
            f"| {run['run_id']} | {number(cores['0'])} | {number(cores['1'])} | "
            f"{number(cores['2'])} | {number(proxy['aggregate_capacity_proxy_pct'])} |"
        )
    lines += [
        "",
        "| 视频 | YOLO NPU 调用数 no-UI / UI | 单次 PERF_RUN no-UI / UI (ms) |",
        "|---|---:|---:|",
    ]
    for impact in impacts:
        lines.append(
            f"| {impact['video_key']} | {impact['npu_yolo_runs_no_ui']} / "
            f"{impact['npu_yolo_runs_ui']} | "
            f"{number(impact['npu_yolo_mean_ms_no_ui'])} / "
            f"{number(impact['npu_yolo_mean_ms_ui'])} |"
        )
    lines += [
        "",
        "每对运行的 YOLO NPU 调用次数相同，单次硬件推理时间也基本不变；UI 组的计算时间占比代理较低，是因为 CPU/JPEG/浏览器让整段 wall time 变长，不代表 NPU 模型本身更快。",
    ]
    lines += [
        "",
        "## 识别代理指标",
        "",
        "| 视频 / 模式 | 全部 / 目标大类 fresh shadow | 最终接受率 | 语义拒识率 | confirming 率 | 文件名 token 一致率 | 目标记录 token yield |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        proxy = run["recognition_proxy"]
        lines.append(
            f"| {run['run_id']} | {run['shadow_records']} / {proxy['records']} | "
            f"{percent(proxy['accepted_rate'])} | "
            f"{percent(proxy['semantic_rejection_rate'])} | {percent(proxy['confirming_rate'])} | "
            f"{percent(proxy['filename_token_consistency'])} | "
            f"{percent(proxy['target_record_token_yield'])} |"
        )
    lines += [
        "",
        "文件名 token 一致率的分母是最终 accepted；目标记录 token yield 的分母是对应大类的所有 fresh shadow。两者都只是录像命名一致性代理，不是真值正确率。`confirming` 是时序待确认，不算拒识。完整预测分布见 `summary.json`。",
        "",
        "### 现有稀疏标注的条件准确率",
        "",
        "| 视频 / 模式 / 大类 | 稀疏 GT 帧 / 框 | 可观测帧率 | IoU 对齐框率 | Raw 条件准确率 | Final 条件准确率 | Accepted precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for major, sparse in sorted(run["sparse_reference"].items()):
            lines.append(
                f"| {run['run_id']} / {major} | {sparse['ground_truth_frames']} / "
                f"{sparse['ground_truth_boxes']} | {percent(sparse['frame_observability'])} | "
                f"{percent(sparse['iou_alignment_coverage'])} | "
                f"{percent(sparse['raw_conditional_accuracy'])} | "
                f"{percent(sparse['final_conditional_accuracy'])} | "
                f"{percent(sparse['accepted_precision'])} |"
            )
    lines += [
        "",
        "这里仅在“同 raw 帧、同大类、bbox 一对一 IoU >= 0.5 且该框恰好写入 fresh shadow”的子集上计算身份类别。未出现在 shadow 的稀疏 GT 不能算漏检，因为 shadow 不是逐帧 stage-1 trace。`all/k` 刀具标注分别来自分类数据的 test/train；人脸项就是当前 gallery reference，自匹配偏差很大。",
        "",
        "## UI / 无 UI 同源一致性",
        "",
        "| 视频 | provenance 相同 | no-UI / UI 记录 | 配对记录 | 未配对 no-UI / UI | 原始二阶段一致率 | 最终决策一致率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        lines.append(
            f"| {comparison['video_key']} | {'是' if comparison['same_input_provenance'] else '否'} | "
            f"{comparison['no_ui_records']} / {comparison['ui_records']} | "
            f"{comparison['paired_records']} | {comparison['unpaired_no_ui']} / "
            f"{comparison['unpaired_ui']} | {percent(comparison['raw_stage2_equality_rate'])} | "
            f"{percent(comparison['final_decision_equality_rate'])} |"
        )
    lines += [
        "",
        "比较按 `shadow.source.id` 对齐，未对齐项再用同帧、同大类且 bbox IoU >= 0.9 补配；时间戳和耗时不参与数值一致性比较。UI 改变异步结果落在哪一帧属于真实运行行为差异，不自动等同于模型数值变差。",
        "",
        "## 严格准确率缺口",
        "",
        "要得到可对外使用的正确率，需要人工标注评估帧中的全部 GT 框、身份/刀具类别和 unknown，并按同帧 IoU >= 0.5 一对一匹配。漏框必须计错，同时分别报告 detection precision/recall、匹配框分类准确率和 unknown rejection。现有 golden evaluator 只能评价已经进入 shadow 的框，不能发现 YOLO 漏检。",
        "",
        "## 可复现性",
        "",
        "`benchmark.json` 保存三段输入视频、运行二进制、基准脚本、分析脚本、四个 RKNN 模型、人脸图库和稀疏标注 manifest 的 SHA-256。六轮均显式使用 `--provider rknn`，因此本报告不依赖 `auto` provider 路由。",
        "",
    ]
    return "\n".join(lines)


def flatten_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    latency = run["latency_ms"]
    cpu = run["cpu_pct"]
    npu = run["npu"]
    proxy = run["recognition_proxy"]
    return {
        "run_id": run["run_id"],
        "video_key": run["video_key"],
        "ui": run["ui"],
        "processed_frames": run["processed_frames"],
        "effective_fps": run["effective_fps"],
        "frame_total_mean_ms": latency["frame_total"]["mean"],
        "frame_total_p50_ms": latency["frame_total"]["p50"],
        "frame_total_p95_ms": latency["frame_total"]["p95"],
        "frame_total_p99_ms": latency["frame_total"]["p99"],
        "yolo_mean_ms": latency["yolo"]["mean"],
        "yolo_p95_ms": latency["yolo"]["p95"],
        "face_stage2_p95_ms": latency["face_fresh_stage2"]["p95"],
        "knife_stage2_p95_ms": latency["knife_fresh_stage2"]["p95"],
        "app_cpu_core_mean_pct": cpu["app_core_equivalent"]["mean"],
        "app_cpu_core_p95_pct": cpu["app_core_equivalent"]["p95"],
        "app_cpu_machine_mean_pct": cpu["app_machine_share"]["mean"],
        "browser_cpu_machine_mean_pct": cpu["browser_machine_share"]["mean"],
        "system_cpu_mean_pct": cpu["system_total"]["mean"],
        "system_cpu_p95_pct": cpu["system_total"]["p95"],
        "npu_debug_mean_pct": npu["debug_capacity_mean_pct"]["mean"],
        "npu_power_active_mean_pct": npu["power_active_residency_pct"]["mean"],
        "npu_core0_compute_proxy_pct": npu["app_compute_time_proxy"][
            "core_compute_time_proxy_pct"
        ]["0"],
        "npu_core1_compute_proxy_pct": npu["app_compute_time_proxy"][
            "core_compute_time_proxy_pct"
        ]["1"],
        "npu_core2_compute_proxy_pct": npu["app_compute_time_proxy"][
            "core_compute_time_proxy_pct"
        ]["2"],
        "npu_aggregate_capacity_proxy_pct": npu["app_compute_time_proxy"][
            "aggregate_capacity_proxy_pct"
        ],
        "filename_token_consistency": proxy["filename_token_consistency"],
        "target_record_token_yield": proxy["target_record_token_yield"],
        "accepted_rate": proxy["accepted_rate"],
        "semantic_rejection_rate": proxy["semantic_rejection_rate"],
        "app_cpu_core_seconds": run["cpu_cost"]["app"]["core_seconds"],
        "app_cpu_core_ms_per_frame": run["cpu_cost"]["app"][
            "core_ms_per_frame"
        ],
        "browser_cpu_core_ms_per_frame": run["cpu_cost"]["browser"][
            "core_ms_per_frame"
        ],
    }


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    benchmark = read_json(root / "benchmark.json")
    run_metadata = benchmark.get("runs", [])
    run_ids = [str(meta.get("run_id")) for meta in run_metadata]
    if (
        len(run_metadata) != 6
        or len(set(run_ids)) != 6
        or set(run_ids) != EXPECTED_RUN_IDS
        or any(meta.get("status") != "completed" for meta in run_metadata)
    ):
        raise SystemExit("Expected six unique completed benchmark runs")
    metas = {str(meta["run_id"]): meta for meta in benchmark["runs"]}
    sparse_ground_truth = load_sparse_ground_truth(Path(benchmark["project_root"]))
    analyzed: list[dict[str, Any]] = []
    resources: dict[str, list[dict[str, Any]]] = {}
    for meta in benchmark["runs"]:
        metrics, run_resources = analyze_run(
            meta, sparse_ground_truth[str(meta["video_key"])]
        )
        analyzed.append(metrics)
        resources[str(meta["run_id"])] = run_resources
        required_costs = [metrics["cpu_cost"]["app"]]
        if bool(metrics["ui"]):
            required_costs.append(metrics["cpu_cost"]["browser"])
        if not metrics["complete_frame_log"] or any(
            cost["source"] != "proc_tick_delta"
            or cost["media_window_coverage"] is None
            or float(cost["media_window_coverage"]) < 0.99
            for cost in required_costs
        ):
            raise SystemExit(
                f"Incomplete frame log or CPU sampling coverage for {meta['run_id']}"
            )
    analyzed.sort(
        key=lambda run: (
            list(RECORDING_IDS).index(str(run["video_key"])),
            bool(run["ui"]),
        )
    )
    by_video: dict[str, dict[bool, dict[str, Any]]] = {}
    for run in analyzed:
        by_video.setdefault(str(run["video_key"]), {})[bool(run["ui"])] = run
    comparisons = [
        compare_pair(modes[False], modes[True])
        for _, modes in sorted(by_video.items())
        if False in modes and True in modes
    ]
    impacts = ui_impact(by_video)

    chart_names: dict[str, str] = {}
    for video_key, modes in sorted(by_video.items()):
        if False not in modes or True not in modes:
            continue
        filename = f"cpu_{video_key}.png"
        plot_cpu_pair(
            root / filename,
            video_key,
            modes[False],
            modes[True],
            resources,
            metas,
        )
        chart_names[video_key] = filename
    plot_cpu_summary(root / "cpu_summary.png", analyzed)

    output = {
        "schema": "insight_cup_offline_benchmark_analysis.v1",
        "benchmark": str(root / "benchmark.json"),
        "strict_accuracy_available": False,
        "accuracy_limit": (
            "No exhaustive human ground truth; filename-set metrics are proxies and "
            "the clips overlap training/gallery sources."
        ),
        "runs": analyzed,
        "ui_no_ui_consistency": comparisons,
        "ui_impact": impacts,
        "charts": {**chart_names, "summary": "cpu_summary.png"},
    }
    (root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [flatten_summary(run) for run in analyzed]
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "REPORT.md").write_text(
        markdown_report(benchmark, analyzed, comparisons, chart_names, impacts),
        encoding="utf-8",
    )
    print(f"Analysis written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
