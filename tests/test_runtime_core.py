from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.app.contracts import (
    DetectedRegion,
    Stage1Detection,
    Stage2Decision,
    YoloDebugInfo,
)
from insight_cup.app.cli import _build_config, build_parser
from insight_cup.app.detector import (
    collapse_yolo_class,
    suppress_major_class_duplicates,
)
from insight_cup.app.logging_utils import EventJournal
from insight_cup.app.router import RecognitionRouter
from insight_cup.app.state import RuntimeState
from insight_cup.app.sources import is_realsense_source, is_video_source
from insight_cup.app.stage2 import (
    AsyncStage2Scheduler,
    TemporalDecisionSmoother,
    bbox_iou,
)
from insight_cup.app.web import create_debug_server
from insight_cup.common.onnx import cpu_session_options


def detection(
    major: str = "f",
    frame_index: int = 1,
    bbox: tuple[int, int, int, int] = (1, 2, 8, 9),
) -> Stage1Detection:
    return Stage1Detection(
        detection_id=f"frame{frame_index:06d}:det000",
        frame_index=frame_index,
        major_class=major,  # type: ignore[arg-type]
        confidence=0.91,
        bbox_xyxy=bbox,
        crop=np.full((12, 10, 3), 127, dtype=np.uint8),
    )


def detected_region(
    major: str,
    raw_name: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
) -> DetectedRegion:
    item = detection(major=major, bbox=bbox)
    item = Stage1Detection(
        detection_id=item.detection_id,
        frame_index=item.frame_index,
        major_class=item.major_class,
        confidence=confidence,
        bbox_xyxy=item.bbox_xyxy,
        crop=item.crop,
    )
    return DetectedRegion(
        handoff=item,
        debug=YoloDebugInfo(raw_class_id=0, raw_class_name=raw_name),
    )


class StubFaceService:
    def __init__(self) -> None:
        self.received = []

    def recognize(self, crop: np.ndarray) -> SimpleNamespace:
        self.received.append(crop)
        return SimpleNamespace(
            status="matched",
            predicted_class="f3",
            candidate_class="f3",
            similarity=0.82,
            second_best_class="f4",
            second_best_similarity=0.44,
            margin=0.38,
            face_detection_confidence=0.96,
            face_bbox_xyxy=(1.0, 1.0, 8.0, 9.0),
        )


class StubKnifeService:
    def __init__(self) -> None:
        self.received = []

    def recognize(self, crop: np.ndarray) -> SimpleNamespace:
        self.received.append(crop)
        return SimpleNamespace(
            status="matched",
            predicted_class="k4",
            candidate_class="k4",
            confidence=0.88,
            second_best_class="k2",
            second_best_confidence=0.07,
            margin=0.81,
        )


class RuntimeContractTests(unittest.TestCase):
    def test_start_script_defaults_to_no_ui_and_allows_explicit_ui(self) -> None:
        environment = {**os.environ, "INSIGHT_CUP_BIN": "/bin/echo"}
        default = subprocess.run(
            [str(PROJECT_ROOT / "start.sh"), "--source", "sample.mp4"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        explicit_ui = subprocess.run(
            [str(PROJECT_ROOT / "start.sh"), "--ui", "--source", "sample.mp4"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        self.assertEqual(
            default,
            "--source sample.mp4 --no-ui",
        )
        self.assertEqual(explicit_ui, "--ui --source sample.mp4")

    def test_debug_ui_opens_browser_by_default(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertTrue(defaults.open_browser)
        self.assertEqual(defaults.source, "realsense")
        self.assertEqual(defaults.face_cpu_threads, 1)
        self.assertEqual(defaults.face_det_size, 320)
        self.assertEqual(defaults.face_rotation_retry_degrees, 0.0)
        self.assertEqual(defaults.preview_width, 960)
        self.assertEqual(defaults.jpeg_quality, 82)
        self.assertEqual(defaults.frame_log_every, 30)
        self.assertEqual(defaults.stage2_workers, 2)
        self.assertEqual(defaults.stage2_refresh_frames, 15)
        self.assertEqual(defaults.stage2_stable_refresh_frames, 30)
        self.assertEqual(defaults.knife_cpu_threads, 1)
        self.assertEqual(defaults.stage2_smoothing, "vote")
        self.assertEqual(defaults.stage2_smoothing_window, 3)
        self.assertEqual(defaults.stage2_switch_confirmations, 2)
        self.assertEqual(defaults.knife_mode, "classification")
        self.assertEqual(defaults.yolo_cpu_threads, 1)
        self.assertFalse(defaults.unthrottled_video)
        self.assertTrue(_build_config(defaults).pace_video)
        self.assertEqual(_build_config(defaults).stage2_stable_refresh_frames, 30)
        self.assertFalse(
            _build_config(
                parser.parse_args(["--unthrottled-video"])
            ).pace_video
        )
        self.assertEqual(
            parser.parse_args(["--knife-mode", "retrieval"]).knife_mode,
            "retrieval",
        )
        self.assertFalse(parser.parse_args(["--no-open-browser"]).open_browser)
        self.assertTrue(_build_config(defaults).publish_preview)
        self.assertFalse(
            _build_config(parser.parse_args(["--no-ui"])).publish_preview
        )

    def test_realsense_source_aliases_are_explicit(self) -> None:
        self.assertTrue(is_realsense_source("realsense"))
        self.assertTrue(is_realsense_source("D455"))
        self.assertFalse(is_realsense_source("0"))
        self.assertFalse(is_realsense_source("camera.mp4"))

    def test_video_source_uses_camera_style_pipeline(self) -> None:
        self.assertTrue(is_video_source("camera.mp4"))
        self.assertFalse(is_video_source("0"))
        self.assertFalse(is_video_source("images"))

    def test_yolo_classes_are_collapsed_to_public_classes(self) -> None:
        self.assertEqual(collapse_yolo_class("b0"), "b0")
        self.assertEqual(collapse_yolo_class("F10"), "f")
        self.assertEqual(collapse_yolo_class("k2"), "k")
        self.assertIsNone(collapse_yolo_class("person"))

    def test_major_class_nms_removes_cross_subclass_duplicate(self) -> None:
        weaker = detected_region("f", "f8", 0.71, (10, 10, 50, 50))
        stronger = detected_region("f", "f10", 0.92, (11, 11, 51, 51))
        kept = suppress_major_class_duplicates([weaker, stronger], 0.45)
        self.assertEqual(kept, [stronger])

    def test_major_class_nms_keeps_different_major_classes(self) -> None:
        face = detected_region("f", "f2", 0.91, (10, 10, 50, 50))
        knife = detected_region("k", "k2", 0.90, (10, 10, 50, 50))
        kept = suppress_major_class_duplicates([face, knife], 0.45)
        self.assertCountEqual(kept, [face, knife])

    def test_major_class_nms_keeps_separated_same_class_boxes(self) -> None:
        left = detected_region("f", "f2", 0.91, (0, 0, 20, 20))
        right = detected_region("f", "f8", 0.90, (30, 0, 50, 20))
        kept = suppress_major_class_duplicates([left, right], 0.45)
        self.assertCountEqual(kept, [left, right])

    def test_stage1_handoff_excludes_raw_yolo_subclass_and_pixels(self) -> None:
        payload = detection().handoff_dict("crops/f/test.jpg")
        self.assertEqual(payload["major_class"], "f")
        self.assertNotIn("raw_yolo_class", payload)
        self.assertNotIn("crop", payload)

    def test_router_uses_only_major_class_to_select_stage2(self) -> None:
        face = StubFaceService()
        knife = StubKnifeService()
        router = RecognitionRouter(face, knife)

        face_result = router.route(detection("f"))
        knife_result = router.route(detection("k"))
        b0_result = router.route(detection("b0"))

        self.assertEqual(face_result.predicted_class, "f3")
        self.assertEqual(face_result.score_kind, "cosine_similarity")
        self.assertEqual(face_result.second_best_class, "f4")
        self.assertEqual(knife_result.predicted_class, "k4")
        self.assertEqual(knife_result.score_kind, "softmax_confidence")
        self.assertEqual(b0_result.status, "passthrough")
        self.assertEqual(len(face.received), 1)
        self.assertEqual(len(knife.received), 1)

    def test_event_console_log_contains_only_ranked_result_fields(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("recognition-test-event-format")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        event = {
            "id": "frame000007:det000",
            "frame": {"index": 7},
            "stage1": {"major_class": "f", "confidence": 0.91},
            "route": {"module": "face", "execution": "fresh"},
            "stage2": {
                "status": "unknown",
                "candidate_class": "f3",
                "score": 0.48,
                "second_best_class": "f4",
                "second_best_score": 0.31,
            },
            "timing_ms": {"stage2": 210.0},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = EventJournal(Path(temp_dir) / "events.jsonl", logger)
            try:
                journal.write(event)
            finally:
                journal.close()

        line = stream.getvalue().strip()
        self.assertEqual(
            line,
            "RESULT frame=7 id=frame000007:det000 label=f3 probability=0.480 "
            "second=f4 second_probability=0.310",
        )
        self.assertNotIn("route=", line)
        self.assertNotIn("stage2_ms=", line)


class RuntimeStateTests(unittest.TestCase):
    def test_pause_resume_stop_and_event_counters(self) -> None:
        state = RuntimeState()
        state.configure_session("0", "test", Path("/tmp/test"))
        state.set_status("running")
        event = {
            "stage1": {"major_class": "f"},
            "stage2": {"status": "matched"},
        }
        state.record_detection("f")
        state.record_event(event)
        self.assertEqual(state.snapshot()["major_counts"]["f"], 1)
        self.assertEqual(state.snapshot()["status_counts"]["matched"], 1)

        self.assertTrue(state.control("pause")[0])
        self.assertEqual(state.snapshot()["status"], "paused")
        self.assertTrue(state.control("resume")[0])
        self.assertEqual(state.snapshot()["status"], "running")
        self.assertTrue(state.control("stop")[0])
        self.assertTrue(state.stop_requested)

    def test_headless_state_keeps_counts_without_ui_payloads(self) -> None:
        state = RuntimeState(retain_events=False)
        state.record_event(
            {
                "stage1": {"major_class": "f"},
                "stage2": {"status": "matched"},
            }
        )
        state.publish_frame(
            jpeg=None,
            frame_index=7,
            processed_frames=8,
            fps=24.0,
            timing_ms={"yolo": 1.0, "stage2": 0.0, "total": 2.0},
            detection_count=2,
        )
        snapshot = state.snapshot()
        self.assertEqual(snapshot["status_counts"], {"matched": 1})
        self.assertEqual(snapshot["processed_frames"], 8)
        self.assertFalse(snapshot["has_frame"])
        self.assertEqual(state.recent_events(), [])


class AsyncStage2SchedulerTests(unittest.TestCase):
    def test_synchronous_mode_preserves_offline_frame_results(self) -> None:
        scheduler = AsyncStage2Scheduler(asynchronous=False)
        calls: list[str] = []

        def recognize(item: Stage1Detection) -> Stage2Decision:
            calls.append(item.detection_id)
            return Stage2Decision(module="face", status="no_face")

        try:
            result = scheduler.resolve(
                detection("f", 1),
                recognize,
                prepare_job=lambda: "crops/f/source.jpg",
            )
            self.assertEqual(result.mode, "direct")
            self.assertEqual(result.decision.status, "no_face")
            self.assertEqual(result.source_crop_path, "crops/f/source.jpg")
            self.assertEqual(calls, ["frame000001:det000"])
        finally:
            scheduler.close()

    def test_spatial_result_is_async_reused_and_refreshed(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=3, iou_threshold=0.5)
        calls: list[str] = []
        prepared: list[str] = []
        release = threading.Event()

        def recognize(item: Stage1Detection) -> Stage2Decision:
            release.wait(timeout=1.0)
            calls.append(item.detection_id)
            return Stage2Decision(
                module="face",
                status="matched",
                predicted_class="f3",
            )

        def prepare() -> str:
            path = f"crop-{len(prepared)}.jpg"
            prepared.append(path)
            return path

        try:
            first = scheduler.resolve(detection("f", 1), recognize, prepare)
            self.assertEqual(first.mode, "processing")
            release.set()
            self.assertTrue(scheduler.wait_for_idle(1.0))

            fresh = scheduler.resolve(
                detection("f", 2, (2, 2, 9, 9)), recognize, prepare
            )
            self.assertEqual(fresh.mode, "fresh")
            self.assertEqual(fresh.decision.predicted_class, "f3")
            self.assertEqual(fresh.source_crop_path, "crop-0.jpg")

            cached = scheduler.resolve(
                detection("f", 3, (2, 3, 9, 10)), recognize, prepare
            )
            self.assertEqual(cached.mode, "cached")
            self.assertEqual(len(calls), 1)

            refreshing = scheduler.resolve(
                detection("f", 4, (3, 3, 10, 10)), recognize, prepare
            )
            self.assertEqual(refreshing.mode, "cached")
            self.assertTrue(scheduler.wait_for_idle(1.0))
            refreshed = scheduler.resolve(
                detection("f", 5, (3, 3, 10, 10)), recognize, prepare
            )
            self.assertEqual(refreshed.mode, "fresh")
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(prepared), 2)
        finally:
            scheduler.close()

    def test_confirmed_result_uses_longer_refresh_interval(self) -> None:
        scheduler = AsyncStage2Scheduler(
            refresh_frames=3,
            stable_refresh_frames=6,
            iou_threshold=0.5,
        )
        calls: list[str] = []
        release_first = threading.Event()
        release_second = threading.Event()

        def recognize(item: Stage1Detection) -> Stage2Decision:
            calls.append(item.detection_id)
            if len(calls) == 1:
                release_first.wait(timeout=1.0)
            elif len(calls) == 2:
                release_second.wait(timeout=1.0)
            return Stage2Decision(
                module="face",
                status="matched",
                predicted_class="f2",
                candidate_class="f2",
            )

        try:
            first = scheduler.resolve(detection("f", 1), recognize)
            self.assertEqual(first.mode, "processing")
            release_first.set()
            self.assertTrue(scheduler.wait_for_idle(1.0))
            fresh = scheduler.resolve(detection("f", 2), recognize)
            self.assertEqual(fresh.mode, "fresh")

            before_stable_refresh = scheduler.resolve(
                detection("f", 4), recognize
            )
            self.assertEqual(before_stable_refresh.mode, "cached")
            self.assertEqual(len(calls), 1)

            refreshing = scheduler.resolve(detection("f", 7), recognize)
            self.assertEqual(refreshing.mode, "cached")
            release_second.set()
            self.assertTrue(scheduler.wait_for_idle(1.0))
            refreshed = scheduler.resolve(detection("f", 8), recognize)
            self.assertEqual(refreshed.mode, "fresh")
            self.assertEqual(len(calls), 2)
        finally:
            scheduler.close()

    def test_onnx_cpu_policy_bounds_threads_and_disables_spinning(self) -> None:
        import onnxruntime as ort

        options = cpu_session_options(ort, 2)
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(
            options.get_session_config_entry("session.intra_op.allow_spinning"),
            "0",
        )
        self.assertEqual(
            options.get_session_config_entry("session.inter_op.allow_spinning"),
            "0",
        )

    def test_finalize_results_collects_result_after_last_frame(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=10, iou_threshold=0.5)
        release = threading.Event()

        def recognize(item: Stage1Detection) -> Stage2Decision:
            release.wait(timeout=1.0)
            return Stage2Decision(module="face", status="matched", predicted_class="f2")

        try:
            first = scheduler.resolve(detection("f", 1), recognize)
            self.assertEqual(first.mode, "processing")
            release.set()
            self.assertTrue(scheduler.wait_for_idle(1.0))
            results = scheduler.finalize_results()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].detection.detection_id, "frame000001:det000")
            self.assertEqual(results[0].resolution.decision.predicted_class, "f2")
            self.assertEqual(scheduler.finalize_results(), [])
        finally:
            scheduler.close()

    def test_non_overlapping_box_gets_a_distinct_track(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=10, iou_threshold=0.5)

        def recognize(item: Stage1Detection) -> Stage2Decision:
            return Stage2Decision(module="knife", status="unknown")

        try:
            first = scheduler.resolve(
                detection("k", 1, (0, 0, 10, 10)), recognize
            )
            second = scheduler.resolve(
                detection("k", 1, (30, 30, 40, 40)), recognize
            )
            self.assertNotEqual(first.track_id, second.track_id)
            self.assertEqual(bbox_iou((0, 0, 10, 10), (30, 30, 40, 40)), 0.0)
        finally:
            scheduler.close()

    def test_nearby_box_with_low_iou_reuses_track(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=10, iou_threshold=0.55)

        def recognize(item: Stage1Detection) -> Stage2Decision:
            return Stage2Decision(module="face", status="matched")

        try:
            first = scheduler.resolve(
                detection("f", 1, (100, 100, 160, 180)), recognize
            )
            self.assertTrue(scheduler.wait_for_idle(1.0))
            moved = scheduler.resolve(
                detection("f", 2, (118, 104, 178, 184)), recognize
            )
            self.assertLess(
                bbox_iou((100, 100, 160, 180), (118, 104, 178, 184)),
                0.55,
            )
            self.assertEqual(moved.track_id, first.track_id)
            self.assertNotEqual(moved.mode, "processing")
        finally:
            scheduler.close()

    def test_two_nearby_detections_cannot_share_one_track(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=10, iou_threshold=0.55)

        def recognize(item: Stage1Detection) -> Stage2Decision:
            return Stage2Decision(module="face", status="matched")

        try:
            first = scheduler.resolve(
                detection("f", 1, (100, 100, 160, 180)), recognize
            )
            second = scheduler.resolve(
                detection("f", 1, (180, 100, 240, 180)), recognize
            )
            self.assertNotEqual(first.track_id, second.track_id)
            self.assertTrue(scheduler.wait_for_idle(1.0))

            first_moved = scheduler.resolve(
                detection("f", 2, (116, 102, 176, 182)), recognize
            )
            second_moved = scheduler.resolve(
                detection("f", 2, (164, 102, 224, 182)), recognize
            )
            self.assertEqual(first_moved.track_id, first.track_id)
            self.assertEqual(second_moved.track_id, second.track_id)
        finally:
            scheduler.close()

    def test_frame_association_uses_motion_when_tracks_cross(self) -> None:
        scheduler = AsyncStage2Scheduler(refresh_frames=10, iou_threshold=0.55)

        def recognize(item: Stage1Detection) -> Stage2Decision:
            return Stage2Decision(module="face", status="matched")

        try:
            initial = scheduler.resolve_frame(
                [
                    detection("f", 1, (0, 0, 20, 20)),
                    detection("f", 1, (60, 0, 80, 20)),
                ],
                recognize,
            )
            scheduler.resolve_frame(
                [
                    detection("f", 2, (12, 0, 32, 20)),
                    detection("f", 2, (48, 0, 68, 20)),
                ],
                recognize,
            )
            crossed_reversed = scheduler.resolve_frame(
                [
                    detection("f", 3, (34, 0, 54, 20)),
                    detection("f", 3, (26, 0, 46, 20)),
                ],
                recognize,
            )
            self.assertEqual(crossed_reversed[0].track_id, initial[1].track_id)
            self.assertEqual(crossed_reversed[1].track_id, initial[0].track_id)
        finally:
            scheduler.close()


class TemporalDecisionSmootherTests(unittest.TestCase):
    @staticmethod
    def decision(label: str, first: float, second: float) -> Stage2Decision:
        other = "k2" if label == "k1" else "k1"
        return Stage2Decision(
            module="knife",
            status="matched",
            predicted_class=label,
            candidate_class=label,
            score=first,
            score_kind="softmax_confidence",
            second_best_class=other,
            second_best_score=second,
            margin=first - second,
            class_scores={label: first, other: second},
        )

    def test_mean_ignores_one_frame_spike_and_requires_confirmed_switch(self) -> None:
        smoother = TemporalDecisionSmoother("mean", 3, 2, 0.0, 0.0)
        first = smoother.update(self.decision("k1", 0.9, 0.1))
        spike = smoother.update(self.decision("k2", 0.9, 0.1))
        recovered = smoother.update(self.decision("k1", 0.9, 0.1))
        pending = smoother.update(self.decision("k2", 0.9, 0.1))
        pending_again = smoother.update(self.decision("k2", 0.9, 0.1))
        switched = smoother.update(self.decision("k2", 0.9, 0.1))

        self.assertEqual(first.status, "confirming")
        self.assertIsNone(first.predicted_class)
        self.assertEqual(spike.predicted_class, "k1")
        self.assertEqual(recovered.predicted_class, "k1")
        self.assertEqual(pending.predicted_class, "k1")
        self.assertEqual(pending_again.predicted_class, "k2")
        self.assertEqual(switched.predicted_class, "k2")
        self.assertEqual(switched.metadata["temporal"]["method"], "mean")

    def test_vote_switches_after_a_repeated_majority(self) -> None:
        smoother = TemporalDecisionSmoother("vote", 3, 2, 0.0, 0.0)
        first = smoother.update(self.decision("k1", 0.9, 0.1))
        second = smoother.update(self.decision("k2", 0.9, 0.1))
        self.assertEqual(first.status, "confirming")
        self.assertEqual(second.status, "confirming")
        self.assertEqual(
            smoother.update(self.decision("k2", 0.9, 0.1)).predicted_class,
            "k2",
        )

    def test_transient_no_face_holds_then_expires_stable_identity(self) -> None:
        smoother = TemporalDecisionSmoother("mean", 3, 2, 0.0, 0.0)
        smoother.update(self.decision("k1", 0.9, 0.1))
        smoother.update(self.decision("k1", 0.9, 0.1))
        missing = Stage2Decision(module="knife", status="unknown")

        self.assertEqual(smoother.update(missing).predicted_class, "k1")
        self.assertEqual(smoother.update(missing).predicted_class, "k1")
        expired = smoother.update(missing)
        self.assertEqual(expired.status, "unknown")
        self.assertIsNone(expired.predicted_class)

    def test_internal_class_scores_are_not_serialized(self) -> None:
        decision = self.decision("k1", 0.9, 0.1)
        self.assertNotIn("class_scores", decision.to_dict())


class DebugHTTPTests(unittest.TestCase):
    def test_state_events_and_control_apis(self) -> None:
        state = RuntimeState()
        state.configure_session("video.mp4", "http-test", Path("/tmp/http-test"))
        state.set_status("running")
        server = create_debug_server("127.0.0.1", 0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(f"{base}/api/state", timeout=3) as response:
                payload = json.load(response)
            self.assertEqual(payload["session_id"], "http-test")

            request = urllib.request.Request(
                f"{base}/api/control",
                data=json.dumps({"action": "pause"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                control = json.load(response)
            self.assertTrue(control["accepted"])
            self.assertEqual(control["state"]["status"], "paused")

            with urllib.request.urlopen(f"{base}/", timeout=3) as response:
                page = response.read().decode("utf-8")
            self.assertIn("识别调试台", page)

            jpeg = b"\xff\xd8test-frame\xff\xd9"
            with urllib.request.urlopen(
                f"{base}/api/stream.mjpg", timeout=3
            ) as response:
                self.assertIn(
                    "multipart/x-mixed-replace",
                    response.headers["Content-Type"],
                )
                state.publish_frame(
                    jpeg=jpeg,
                    frame_index=1,
                    processed_frames=1,
                    fps=30.0,
                    timing_ms={"yolo": 1.0, "stage2": 0.0, "total": 2.0},
                    detection_count=0,
                )
                self.assertEqual(response.readline().strip(), b"--insightcupframe")
                self.assertEqual(response.readline().strip(), b"Content-Type: image/jpeg")
                self.assertEqual(
                    response.readline().strip(),
                    f"Content-Length: {len(jpeg)}".encode("ascii"),
                )
                self.assertEqual(response.readline(), b"\r\n")
                self.assertEqual(response.read(len(jpeg)), jpeg)

            head_request = urllib.request.Request(f"{base}/", method="HEAD")
            with urllib.request.urlopen(head_request, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(int(response.headers["Content-Length"]), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
