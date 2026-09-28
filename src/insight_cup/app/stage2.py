"""Asynchronous spatial scheduling for expensive stage-2 recognition."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Literal

from .contracts import MajorClass, Stage1Detection, Stage2Decision


ExecutionMode = Literal["direct", "processing", "fresh", "cached"]
SmoothingMethod = Literal["none", "mean", "vote"]
Recognize = Callable[[Stage1Detection], Stage2Decision]
PrepareJob = Callable[[], str | None]


@dataclass(frozen=True)
class Stage2Resolution:
    decision: Stage2Decision
    mode: ExecutionMode
    inference_ms: float = 0.0
    track_id: int | None = None
    cache_age_frames: int | None = None
    source_detection_id: str | None = None
    source_crop_path: str | None = None

    @property
    def has_new_result(self) -> bool:
        return self.mode in {"direct", "fresh"}


@dataclass(frozen=True)
class FinalStage2Result:
    """A completed result that was not consumed by a later video frame."""

    resolution: Stage2Resolution
    detection: Stage1Detection


class TemporalDecisionSmoother:
    """Stabilize one spatial track without exposing YOLO subclasses."""

    def __init__(
        self,
        method: SmoothingMethod,
        window_size: int,
        switch_confirmations: int,
        threshold: float,
        min_margin: float,
    ) -> None:
        if method not in {"mean", "vote"}:
            raise ValueError("Temporal smoothing method must be mean or vote")
        if window_size <= 0 or switch_confirmations <= 0:
            raise ValueError("Temporal smoothing window and confirmations must be positive")
        self.method = method
        self.window_size = int(window_size)
        self.switch_confirmations = int(switch_confirmations)
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self._history: deque[Stage2Decision] = deque(maxlen=self.window_size)
        self._stable_class: str | None = None
        self._pending_class: str | None = None
        self._pending_count = 0
        self._uncertain_count = 0
        self._last_output: Stage2Decision | None = None

    @staticmethod
    def _decision_scores(decision: Stage2Decision) -> dict[str, float]:
        if decision.class_scores:
            return {
                str(label): float(score)
                for label, score in decision.class_scores.items()
            }
        scores: dict[str, float] = {}
        if decision.candidate_class is not None and decision.score is not None:
            scores[decision.candidate_class] = float(decision.score)
        if (
            decision.second_best_class is not None
            and decision.second_best_score is not None
        ):
            scores[decision.second_best_class] = float(decision.second_best_score)
        return scores

    @staticmethod
    def _raw_summary(decision: Stage2Decision) -> dict[str, object]:
        return {
            "status": decision.status,
            "predicted_class": decision.predicted_class,
            "candidate_class": decision.candidate_class,
            "score": decision.score,
            "second_best_class": decision.second_best_class,
            "second_best_score": decision.second_best_score,
            "margin": decision.margin,
        }

    def _metadata(
        self,
        decision: Stage2Decision,
        proposed_class: str | None,
    ) -> dict[str, object]:
        return {
            **decision.metadata,
            "temporal": {
                "method": self.method,
                "window_size": self.window_size,
                "samples": len(self._history),
                "switch_confirmations": self.switch_confirmations,
                "stable_class": self._stable_class,
                "proposed_class": proposed_class,
                "pending_class": self._pending_class,
                "pending_count": self._pending_count,
                "raw": self._raw_summary(decision),
            },
        }

    def _hold_or_reject(self, decision: Stage2Decision) -> Stage2Decision:
        self._uncertain_count += 1
        self._pending_class = None
        self._pending_count = 0
        if (
            self._stable_class is not None
            and self._last_output is not None
            and self._uncertain_count < self.window_size
        ):
            held = Stage2Decision(
                module=decision.module,
                status="matched",
                predicted_class=self._stable_class,
                candidate_class=self._stable_class,
                score=self._last_output.score,
                score_kind=self._last_output.score_kind,
                second_best_class=self._last_output.second_best_class,
                second_best_score=self._last_output.second_best_score,
                margin=self._last_output.margin,
                class_scores=dict(self._last_output.class_scores),
                metadata=self._metadata(decision, None),
            )
            self._last_output = held
            return held
        self._stable_class = None
        rejected = Stage2Decision(
            module=decision.module,
            status=decision.status,
            predicted_class=None,
            candidate_class=decision.candidate_class,
            score=decision.score,
            score_kind=decision.score_kind,
            second_best_class=decision.second_best_class,
            second_best_score=decision.second_best_score,
            margin=decision.margin,
            class_scores=dict(decision.class_scores),
            metadata=self._metadata(decision, decision.candidate_class),
        )
        self._last_output = rejected
        return rejected

    def update(self, decision: Stage2Decision) -> Stage2Decision:
        expected_prefix = "f" if decision.module == "face" else "k"
        score_map = {
            label: score
            for label, score in self._decision_scores(decision).items()
            if label.startswith(expected_prefix)
        }
        if (
            decision.candidate_class is None
            or not decision.candidate_class.startswith(expected_prefix)
            or not score_map
        ):
            return self._hold_or_reject(decision)

        self._history.append(decision)
        history_scores = [self._decision_scores(item) for item in self._history]
        labels = sorted({label for scores in history_scores for label in scores})
        mean_scores = {
            label: sum(scores.get(label, 0.0) for scores in history_scores)
            / len(history_scores)
            for label in labels
            if label.startswith(expected_prefix)
        }
        if not mean_scores:
            return self._hold_or_reject(decision)

        if self.method == "mean":
            ranked = sorted(
                mean_scores,
                key=lambda label: mean_scores[label],
                reverse=True,
            )
        else:
            votes = Counter(
                item.candidate_class
                for item in self._history
                if item.candidate_class in mean_scores
            )
            latest_position = {
                item.candidate_class: index
                for index, item in enumerate(self._history)
                if item.candidate_class in mean_scores
            }
            ranked = sorted(
                votes,
                key=lambda label: (
                    votes[label],
                    mean_scores[label],
                    latest_position[label],
                ),
                reverse=True,
            )

        proposed = ranked[0]
        proposed_score = mean_scores[proposed]
        competing = sorted(
            (label for label in mean_scores if label != proposed),
            key=lambda label: mean_scores[label],
            reverse=True,
        )
        proposed_second = competing[0] if competing else None
        proposed_second_score = (
            mean_scores[proposed_second] if proposed_second is not None else None
        )
        proposed_margin = (
            proposed_score - proposed_second_score
            if proposed_second_score is not None
            else None
        )
        if proposed_score < self.threshold:
            proposed_status = "unknown"
        elif proposed_margin is not None and proposed_margin < self.min_margin:
            proposed_status = "ambiguous"
        else:
            proposed_status = "matched"

        if proposed_status == "matched":
            self._uncertain_count = 0
            if self._stable_class is None:
                if proposed == self._pending_class:
                    self._pending_count += 1
                else:
                    self._pending_class = proposed
                    self._pending_count = 1
                if self._pending_count >= self.switch_confirmations:
                    self._stable_class = proposed
                    self._pending_class = None
                    self._pending_count = 0
            elif proposed == self._stable_class:
                self._pending_class = None
                self._pending_count = 0
            elif proposed == self._pending_class:
                self._pending_count += 1
                if self._pending_count >= self.switch_confirmations:
                    self._stable_class = proposed
                    self._pending_class = None
                    self._pending_count = 0
            else:
                self._pending_class = proposed
                self._pending_count = 1
        else:
            self._uncertain_count += 1
            self._pending_class = None
            self._pending_count = 0
            if self._uncertain_count >= self.window_size:
                self._stable_class = None

        if self._stable_class is None:
            output = Stage2Decision(
                module=decision.module,
                status=(
                    "confirming" if proposed_status == "matched" else proposed_status
                ),
                predicted_class=None,
                candidate_class=proposed,
                score=proposed_score,
                score_kind=decision.score_kind,
                second_best_class=proposed_second,
                second_best_score=proposed_second_score,
                margin=proposed_margin,
                class_scores=mean_scores,
                metadata=self._metadata(decision, proposed),
            )
            self._last_output = output
            return output

        stable_score = mean_scores.get(self._stable_class, 0.0)
        stable_competitors = sorted(
            (label for label in mean_scores if label != self._stable_class),
            key=lambda label: mean_scores[label],
            reverse=True,
        )
        stable_second = stable_competitors[0] if stable_competitors else None
        stable_second_score = (
            mean_scores[stable_second] if stable_second is not None else None
        )
        output = Stage2Decision(
            module=decision.module,
            status="matched",
            predicted_class=self._stable_class,
            candidate_class=self._stable_class,
            score=stable_score,
            score_kind=decision.score_kind,
            second_best_class=stable_second,
            second_best_score=stable_second_score,
            margin=(
                stable_score - stable_second_score
                if stable_second_score is not None
                else None
            ),
            class_scores=mean_scores,
            metadata=self._metadata(decision, proposed),
        )
        self._last_output = output
        return output


@dataclass
class _Track:
    track_id: int
    major_class: MajorClass
    bbox_xyxy: tuple[int, int, int, int]
    last_seen_frame: int
    decision: Stage2Decision | None = None
    result_frame: int | None = None
    result_detection_id: str | None = None
    result_detection: Stage1Detection | None = None
    result_crop_path: str | None = None
    inference_ms: float = 0.0
    revision: int = 0
    emitted_revision: int = 0
    future: Future[tuple[Stage2Decision, float]] | None = None
    job_token: int = 0
    smoother: TemporalDecisionSmoother | None = None
    velocity_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    hits: int = 1


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
    right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_geometry(
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = float(max(1, x2 - x1))
    height = float(max(1, y2 - y1))
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height


class AsyncStage2Scheduler:
    """Keep rendering current frames while face and knife models run off-thread.

    Detections are associated only by their public major class and geometry. Raw
    YOLO subclasses are intentionally unavailable to this scheduler.
    """

    def __init__(
        self,
        refresh_frames: int = 10,
        stable_refresh_frames: int | None = None,
        iou_threshold: float = 0.55,
        asynchronous: bool = True,
        workers_per_class: int = 2,
        smoothing_method: SmoothingMethod = "none",
        smoothing_window: int = 3,
        switch_confirmations: int = 2,
        face_threshold: float = 0.40,
        face_min_margin: float = 0.03,
        knife_threshold: float = 0.50,
        knife_min_margin: float = 0.15,
    ) -> None:
        if refresh_frames <= 0:
            raise ValueError("refresh_frames must be positive")
        if stable_refresh_frames is None:
            stable_refresh_frames = refresh_frames
        if stable_refresh_frames < refresh_frames:
            raise ValueError(
                "stable_refresh_frames must be at least refresh_frames"
            )
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if workers_per_class <= 0:
            raise ValueError("workers_per_class must be positive")
        if smoothing_method not in {"none", "mean", "vote"}:
            raise ValueError("smoothing_method must be none, mean, or vote")
        if smoothing_window <= 0 or switch_confirmations <= 0:
            raise ValueError("Smoothing window and confirmations must be positive")
        self.refresh_frames = int(refresh_frames)
        self.stable_refresh_frames = int(stable_refresh_frames)
        self.iou_threshold = float(iou_threshold)
        self.asynchronous = bool(asynchronous)
        self.workers_per_class = int(workers_per_class)
        self.smoothing_method = smoothing_method
        self.smoothing_window = int(smoothing_window)
        self.switch_confirmations = int(switch_confirmations)
        self._thresholds = {
            "f": (float(face_threshold), float(face_min_margin)),
            "k": (float(knife_threshold), float(knife_min_margin)),
        }
        self.max_idle_frames = max(15, self.refresh_frames * 2)
        self.stale_job_gap_frames = max(3, min(self.refresh_frames, 5))
        self._lock = threading.RLock()
        self._executors = (
            {
                "f": ThreadPoolExecutor(
                    max_workers=self.workers_per_class,
                    thread_name_prefix="face-stage2",
                ),
                "k": ThreadPoolExecutor(
                    max_workers=self.workers_per_class,
                    thread_name_prefix="knife-stage2",
                ),
            }
            if self.asynchronous
            else {}
        )
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._active_frame: int | None = None
        self._used_track_ids: set[int] = set()
        self._closed = False

    @staticmethod
    def _processing_decision(major_class: MajorClass) -> Stage2Decision:
        module = "face" if major_class == "f" else "knife"
        return Stage2Decision(module=module, status="processing")

    @staticmethod
    def _run_job(
        recognize: Recognize,
        detection: Stage1Detection,
    ) -> tuple[Stage2Decision, float]:
        started = time.perf_counter()
        decision = recognize(detection)
        return decision, (time.perf_counter() - started) * 1000.0

    def _start_frame(self, frame_index: int) -> None:
        if self._active_frame == frame_index:
            return
        self._active_frame = frame_index
        self._used_track_ids.clear()
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame > self.max_idle_frames
        ]
        for track_id in expired:
            track = self._tracks.pop(track_id)
            if track.future is not None:
                track.future.cancel()

    @staticmethod
    def _predicted_bbox(
        track: _Track,
        frame_index: int,
    ) -> tuple[int, int, int, int]:
        gap = max(0, frame_index - track.last_seen_frame)
        predicted = tuple(
            int(round(value + velocity * gap))
            for value, velocity in zip(track.bbox_xyxy, track.velocity_xyxy)
        )
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return track.bbox_xyxy
        return predicted  # type: ignore[return-value]

    def _association_score(
        self,
        track: _Track,
        detection: Stage1Detection,
    ) -> float | None:
        predicted_bbox = self._predicted_bbox(track, detection.frame_index)
        predicted_overlap = bbox_iou(predicted_bbox, detection.bbox_xyxy)
        previous_overlap = bbox_iou(track.bbox_xyxy, detection.bbox_xyxy)
        current_cx, current_cy, current_w, current_h = _box_geometry(
            detection.bbox_xyxy
        )
        predicted_cx, predicted_cy, predicted_w, predicted_h = _box_geometry(
            predicted_bbox
        )
        width_ratio = min(current_w, predicted_w) / max(current_w, predicted_w)
        height_ratio = min(current_h, predicted_h) / max(current_h, predicted_h)
        if width_ratio < 0.5 or height_ratio < 0.5:
            return None

        center_distance = math.hypot(
            current_cx - predicted_cx,
            current_cy - predicted_cy,
        )
        box_diagonal = max(
            math.hypot(current_w, current_h),
            math.hypot(predicted_w, predicted_h),
        )
        frame_gap = max(1, detection.frame_index - track.last_seen_frame)
        distance_limit = max(24.0, box_diagonal * 0.45) * min(
            2.0, math.sqrt(frame_gap)
        )
        if (
            max(predicted_overlap, previous_overlap) < self.iou_threshold
            and center_distance > distance_limit
        ):
            return None

        normalized_distance = center_distance / max(box_diagonal, 1.0)
        return (
            predicted_overlap * 3.0
            + previous_overlap
            + max(0.0, 1.0 - normalized_distance)
            + 0.25 * (width_ratio + height_ratio)
        )

    def _new_track(self, detection: Stage1Detection) -> _Track:
        threshold, min_margin = self._thresholds[detection.major_class]
        track = _Track(
            track_id=self._next_track_id,
            major_class=detection.major_class,
            bbox_xyxy=detection.bbox_xyxy,
            last_seen_frame=detection.frame_index,
            smoother=(
                TemporalDecisionSmoother(
                    method=self.smoothing_method,
                    window_size=self.smoothing_window,
                    switch_confirmations=self.switch_confirmations,
                    threshold=threshold,
                    min_margin=min_margin,
                )
                if self.smoothing_method != "none"
                else None
            ),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _update_track(self, track: _Track, detection: Stage1Detection) -> None:
        frame_gap = max(1, detection.frame_index - track.last_seen_frame)
        if frame_gap > self.stale_job_gap_frames and track.future is not None:
            track.job_token += 1
            track.future.cancel()
            track.future = None

        observed_velocity = tuple(
            (current - previous) / frame_gap
            for current, previous in zip(detection.bbox_xyxy, track.bbox_xyxy)
        )
        if track.hits <= 1:
            track.velocity_xyxy = observed_velocity  # type: ignore[assignment]
        else:
            track.velocity_xyxy = tuple(
                0.70 * observed + 0.30 * previous
                for observed, previous in zip(
                    observed_velocity,
                    track.velocity_xyxy,
                )
            )  # type: ignore[assignment]
        track.bbox_xyxy = detection.bbox_xyxy
        track.last_seen_frame = detection.frame_index
        track.hits += 1

    def _associate_tracks(
        self,
        detections: Sequence[Stage1Detection],
    ) -> list[_Track]:
        candidates = [
            track
            for track in self._tracks.values()
            if track.track_id not in self._used_track_ids
        ]
        pairs: list[tuple[float, int, _Track]] = []
        for detection_index, detection in enumerate(detections):
            for track in candidates:
                if track.major_class != detection.major_class:
                    continue
                score = self._association_score(track, detection)
                if score is not None:
                    pairs.append((score, detection_index, track))

        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        matches: dict[int, _Track] = {}
        for _, detection_index, track in sorted(
            pairs,
            key=lambda item: item[0],
            reverse=True,
        ):
            if (
                detection_index in assigned_detections
                or track.track_id in assigned_tracks
            ):
                continue
            assigned_detections.add(detection_index)
            assigned_tracks.add(track.track_id)
            matches[detection_index] = track

        resolved: list[_Track] = []
        for detection_index, detection in enumerate(detections):
            track = matches.get(detection_index)
            if track is None:
                track = self._new_track(detection)
            else:
                self._update_track(track, detection)
            self._used_track_ids.add(track.track_id)
            resolved.append(track)
        return resolved

    def _submit(
        self,
        track: _Track,
        detection: Stage1Detection,
        recognize: Recognize,
        crop_path: str | None,
    ) -> None:
        if self._closed or track.future is not None:
            return
        track.job_token += 1
        token = track.job_token
        source_id = detection.detection_id
        source_frame = detection.frame_index
        future = self._executors[detection.major_class].submit(
            self._run_job,
            recognize,
            detection,
        )
        track.future = future

        def completed(done: Future[tuple[Stage2Decision, float]]) -> None:
            try:
                decision, inference_ms = done.result()
            except CancelledError:
                return
            except Exception as exc:
                module = "face" if detection.major_class == "f" else "knife"
                decision = Stage2Decision(
                    module=module,
                    status="error",
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
                inference_ms = 0.0
            with self._lock:
                current = self._tracks.get(track.track_id)
                if current is None or current.job_token != token:
                    return
                current.future = None
                current.decision = (
                    current.smoother.update(decision)
                    if current.smoother is not None
                    else decision
                )
                current.result_frame = source_frame
                current.result_detection_id = source_id
                current.result_detection = detection
                current.result_crop_path = crop_path
                current.inference_ms = inference_ms
                current.revision += 1

        future.add_done_callback(completed)

    @staticmethod
    def _resolve_direct(
        detection: Stage1Detection,
        recognize: Recognize,
        prepare_job: PrepareJob | None,
    ) -> Stage2Resolution:
        crop_path = (
            prepare_job()
            if detection.major_class != "b0" and prepare_job is not None
            else None
        )
        started = time.perf_counter()
        decision = recognize(detection)
        return Stage2Resolution(
            decision=decision,
            mode="direct",
            inference_ms=(time.perf_counter() - started) * 1000.0,
            source_detection_id=detection.detection_id,
            source_crop_path=crop_path,
        )

    def _resolve_track(
        self,
        track: _Track,
        detection: Stage1Detection,
        recognize: Recognize,
        prepare_job: PrepareJob | None,
    ) -> Stage2Resolution:
        result_age = (
            detection.frame_index - track.result_frame
            if track.result_frame is not None
            else None
        )
        refresh_interval = (
            self.stable_refresh_frames
            if track.decision is not None
            and track.decision.status == "matched"
            and track.decision.predicted_class is not None
            else self.refresh_frames
        )
        needs_refresh = (
            track.decision is None
            or result_age is None
            or result_age >= refresh_interval
        )
        if needs_refresh and track.future is None:
            crop_path = prepare_job() if prepare_job is not None else None
            self._submit(track, detection, recognize, crop_path)

        if track.decision is None:
            return Stage2Resolution(
                decision=self._processing_decision(detection.major_class),
                mode="processing",
                track_id=track.track_id,
                source_detection_id=detection.detection_id,
            )

        is_fresh = track.revision > track.emitted_revision
        if is_fresh:
            track.emitted_revision = track.revision
        return Stage2Resolution(
            decision=track.decision,
            mode="fresh" if is_fresh else "cached",
            inference_ms=track.inference_ms if is_fresh else 0.0,
            track_id=track.track_id,
            cache_age_frames=result_age,
            source_detection_id=track.result_detection_id,
            source_crop_path=track.result_crop_path,
        )

    def resolve_frame(
        self,
        detections: Sequence[Stage1Detection],
        recognize: Recognize,
        prepare_jobs: Sequence[PrepareJob | None] | None = None,
    ) -> list[Stage2Resolution]:
        """Associate all detections together before scheduling stage-2 work."""

        if not detections:
            return []
        frame_indexes = {item.frame_index for item in detections}
        if len(frame_indexes) != 1:
            raise ValueError("resolve_frame detections must belong to one frame")
        if prepare_jobs is None:
            jobs: list[PrepareJob | None] = [None] * len(detections)
        else:
            if len(prepare_jobs) != len(detections):
                raise ValueError("prepare_jobs must match the detection count")
            jobs = list(prepare_jobs)

        if not self.asynchronous:
            return [
                self._resolve_direct(detection, recognize, prepare_job)
                for detection, prepare_job in zip(detections, jobs)
            ]

        resolutions: list[Stage2Resolution | None] = [None] * len(detections)
        async_indexes = [
            index
            for index, detection in enumerate(detections)
            if detection.major_class != "b0"
        ]
        for index, detection in enumerate(detections):
            if detection.major_class == "b0":
                resolutions[index] = self._resolve_direct(
                    detection,
                    recognize,
                    jobs[index],
                )

        with self._lock:
            if self._closed:
                raise RuntimeError("Stage-2 scheduler is closed")
            self._start_frame(detections[0].frame_index)
            async_detections = [detections[index] for index in async_indexes]
            tracks = self._associate_tracks(async_detections)
            for index, detection, track in zip(
                async_indexes,
                async_detections,
                tracks,
            ):
                resolutions[index] = self._resolve_track(
                    track,
                    detection,
                    recognize,
                    jobs[index],
                )

        if any(resolution is None for resolution in resolutions):
            raise RuntimeError("Stage-2 frame resolution is incomplete")
        return [resolution for resolution in resolutions if resolution is not None]

    def resolve(
        self,
        detection: Stage1Detection,
        recognize: Recognize,
        prepare_job: PrepareJob | None = None,
    ) -> Stage2Resolution:
        return self.resolve_frame(
            [detection],
            recognize,
            [prepare_job],
        )[0]

    def wait_for_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                futures = [
                    track.future
                    for track in self._tracks.values()
                    if track.future is not None
                ]
            if not futures:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait(futures, timeout=remaining)

    def finalize_results(self) -> list[FinalStage2Result]:
        """Collect completed results that arrived after the last frame."""
        if not self.asynchronous:
            return []
        with self._lock:
            results: list[FinalStage2Result] = []
            for track in self._tracks.values():
                if (
                    track.decision is None
                    or track.result_detection is None
                    or track.revision <= track.emitted_revision
                ):
                    continue
                track.emitted_revision = track.revision
                resolution = Stage2Resolution(
                    decision=track.decision,
                    mode="fresh",
                    inference_ms=track.inference_ms,
                    track_id=track.track_id,
                    source_detection_id=track.result_detection_id,
                    source_crop_path=track.result_crop_path,
                )
                results.append(
                    FinalStage2Result(
                        resolution=resolution,
                        detection=track.result_detection,
                    )
                )
            return results

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = [
                track.future
                for track in self._tracks.values()
                if track.future is not None
            ]
            for future in futures:
                future.cancel()
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
