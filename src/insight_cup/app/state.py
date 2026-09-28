"""Thread-safe state shared by the runtime worker and debug HTTP server."""

from __future__ import annotations

import threading
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RuntimeState:
    def __init__(self, max_events: int = 200, retain_events: bool = True) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._retain_events = bool(retain_events)
        self._frame_jpeg: bytes | None = None
        self._frame_version = 0
        self._stop_requested = False
        self._paused = False
        self._version = 0
        self._state: dict[str, Any] = {
            "schema": "recognition_runtime_state.v1",
            "status": "starting",
            "source": None,
            "source_info": {},
            "session_id": None,
            "session_dir": None,
            "started_at": _now(),
            "ended_at": None,
            "error": None,
            "frame_index": None,
            "processed_frames": 0,
            "fps": 0.0,
            "last_timing_ms": {"yolo": 0.0, "stage2": 0.0, "total": 0.0},
            "detections_in_frame": 0,
            "major_counts": {"b0": 0, "f": 0, "k": 0},
            "status_counts": {},
            "modules": {
                "yolo": {"status": "waiting", "detail": ""},
                "face": {"status": "waiting", "detail": ""},
                "knife": {"status": "waiting", "detail": ""},
            },
            "has_frame": False,
        }
        self._major_counts: Counter[str] = Counter()
        self._status_counts: Counter[str] = Counter()

    def _changed(self) -> None:
        self._version += 1
        self._condition.notify_all()

    def configure_session(self, source: str, session_id: str, session_dir: Path) -> None:
        with self._condition:
            self._state.update(
                {
                    "source": source,
                    "session_id": session_id,
                    "session_dir": str(session_dir),
                }
            )
            self._changed()

    def set_source_info(self, **values: Any) -> None:
        with self._condition:
            self._state["source_info"].update(values)
            self._changed()

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._condition:
            self._state["status"] = status
            self._state["error"] = error
            if status in {"completed", "stopped", "error"}:
                self._state["ended_at"] = _now()
            self._changed()

    def set_module(self, name: str, status: str, detail: str = "") -> None:
        with self._condition:
            self._state["modules"][name] = {"status": status, "detail": detail}
            self._changed()

    def record_event(self, event: dict[str, Any]) -> None:
        with self._condition:
            status = event["stage2"]["status"]
            self._status_counts[status] += 1
            self._state["status_counts"] = dict(self._status_counts)
            if self._retain_events:
                self._events.append(deepcopy(event))
            self._changed()

    def record_detection(self, major_class: str) -> None:
        if major_class not in {"b0", "f", "k"}:
            raise ValueError(f"Unsupported major class: {major_class}")
        with self._condition:
            self._major_counts[major_class] += 1
            self._state["major_counts"] = {
                name: self._major_counts[name] for name in ("b0", "f", "k")
            }
            self._changed()

    def publish_frame(
        self,
        jpeg: bytes | None,
        frame_index: int,
        processed_frames: int,
        fps: float,
        timing_ms: dict[str, float],
        detection_count: int,
    ) -> None:
        with self._condition:
            if jpeg is not None:
                self._frame_jpeg = jpeg
                self._frame_version += 1
            self._state.update(
                {
                    "frame_index": frame_index,
                    "processed_frames": processed_frames,
                    "fps": round(float(fps), 2),
                    "last_timing_ms": {
                        key: round(float(value), 2) for key, value in timing_ms.items()
                    },
                    "detections_in_frame": detection_count,
                    "has_frame": self._frame_jpeg is not None,
                }
            )
            self._changed()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            payload = deepcopy(self._state)
            payload["version"] = self._version
            payload["paused"] = self._paused
            payload["stop_requested"] = self._stop_requested
            return payload

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._condition:
            count = max(1, min(int(limit), self._events.maxlen or 200))
            return [deepcopy(item) for item in list(self._events)[-count:][::-1]]

    def latest_frame(self) -> bytes | None:
        with self._condition:
            return self._frame_jpeg

    def wait_for_frame(
        self,
        after_version: int,
        timeout: float = 1.0,
    ) -> tuple[int, bytes | None, str]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_version > after_version
                or self._state["status"] in {"completed", "stopped", "error"},
                timeout=max(0.0, timeout),
            )
            return self._frame_version, self._frame_jpeg, self._state["status"]

    def control(self, action: str) -> tuple[bool, str]:
        normalized = action.strip().lower()
        with self._condition:
            status = self._state["status"]
            if normalized == "pause":
                if status != "running":
                    return False, f"cannot pause while status is {status}"
                self._paused = True
                self._state["status"] = "paused"
            elif normalized == "resume":
                if status != "paused":
                    return False, f"cannot resume while status is {status}"
                self._paused = False
                self._state["status"] = "running"
            elif normalized == "stop":
                if status in {"completed", "stopped", "error"}:
                    return False, f"runtime is already {status}"
                self._stop_requested = True
                self._paused = False
                self._state["status"] = "stopping"
            else:
                return False, f"unsupported action: {action}"
            self._changed()
            return True, normalized

    def wait_until_runnable(self) -> bool:
        with self._condition:
            while self._paused and not self._stop_requested:
                self._condition.wait(timeout=0.5)
            return not self._stop_requested

    @property
    def stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested
