"""Camera source adapters used by the recognition runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REALSENSE_SOURCE_NAMES = frozenset(
    {"realsense", "d455", "depth-camera", "depth_camera"}
)
VIDEO_EXTENSIONS = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
)


def is_realsense_source(value: str) -> bool:
    return value.strip().lower() in REALSENSE_SOURCE_NAMES


def is_video_source(value: str) -> bool:
    """Return whether a source should use the camera-style async pipeline."""
    if is_realsense_source(value) or value.strip().isdigit():
        return False
    path = Path(value).expanduser()
    if path.is_dir():
        return False
    return path.suffix.lower() in VIDEO_EXTENSIONS


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    index: int
    timestamp_ms: float | None


class RealSenseColorSource:
    """Read the BGR color stream from an Intel RealSense depth camera."""

    def __init__(
        self,
        serial: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.requested_serial = serial.strip() if serial else None
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._pipeline: Any | None = None
        self._frame_index = 0
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._metadata:
            raise RuntimeError("RealSense source has not been started")
        return dict(self._metadata)

    def start(self) -> dict[str, Any]:
        if self._pipeline is not None:
            raise RuntimeError("RealSense source is already running")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for --source realsense. "
                "Install requirements/runtime.txt first."
            ) from exc

        context = rs.context()
        devices = list(context.query_devices())
        available = [
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
            }
            for device in devices
        ]
        if not available:
            raise RuntimeError("No Intel RealSense camera was detected")
        if self.requested_serial and not any(
            item["serial"] == self.requested_serial for item in available
        ):
            serials = ", ".join(item["serial"] for item in available)
            raise RuntimeError(
                f"RealSense serial {self.requested_serial} was not found; "
                f"available: {serials}"
            )

        pipeline = rs.pipeline(context)
        config = rs.config()
        if self.requested_serial:
            config.enable_device(self.requested_serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        try:
            profile = pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(
                "Could not start the RealSense color stream at "
                f"{self.width}x{self.height}@{self.fps}: {exc}"
            ) from exc

        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        firmware = device.get_info(rs.camera_info.firmware_version)
        self._pipeline = pipeline
        self._metadata = {
            "type": "realsense",
            "model": name,
            "serial": serial,
            "firmware": firmware,
            "stream": "color",
            "format": "bgr8",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
        return self.metadata

    def read(self, timeout_ms: int = 5000) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSense source has not been started")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            raise RuntimeError(f"RealSense frame timeout: {exc}") from exc
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense frameset did not contain a color frame")
        image = np.asanyarray(color.get_data()).copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Unexpected RealSense color shape: {image.shape}")
        timestamp = float(color.get_timestamp())
        result = CameraFrame(
            image=image,
            index=self._frame_index,
            timestamp_ms=timestamp if np.isfinite(timestamp) else None,
        )
        self._frame_index += 1
        return result

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def __enter__(self) -> "RealSenseColorSource":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
