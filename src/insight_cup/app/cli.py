"""Command-line entry point for the unified recognition runtime."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from .config import (
    DEFAULT_FACE_GALLERY,
    DEFAULT_FACE_MODEL_ROOT,
    DEFAULT_KNIFE_MODEL_DIR,
    DEFAULT_KNIFE_GALLERY,
    DEFAULT_KNIFE_RETRIEVAL_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_YOLO_WEIGHTS,
    RuntimeConfig,
)
from .logging_utils import EventJournal, configure_logging
from .runtime import RecognitionRuntime, create_session_directory
from .state import RuntimeState
from .web import create_debug_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO -> face/knife recognition with logs and a debug UI."
    )
    parser.add_argument(
        "--source",
        default="realsense",
        help="realsense (default), OpenCV camera index, video, image, or image directory.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="Ultralytics device, for example auto, cpu, or 0.")
    parser.add_argument("--yolo-cpu-threads", type=int, default=1)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--face-gallery", default=str(DEFAULT_FACE_GALLERY))
    parser.add_argument("--face-model-root", default=str(DEFAULT_FACE_MODEL_ROOT))
    parser.add_argument("--face-model-name", default="face_only_r50")
    parser.add_argument("--face-threshold", type=float, default=0.40)
    parser.add_argument("--face-min-margin", type=float, default=0.03)
    parser.add_argument("--face-det-thresh", type=float, default=0.18)
    parser.add_argument("--face-det-size", type=int, default=320)
    parser.add_argument("--face-upsample-min-side", type=int, default=320)
    parser.add_argument("--face-cpu-threads", type=int, default=1)
    parser.add_argument(
        "--face-rotation-retry-degrees",
        type=float,
        default=0.0,
        help="Retry weak tilted faces after roll correction; use 0 to disable.",
    )
    parser.add_argument("--knife-model-dir", default=str(DEFAULT_KNIFE_MODEL_DIR))
    parser.add_argument(
        "--knife-mode",
        choices=("classification", "retrieval"),
        default="classification",
    )
    parser.add_argument(
        "--knife-retrieval-model",
        default=str(DEFAULT_KNIFE_RETRIEVAL_MODEL),
    )
    parser.add_argument("--knife-gallery", default=str(DEFAULT_KNIFE_GALLERY))
    parser.add_argument("--knife-threshold", type=float, default=0.50)
    parser.add_argument("--knife-min-margin", type=float, default=0.15)
    parser.add_argument("--knife-retrieval-threshold", type=float, default=0.50)
    parser.add_argument("--knife-retrieval-min-margin", type=float, default=0.02)
    parser.add_argument("--knife-cpu-threads", type=int, default=1)
    parser.add_argument(
        "--stage2-refresh-frames",
        type=int,
        default=15,
        help="Refresh an unconfirmed face/knife result after this many frames.",
    )
    parser.add_argument(
        "--stage2-stable-refresh-frames",
        type=int,
        default=30,
        help="Refresh an already confirmed face/knife result after this many frames.",
    )
    parser.add_argument("--stage2-cache-iou", type=float, default=0.55)
    parser.add_argument(
        "--stage2-workers",
        type=int,
        default=2,
        help="Concurrent workers for each face/knife stage-2 queue.",
    )
    parser.add_argument(
        "--stage2-smoothing",
        choices=("none", "mean", "vote"),
        default="vote",
        help="Temporal decision strategy shared by face and knife tracks.",
    )
    parser.add_argument("--stage2-smoothing-window", type=int, default=3)
    parser.add_argument("--stage2-switch-confirmations", type=int, default=2)
    parser.add_argument(
        "--realsense-serial",
        default=None,
        help="Optional Intel RealSense serial number when multiple devices are connected.",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument(
        "--unthrottled-video",
        action="store_true",
        help="Process video files as fast as possible instead of their source FPS.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--ui",
        dest="no_ui",
        action="store_false",
        help="Enable the HTTP debug UI (start.sh defaults to no UI).",
    )
    ui_group.add_argument(
        "--no-ui",
        dest="no_ui",
        action="store_true",
        help="Run to completion without the HTTP UI.",
    )
    parser.set_defaults(no_ui=False)
    parser.add_argument("--exit-on-complete", action="store_true")
    browser_group = parser.add_mutually_exclusive_group()
    browser_group.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="Open the debug UI in the default browser (default).",
    )
    browser_group.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Start the debug UI without opening a browser window.",
    )
    parser.set_defaults(open_browser=True)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--no-save-crops", action="store_true")
    parser.add_argument("--show-raw-label", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--frame-log-every", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolved_source(value: str) -> str:
    source = value.strip()
    if source.isdigit():
        return source
    candidate = Path(source).expanduser()
    return str(candidate.resolve()) if candidate.exists() else source


def _build_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        source=_resolved_source(args.source),
        weights=Path(args.weights).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        yolo_device=args.device,
        yolo_cpu_threads=args.yolo_cpu_threads,
        provider=args.provider,
        face_gallery=Path(args.face_gallery).expanduser().resolve(),
        face_model_root=Path(args.face_model_root).expanduser().resolve(),
        face_model_name=args.face_model_name,
        face_threshold=args.face_threshold,
        face_min_margin=args.face_min_margin,
        face_detection_threshold=args.face_det_thresh,
        face_detection_size=args.face_det_size,
        face_upsample_min_side=args.face_upsample_min_side,
        face_cpu_threads=args.face_cpu_threads,
        face_rotation_retry_degrees=args.face_rotation_retry_degrees,
        knife_model_dir=Path(args.knife_model_dir).expanduser().resolve(),
        knife_mode=args.knife_mode,
        knife_retrieval_model=Path(args.knife_retrieval_model).expanduser().resolve(),
        knife_gallery=Path(args.knife_gallery).expanduser().resolve(),
        knife_threshold=args.knife_threshold,
        knife_min_margin=args.knife_min_margin,
        knife_retrieval_threshold=args.knife_retrieval_threshold,
        knife_retrieval_min_margin=args.knife_retrieval_min_margin,
        knife_cpu_threads=args.knife_cpu_threads,
        stage2_refresh_frames=args.stage2_refresh_frames,
        stage2_stable_refresh_frames=args.stage2_stable_refresh_frames,
        stage2_cache_iou=args.stage2_cache_iou,
        stage2_workers=args.stage2_workers,
        stage2_smoothing=args.stage2_smoothing,
        stage2_smoothing_window=args.stage2_smoothing_window,
        stage2_switch_confirmations=args.stage2_switch_confirmations,
        realsense_serial=args.realsense_serial,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        video_stride=args.vid_stride,
        pace_video=not args.unthrottled_video,
        max_frames=args.max_frames,
        save_video=not args.no_save_video,
        save_crops=not args.no_save_crops,
        show_raw_label=args.show_raw_label,
        publish_preview=not args.no_ui,
        jpeg_quality=args.jpeg_quality,
        preview_width=args.preview_width,
        frame_log_every=args.frame_log_every,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    config = _build_config(args)
    logger = configure_logging(args.verbose)
    session_dir = create_session_directory(config.output_root, args.session_id)
    state = RuntimeState(retain_events=not args.no_ui)
    state.configure_session(config.source, session_dir.name, session_dir)
    journal = EventJournal(session_dir / "events.jsonl", logger)
    runtime = RecognitionRuntime(config, session_dir, state, journal, logger)

    logger.info("SESSION id=%s source=%s", session_dir.name, config.source)
    logger.info("OUTPUT path=%s", session_dir)

    if args.no_ui:
        runtime.run()
        return 1 if state.snapshot()["status"] == "error" else 0

    try:
        server = create_debug_server(args.host, args.port, state)
    except Exception:
        journal.close()
        raise

    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{server.server_address[1]}"
    logger.info("UI url=%s", url)
    worker = runtime.start()

    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    if args.exit_on_complete:
        def stop_server_after_runtime() -> None:
            worker.join()
            server.shutdown()

        threading.Thread(
            target=stop_server_after_runtime,
            name="runtime-completion-watcher",
            daemon=True,
        ).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        logger.info("STOP reason=keyboard_interrupt")
    finally:
        state.control("stop")
        worker.join()
        server.server_close()
    return 1 if state.snapshot()["status"] == "error" else 0
