#!/usr/bin/env python3
"""Standalone D455 YOLO preview. No ROS or production runtime."""
from pathlib import Path
import argparse
import threading
import time
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--weights', default=str(ROOT / 'models/yolo/best.pt'))
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='auto', help='auto, cpu, or CUDA index such as 0')
    p.add_argument('--serial', default=None)
    p.add_argument('--cpu-threads', type=int, default=2)
    return p


def load_yolo(args):
    import torch
    from ultralytics import YOLO
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError('conf and iou must be between 0 and 1')
    if args.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(args.weights)
    print(f'YOLO weights={args.weights} device={device}', flush=True)
    return model, device


def detect(model, device, args, image):
    return model.predict(image, conf=args.conf, iou=args.iou, imgsz=640,
                         device=device, verbose=False)[0]


class LatestCamera:
    """Keep only the newest frame, regardless of inference speed."""
    def __init__(self, serial):
        from insight_cup.app.sources import RealSenseColorSource
        self.source = RealSenseColorSource(serial=serial, width=848, height=480, fps=15)
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.thread = None

    def __enter__(self):
        print(self.source.start(), flush=True)
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        return self

    def _capture(self):
        try:
            while not self.stop.is_set():
                frame = self.source.read(timeout_ms=1000)
                with self.condition:
                    self.latest = (frame.index, frame.image, time.perf_counter())
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def read(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error is not None or (self.latest is not None and self.latest[0] != previous),
                timeout=3)
            if self.error is not None:
                raise self.error
            if not ready:
                raise RuntimeError('No new D455 frame within 3 seconds')
            return self.latest

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=2)
        self.source.stop()


def overlay(image, text):
    import cv2
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parser('D455: YOLO raw boxes only').parse_args()
    import cv2
    model, device = load_yolo(args)
    cv2.namedWindow('YOLO test', cv2.WINDOW_NORMAL)
    previous = -1
    try:
        with LatestCamera(args.serial) as camera:
            while True:
                previous, image, captured = camera.read(previous)
                started = time.perf_counter()
                result = detect(model, device, args, image)
                elapsed = (time.perf_counter() - started) * 1000
                view = result.plot()
                overlay(view, f'YOLO {elapsed:.0f} ms | boxes {len(result.boxes)} | frame age {(time.perf_counter()-captured)*1000:.0f} ms | Q: quit')
                cv2.imshow('YOLO test', view)
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    break
                if cv2.getWindowProperty('YOLO test', cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
