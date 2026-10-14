#!/usr/bin/env python3
"""Standalone D455 YOLO + stage-2 preview without ROI or pixel-area gates."""
import time
from yolo_test import ROOT, LatestCamera, parser, load_yolo, detect, overlay


# Explicitly use the native production pipeline's ONNX assets and ICG gallery.
FACE_MODEL_ROOT = ROOT / 'models/face'
FACE_MODEL_NAME = 'face_only_r50'
FACE_GALLERY = ROOT / 'data/face_gallery/cpp/trainv5_gallery_r50.icg'
KNIFE_MODEL_DIR = ROOT / 'models/knife/pplcnetv2_base_knife10'


def load_production_face_gallery():
    import struct
    import numpy as np
    from insight_cup.face.gallery import FaceGallery
    with FACE_GALLERY.open('rb') as stream:
        def read(size):
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('Truncated production face gallery')
            return data
        if read(4) != b'ICG1':
            raise ValueError('Unsupported production face gallery format')
        rows, columns = struct.unpack('<II', read(8))
        if not 0 < rows <= 1000000 or columns != 512:
            raise ValueError('Invalid production face gallery dimensions')
        labels, embeddings = [], []
        for _ in range(rows):
            length = struct.unpack('<H', read(2))[0]
            if not 0 < length <= 1024:
                raise ValueError('Invalid production face gallery label')
            labels.append(read(length).decode('utf-8'))
            embeddings.append(np.frombuffer(read(columns * 4), dtype='<f4').copy())
    return FaceGallery(np.asarray(labels), np.asarray(embeddings))


def main():
    p = parser('D455: test stage-2 knife classification and face recognition')
    p.set_defaults(cpu_threads=1)
    p.add_argument('--target', choices=['knife', 'face', 'both'], default='both')
    p.add_argument('--provider', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--knife-threshold', type=float, default=0.50)
    p.add_argument('--knife-margin', type=float, default=0.15)
    p.add_argument('--face-threshold', type=float, default=0.40)
    p.add_argument('--face-margin', type=float, default=0.03)
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error('--cpu-threads must be positive')
    import cv2
    import numpy as np
    from insight_cup.common.crops import square_context_crop
    knife = face = None
    if args.target in ('knife', 'both'):
        from insight_cup.knife.engine import PaddleClasONNXEngine
        from insight_cup.knife.service import KnifeRecognitionService
        knife = KnifeRecognitionService(PaddleClasONNXEngine(model_dir=KNIFE_MODEL_DIR, provider=args.provider, cpu_threads=args.cpu_threads),
                                        threshold=args.knife_threshold, min_margin=args.knife_margin)
    if args.target in ('face', 'both'):
        from insight_cup.face.engine import InsightFaceEmbeddingEngine
        from insight_cup.face.service import FaceRecognitionService
        face = FaceRecognitionService(
            InsightFaceEmbeddingEngine(model_root=FACE_MODEL_ROOT, model_name=FACE_MODEL_NAME,
                                       provider=args.provider, det_size=(320, 320), det_thresh=0.18,
                                       upsample_min_side=320,
                                       cpu_threads=args.cpu_threads),
            load_production_face_gallery(),
            threshold=args.face_threshold, min_margin=args.face_margin, rotation_retry_degrees=0.0)
    model, device = load_yolo(args)
    cv2.namedWindow('Second test', cv2.WINDOW_NORMAL)
    previous = -1
    try:
        with LatestCamera(args.serial) as camera:
            while True:
                previous, image, captured = camera.read(previous)
                started = time.perf_counter()
                result = detect(model, device, args, image)
                yolo_ms = (time.perf_counter()-started)*1000
                boxes = []
                for box in result.boxes:
                    raw = result.names[int(box.cls.item())]
                    kind = 'knife' if raw.startswith('k') else 'face' if raw.startswith('f') else None
                    if kind is not None and args.target in (kind, 'both'):
                        boxes.append((tuple(int(v) for v in box.xyxy[0].tolist()), kind, raw))
                # The left preview retains every YOLO box regardless of stage-2 status.
                yolo_view = result.plot()
                stage_started = time.perf_counter()
                view = image.copy()
                for bbox, kind, raw in boxes:
                    crop = square_context_crop(image, bbox, 1.15 if kind == 'knife' else 1.30)
                    service = knife if kind == 'knife' else face
                    decision = service.recognize(crop)
                    label = decision.predicted_class or decision.status
                    color = (0, 220, 0) if decision.status == 'matched' else (0, 180, 255)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(view, label, (max(0, x1), max(48, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                stage_ms = (time.perf_counter()-stage_started)*1000
                overlay(yolo_view, f'YOLO: {len(result.boxes)} boxes | {yolo_ms:.0f} ms')
                overlay(view, f'Stage2: {len(boxes)} crops | {stage_ms:.0f} ms | age {(time.perf_counter()-captured)*1000:.0f} ms | Q: quit')
                cv2.imshow('Second test', np.hstack((yolo_view, view)))
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty('Second test', cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
