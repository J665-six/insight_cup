from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "evaluation"))

from evaluate_face import evaluate
from insight_cup.face.engine import FaceEmbeddingResult
from insight_cup.face.gallery import FaceGallery
from insight_cup.face.pipeline import process_handoff
from insight_cup.face.service import FaceRecognitionService
from insight_cup.vendor.insightface_core.app import FaceAnalysis
from insight_cup.vendor.insightface_core.model_zoo import ArcFaceONNX, SCRFD


class StubEngine:
    metadata = {
        "backend": "stub",
        "loaded_modules": ["detection", "recognition"],
    }

    def extract(self, image: np.ndarray) -> FaceEmbeddingResult | None:
        if int(image.mean()) == 0:
            return None
        return FaceEmbeddingResult(
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            detection_score=0.9,
            bbox_xyxy=(1.0, 2.0, 8.0, 9.0),
            crop_shape=image.shape[:2],
        )


class NoFaceThenMatchEngine:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image: np.ndarray) -> FaceEmbeddingResult | None:
        self.calls += 1
        if self.calls == 1:
            return None
        return FaceEmbeddingResult(
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            detection_score=0.8,
            bbox_xyxy=(1.0, 1.0, 8.0, 8.0),
            crop_shape=image.shape[:2],
        )


class TiltedWeakThenMatchEngine:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image: np.ndarray) -> FaceEmbeddingResult | None:
        self.calls += 1
        embedding = (
            np.array([0.7, 0.7], dtype=np.float32)
            if self.calls == 1
            else np.array([1.0, 0.0], dtype=np.float32)
        )
        return FaceEmbeddingResult(
            embedding=embedding,
            detection_score=0.8,
            bbox_xyxy=(1.0, 1.0, 8.0, 8.0),
            crop_shape=image.shape[:2],
            roll_degrees=18.0 if self.calls == 1 else 0.0,
        )


class FaceGalleryTests(unittest.TestCase):
    def test_focused_upstream_runtime_exports_only_required_models(self) -> None:
        self.assertIsNotNone(FaceAnalysis)
        self.assertIsNotNone(ArcFaceONNX)
        self.assertIsNotNone(SCRFD)

    def test_match_accepts_and_rejects_by_threshold(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        accepted = gallery.match(np.array([0.99, 0.1]), threshold=0.8)
        self.assertEqual(accepted.status, "matched")
        self.assertEqual(accepted.predicted_class, "f1")
        self.assertEqual(accepted.second_best_class, "f2")

        rejected = gallery.match(np.array([1.0, 1.0]), threshold=0.8)
        self.assertEqual(rejected.status, "unknown")
        self.assertIsNone(rejected.predicted_class)

    def test_ambiguous_match_is_not_assigned(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.99, 0.02]], dtype=np.float32),
        )
        result = gallery.match(np.array([1.0, 0.0]), threshold=0.5, min_margin=0.03)
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.predicted_class)

    def test_pipeline_processes_only_macro_f_records(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temp_value:
            temp_dir = Path(temp_value)
            crop_dir = temp_dir / "crops" / "f"
            crop_dir.mkdir(parents=True)
            cv2.imwrite(str(crop_dir / "face.jpg"), np.full((20, 20, 3), 255, np.uint8))
            handoff = {
                "schema": "stage1_yolo_handoff.v1",
                "base_dir": str(temp_dir),
                "frames": [
                    {
                        "frame_index": 7,
                        "detections": [
                            {
                                "id": "f-det",
                                "major_class": "f",
                                "bbox_xyxy": [1, 2, 10, 12],
                                "crop_path": "crops/f/face.jpg",
                            },
                            {
                                "id": "k-det",
                                "major_class": "k",
                                "bbox_xyxy": [2, 3, 11, 13],
                                "crop_path": "crops/k/knife.jpg",
                            },
                        ],
                    }
                ],
            }
            handoff_path = temp_dir / "stage1_handoff.json"
            handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
            output_path = temp_dir / "stage2.json"
            result = process_handoff(
                handoff_path,
                gallery,
                StubEngine(),
                output_path,
                threshold=0.8,
            )

            self.assertEqual(result["summary"]["processed_f_detections"], 1)
            self.assertEqual(result["detections"][0]["predicted_class"], "f1")
            self.assertNotIn("embedding", result["detections"][0])
            self.assertTrue(output_path.is_file())

    def test_single_crop_service_does_not_expose_embedding(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        service = FaceRecognitionService(StubEngine(), gallery, threshold=0.8)
        result = service.recognize(np.full((20, 20, 3), 255, np.uint8))
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.predicted_class, "f1")
        self.assertFalse(hasattr(result, "embedding"))

    def test_no_face_retries_both_rotation_directions(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        engine = NoFaceThenMatchEngine()
        result = FaceRecognitionService(
            engine,  # type: ignore[arg-type]
            gallery,
            threshold=0.8,
            rotation_retry_degrees=25,
        ).recognize(np.full((20, 20, 3), 255, np.uint8))
        self.assertEqual(engine.calls, 3)
        self.assertEqual(result.predicted_class, "f1")
        self.assertEqual(result.attempted_rotation_degrees, (0.0, -25.0, 25.0))

    def test_tilted_weak_face_retries_only_measured_roll(self) -> None:
        gallery = FaceGallery(
            labels=np.array(["f1", "f2"]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        engine = TiltedWeakThenMatchEngine()
        result = FaceRecognitionService(
            engine,  # type: ignore[arg-type]
            gallery,
            threshold=0.8,
            rotation_retry_degrees=25,
        ).recognize(np.full((20, 20, 3), 255, np.uint8))
        self.assertEqual(engine.calls, 2)
        self.assertEqual(result.predicted_class, "f1")
        self.assertEqual(result.attempted_rotation_degrees, (0.0, 18.0))
        self.assertEqual(result.selected_rotation_degrees, 18.0)

    def test_offline_evaluation_joins_by_detection_id(self) -> None:
        predictions = {
            "detections": [
                {
                    "id": "one",
                    "status": "matched",
                    "candidate_class": "f2",
                    "predicted_class": "f2",
                },
                {
                    "id": "two",
                    "status": "no_face",
                    "candidate_class": None,
                    "predicted_class": None,
                },
            ]
        }
        debug = {
            "frames": [
                {
                    "detections": [
                        {"id": "one", "major_class": "f", "raw_yolo_class": "f2"},
                        {"id": "two", "major_class": "f", "raw_yolo_class": "f5"},
                    ]
                }
            ]
        }
        result = evaluate(predictions, debug)
        self.assertEqual(result["assigned_precision"], 1.0)
        self.assertEqual(result["face_detection_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
