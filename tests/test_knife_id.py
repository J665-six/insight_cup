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

from insight_cup.common.crops import square_context_crop
from insight_cup.knife.engine import KnifePrediction
from insight_cup.knife.retrieval import KnifeFeatureGallery
from insight_cup.knife.pipeline import process_handoff
from insight_cup.knife.service import DEFAULT_KNIFE_THRESHOLD, KnifeRecognitionService
from insight_cup.vendor.paddleclas_core import (
    NormalizeImage,
    ResizeImage,
    ToCHWImage,
    Topk,
    FlatInnerProductIndex,
    normalize_features,
)


class StubEngine:
    metadata = {
        "backend": "stub",
        "architecture": "PPLCNetV2_base",
        "labels": [f"k{index}" for index in range(1, 11)],
    }

    def __init__(
        self,
        labels=("k2", "k5"),
        scores=(0.9, 0.05),
        score_kind="softmax_confidence",
    ) -> None:
        self.labels = labels
        self.scores = scores
        self.score_kind = score_kind

    def predict(self, image: np.ndarray) -> KnifePrediction:
        return KnifePrediction(
            class_ids=tuple(range(len(self.labels))),
            labels=tuple(self.labels),
            scores=tuple(self.scores),
            crop_shape=image.shape[:2],
            score_kind=self.score_kind,
        )


class KnifeRecognitionTests(unittest.TestCase):
    def test_focused_upstream_runtime_exports_required_operators(self) -> None:
        self.assertIsNotNone(ResizeImage)
        self.assertIsNotNone(NormalizeImage)
        self.assertIsNotNone(ToCHWImage)
        self.assertIsNotNone(Topk)

    def test_square_context_crop_preserves_center_and_pads_edges(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[0:4, 0:4] = 255
        crop = square_context_crop(image, (-1, -1, 4, 3), context_scale=1.0)
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertTrue(np.any(crop == 127))
        self.assertTrue(np.any(crop == 255))

    def test_ppshitu_normalization_and_inner_product_search(self) -> None:
        gallery = normalize_features(
            np.asarray([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]], dtype=np.float32)
        )
        index = FlatInnerProductIndex(gallery)
        result = index.search(np.asarray([[0.9, 0.1]], dtype=np.float32), topk=2)
        self.assertEqual(result.indices.tolist(), [[0, 1]])
        self.assertGreater(float(result.scores[0, 0]), float(result.scores[0, 1]))

    def test_knife_feature_gallery_rejects_unknown_labels(self) -> None:
        with self.assertRaises(ValueError):
            KnifeFeatureGallery(
                labels=np.asarray(["other"]),
                embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                references=np.asarray(["other.jpg"]),
            )

    def test_service_accepts_rejects_and_marks_ambiguous(self) -> None:
        image = np.full((20, 30, 3), 255, dtype=np.uint8)
        matched = KnifeRecognitionService(StubEngine(), threshold=0.6, min_margin=0.1)
        result = matched.recognize(image)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.predicted_class, "k2")

        retrieval = KnifeRecognitionService(
            StubEngine(score_kind="cosine_similarity"),
            threshold=0.6,
            min_margin=0.1,
        ).recognize(image)
        self.assertEqual(retrieval.score_kind, "cosine_similarity")

        low = KnifeRecognitionService(
            StubEngine(scores=(0.55, 0.30)), threshold=0.6, min_margin=0.1
        ).recognize(image)
        self.assertEqual(low.status, "unknown")
        self.assertIsNone(low.predicted_class)

        ambiguous = KnifeRecognitionService(
            StubEngine(scores=(0.70, 0.65)), threshold=0.6, min_margin=0.1
        ).recognize(image)
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.predicted_class)

        other = KnifeRecognitionService(
            StubEngine(labels=("other", "k1"), scores=(0.95, 0.03)),
            threshold=0.6,
            min_margin=0.1,
        ).recognize(image)
        self.assertEqual(other.status, "unknown")

    def test_default_threshold_accepts_a_clear_top_candidate(self) -> None:
        image = np.full((20, 30, 3), 255, dtype=np.uint8)
        self.assertEqual(DEFAULT_KNIFE_THRESHOLD, 0.50)
        result = KnifeRecognitionService(
            StubEngine(scores=(0.55, 0.30)),
        ).recognize(image)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.predicted_class, "k2")

    def test_pipeline_processes_only_macro_k_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            crop_dir = temporary / "crops" / "k"
            crop_dir.mkdir(parents=True)
            cv2.imwrite(
                str(crop_dir / "knife.jpg"),
                np.full((20, 30, 3), 255, dtype=np.uint8),
            )
            handoff = {
                "schema": "stage1_yolo_handoff.v1",
                "base_dir": str(temporary),
                "frames": [
                    {
                        "frame_index": 7,
                        "detections": [
                            {
                                "id": "k-det",
                                "major_class": "k",
                                "confidence": 0.8,
                                "bbox_xyxy": [2, 3, 11, 13],
                                "crop_path": "crops/k/knife.jpg",
                            },
                            {
                                "id": "f-det",
                                "major_class": "f",
                                "confidence": 0.9,
                                "bbox_xyxy": [1, 2, 10, 12],
                                "crop_path": "crops/f/face.jpg",
                            },
                        ],
                    }
                ],
            }
            handoff_path = temporary / "stage1_handoff.json"
            handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
            output_path = temporary / "stage2.json"
            result = process_handoff(
                handoff_path,
                StubEngine(),
                output_path,
                threshold=0.6,
                min_margin=0.1,
            )

            self.assertEqual(result["summary"]["processed_k_detections"], 1)
            self.assertEqual(result["detections"][0]["predicted_class"], "k2")
            self.assertNotIn("raw_yolo_class", result["detections"][0])
            self.assertNotIn("probabilities", result["detections"][0])
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
