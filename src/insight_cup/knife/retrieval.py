"""PP-ShiTuV2 feature extraction and gallery matching for knife crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from insight_cup.common.onnx import cpu_session_options
from insight_cup.paths import DATA_ROOT, MODELS_ROOT
from insight_cup.vendor.paddleclas_core import (
    FlatInnerProductIndex,
    normalize_features,
)

from .engine import (
    DEFAULT_LABELS,
    UPSTREAM_COMMIT,
    KnifePrediction,
    PaddleClasImagePreprocessor,
    _resolve_providers,
)


DEFAULT_RETRIEVAL_MODEL_DIR = MODELS_ROOT / "knife" / "ppshitu_v2_retrieval"
DEFAULT_RETRIEVAL_MODEL = DEFAULT_RETRIEVAL_MODEL_DIR / "inference.onnx"
DEFAULT_RETRIEVAL_GALLERY = (
    DATA_ROOT / "knife_gallery" / "trainv5_ppshitu_v2.npz"
)


@dataclass(frozen=True)
class KnifeFeatureGallery:
    labels: np.ndarray
    embeddings: np.ndarray
    references: np.ndarray

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.str_).reshape(-1)
        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        references = np.asarray(self.references, dtype=np.str_).reshape(-1)
        if embeddings.ndim != 2 or embeddings.shape[0] != labels.size:
            raise ValueError("knife gallery labels and embeddings are incompatible")
        if labels.size == 0 or references.size != labels.size:
            raise ValueError("knife gallery is empty or has invalid references")
        if any(label not in DEFAULT_LABELS for label in labels.tolist()):
            raise ValueError("knife gallery labels must be k1...k10")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "embeddings", normalize_features(embeddings))
        object.__setattr__(self, "references", references)

    @classmethod
    def load(cls, path: str | Path) -> "KnifeFeatureGallery":
        gallery_path = Path(path).expanduser().resolve()
        with np.load(gallery_path, allow_pickle=False) as payload:
            return cls(
                labels=payload["labels"],
                embeddings=payload["embeddings"],
                references=payload["references"],
            )

    def save(self, path: str | Path) -> Path:
        gallery_path = Path(path).expanduser().resolve()
        gallery_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = gallery_path.with_name(f".{gallery_path.name}.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                labels=self.labels,
                embeddings=self.embeddings,
                references=self.references,
            )
        temporary.replace(gallery_path)
        return gallery_path


class PaddleClasFeatureExtractor:
    """ONNX adapter for the official PP-ShiTuV2 recognition model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RETRIEVAL_MODEL,
        provider: str = "auto",
        cpu_threads: int = 4,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for knife retrieval") from exc

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Knife retrieval model not found: {self.model_path}")
        options = cpu_session_options(ort, cpu_threads)
        self.providers = _resolve_providers(provider)
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=self.providers,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("PP-ShiTu feature model must have one input and output")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        output_shape = outputs[0].shape
        self.embedding_size = int(output_shape[1]) if len(output_shape) == 2 else 0
        if self.embedding_size <= 0:
            raise RuntimeError("PP-ShiTu feature model has an invalid output shape")
        self._preprocessor = PaddleClasImagePreprocessor()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "paddleclas_ppshitu_v2_onnx",
            "upstream_commit": UPSTREAM_COMMIT,
            "model_path": str(self.model_path),
            "architecture": "GeneralRecognitionV2_PPLCNetV2_base",
            "embedding_size": self.embedding_size,
            "providers": list(self._session.get_providers()),
            "input_size": [224, 224],
            "feature_normalize": True,
        }

    def extract_batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        prepared = [self._preprocessor.prepare(image)[0] for image in images]
        if not prepared:
            return np.empty((0, self.embedding_size), dtype=np.float32)
        batch = np.stack(prepared).astype(np.float32, copy=False)
        features = np.asarray(
            self._session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        if features.shape != (len(prepared), self.embedding_size):
            raise RuntimeError(f"Unexpected PP-ShiTu feature shape: {features.shape}")
        return normalize_features(features)

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self.extract_batch([image])[0]


class PaddleClasRetrievalEngine:
    """Match PP-ShiTuV2 embeddings against a labeled knife gallery."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RETRIEVAL_MODEL,
        gallery_path: str | Path = DEFAULT_RETRIEVAL_GALLERY,
        provider: str = "auto",
        cpu_threads: int = 4,
    ) -> None:
        self.extractor = PaddleClasFeatureExtractor(
            model_path=model_path,
            provider=provider,
            cpu_threads=cpu_threads,
        )
        self.gallery_path = Path(gallery_path).expanduser().resolve()
        self.gallery = KnifeFeatureGallery.load(self.gallery_path)
        if self.gallery.embeddings.shape[1] != self.extractor.embedding_size:
            raise ValueError("knife gallery and PP-ShiTu model dimensions do not match")
        self._index = FlatInnerProductIndex(self.gallery.embeddings)
        self.labels = DEFAULT_LABELS

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.extractor.metadata,
            "gallery": str(self.gallery_path),
            "gallery_references": int(self.gallery.labels.size),
            "labels": list(DEFAULT_LABELS),
            "index_method": "Faiss IndexFlatIP",
            "distance": "cosine_similarity",
        }

    def predict_batch(self, images: Iterable[np.ndarray]) -> list[KnifePrediction]:
        image_list = list(images)
        if not image_list:
            return []
        features = self.extractor.extract_batch(image_list)
        search = self._index.search(features, self._index.count)
        predictions: list[KnifePrediction] = []
        for row, image in enumerate(image_list):
            ranked_labels: list[str] = []
            ranked_scores: list[float] = []
            for score, index in zip(search.scores[row], search.indices[row]):
                label = str(self.gallery.labels[int(index)])
                if label in ranked_labels:
                    continue
                ranked_labels.append(label)
                ranked_scores.append(float(score))
                if len(ranked_labels) == len(DEFAULT_LABELS):
                    break
            predictions.append(
                KnifePrediction(
                    class_ids=tuple(
                        DEFAULT_LABELS.index(label) for label in ranked_labels[:2]
                    ),
                    labels=tuple(ranked_labels[:2]),
                    scores=tuple(ranked_scores[:2]),
                    crop_shape=(int(image.shape[0]), int(image.shape[1])),
                    score_kind="cosine_similarity",
                    class_scores=dict(zip(ranked_labels, ranked_scores)),
                )
            )
        return predictions

    def predict(self, image: np.ndarray) -> KnifePrediction:
        return self.predict_batch([image])[0]
