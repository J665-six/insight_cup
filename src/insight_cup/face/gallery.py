"""Face-gallery storage, enrollment, and cosine matching."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .engine import InsightFaceEmbeddingEngine


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IDENTITY_PATTERN = re.compile(r"f\d+", re.IGNORECASE)


def _natural_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([^\d]*)(\d+)", value.lower())
    if not match:
        return value.lower(), -1
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class MatchResult:
    status: str
    predicted_class: str | None
    candidate_class: str
    similarity: float
    second_best_class: str | None
    second_best_similarity: float | None
    margin: float | None
    class_scores: dict[str, float]


class FaceGallery:
    """An in-memory gallery containing normalized reference embeddings."""

    def __init__(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        references: np.ndarray | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.str_).reshape(-1)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != self.labels.size:
            raise ValueError("gallery labels and embeddings have incompatible shapes")
        if self.labels.size == 0:
            raise ValueError("gallery is empty")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("gallery contains invalid embeddings")
        self.embeddings = self.embeddings / norms
        if references is None:
            references = np.full(self.labels.shape, "", dtype=np.str_)
        self.references = np.asarray(references, dtype=np.str_).reshape(-1)
        if self.references.size != self.labels.size:
            raise ValueError("gallery references and labels have incompatible shapes")

    @property
    def identities(self) -> list[str]:
        return sorted(set(self.labels.tolist()), key=_natural_key)

    @classmethod
    def load(cls, path: str | Path) -> "FaceGallery":
        gallery_path = Path(path).expanduser().resolve()
        with np.load(gallery_path, allow_pickle=False) as payload:
            references = payload["references"] if "references" in payload else None
            return cls(payload["labels"], payload["embeddings"], references)

    def save(self, path: str | Path) -> Path:
        gallery_path = Path(path).expanduser().resolve()
        gallery_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = gallery_path.with_name(f".{gallery_path.name}.tmp")
        with temp_path.open("wb") as output:
            np.savez_compressed(
                output,
                labels=self.labels,
                embeddings=self.embeddings,
                references=self.references,
            )
        temp_path.replace(gallery_path)
        return gallery_path

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.40,
        min_margin: float = 0.03,
    ) -> MatchResult:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.size != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match the gallery")
        query_norm = float(np.linalg.norm(query))
        if not np.isfinite(query_norm) or query_norm <= 0.0:
            raise ValueError("query embedding is invalid")
        scores = self.embeddings @ (query / query_norm)

        identity_scores = {
            identity: float(np.max(scores[self.labels == identity]))
            for identity in self.identities
        }
        ranked = sorted(identity_scores.items(), key=lambda item: item[1], reverse=True)
        candidate, best_score = ranked[0]
        second_class = ranked[1][0] if len(ranked) > 1 else None
        second_score = ranked[1][1] if len(ranked) > 1 else None
        margin = best_score - second_score if second_score is not None else None

        if best_score < threshold:
            status = "unknown"
            predicted = None
        elif margin is not None and margin < min_margin:
            status = "ambiguous"
            predicted = None
        else:
            status = "matched"
            predicted = candidate
        return MatchResult(
            status=status,
            predicted_class=predicted,
            candidate_class=candidate,
            similarity=best_score,
            second_best_class=second_class,
            second_best_similarity=second_score,
            margin=margin,
            class_scores=identity_scores,
        )


def iter_reference_images(reference_dir: Path) -> Iterable[tuple[str, Path]]:
    identity_dirs = sorted(
        (path for path in reference_dir.iterdir() if path.is_dir()),
        key=lambda path: _natural_key(path.name),
    )
    for identity_dir in identity_dirs:
        identity = identity_dir.name.lower()
        if not IDENTITY_PATTERN.fullmatch(identity):
            continue
        for image_path in sorted(identity_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                yield identity, image_path


def build_gallery(
    reference_dir: str | Path,
    output_path: str | Path,
    engine: InsightFaceEmbeddingEngine,
) -> dict[str, Any]:
    """Extract reference embeddings from references/<identity>/*.jpg."""

    source_dir = Path(reference_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {source_dir}")

    labels: list[str] = []
    embeddings: list[np.ndarray] = []
    references: list[str] = []
    failures: list[dict[str, str]] = []
    attempted = 0
    for identity, image_path in iter_reference_images(source_dir):
        attempted += 1
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({"path": str(image_path), "reason": "read_error"})
            continue
        result = engine.extract(image)
        if result is None:
            failures.append({"path": str(image_path), "reason": "no_face"})
            continue
        labels.append(identity)
        embeddings.append(result.embedding)
        references.append(str(image_path.relative_to(source_dir)))

    if not embeddings:
        raise RuntimeError("No usable faces were found in the reference directory")

    gallery = FaceGallery(
        labels=np.asarray(labels, dtype=np.str_),
        embeddings=np.stack(embeddings).astype(np.float32),
        references=np.asarray(references, dtype=np.str_),
    )
    gallery_path = gallery.save(output_path)
    counts = Counter(labels)
    metadata = {
        "schema": "face_gallery.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference_dir": str(source_dir),
        "gallery_path": str(gallery_path),
        "model": engine.metadata,
        "attempted_images": attempted,
        "accepted_images": len(embeddings),
        "identity_counts": dict(sorted(counts.items(), key=lambda item: _natural_key(item[0]))),
        "failures": failures,
    }
    metadata_path = gallery_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
