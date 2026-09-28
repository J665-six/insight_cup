"""Focused InsightFace detection and identity-matching module."""

from .engine import FaceEmbeddingResult, InsightFaceEmbeddingEngine
from .gallery import FaceGallery, MatchResult
from .service import FaceRecognitionService, RecognitionResult

__all__ = [
    "FaceEmbeddingResult",
    "FaceGallery",
    "FaceRecognitionService",
    "InsightFaceEmbeddingEngine",
    "MatchResult",
    "RecognitionResult",
]
