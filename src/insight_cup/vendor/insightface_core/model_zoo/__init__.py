"""InsightFace model routing restricted to SCRFD and ArcFace."""

from .arcface_onnx import ArcFaceONNX
from .model_zoo import get_model
from .scrfd import SCRFD

__all__ = ["ArcFaceONNX", "SCRFD", "get_model"]
