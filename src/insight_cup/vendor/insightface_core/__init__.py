"""Detection-and-recognition-only InsightFace runtime.

The inference implementation is vendored from upstream InsightFace. See
UPSTREAM.md and LICENSE in this package for provenance and modifications.
"""

from .app import FaceAnalysis

__version__ = "0.7.2-face-only"
__all__ = ["FaceAnalysis"]
