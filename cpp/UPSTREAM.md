# C++ Runtime Upstream Sources

The C++ deployment runtime keeps the exported upstream ONNX graphs unchanged.
Only image preprocessing, output decoding, and project-specific routing are
implemented in C++.

## InsightFace

- Repository: `https://github.com/deepinsight/insightface.git`
- Commit: `7fadd420c2351d0ffa8cac403421c1a3ed733365`
- License: `src/insight_cup/vendor/insightface_core/LICENSE`
- Ported behavior:
  - `python-package/insightface/model_zoo/scrfd.py`
  - `python-package/insightface/model_zoo/arcface_onnx.py`
  - `python-package/insightface/utils/face_align.py`

The C++ runtime includes only SCRFD face detection, five-point alignment, and
ArcFace embedding extraction. It does not include 3D face reconstruction,
liveness detection, face swapping, attributes, or the InsightFace GUI.

## PaddleClas

- Repository: `https://github.com/PaddlePaddle/PaddleClas.git`
- Branch: `release/2.6`
- Commit: `f1233c18455b8acde4fc42ab0bea575fa06daa8e`
- License: `src/insight_cup/vendor/paddleclas_core/LICENSE`
- Ported behavior:
  - `deploy/python/preprocess.py`: resize, normalize, and CHW conversion
  - `deploy/python/postprocess.py`: classification Top-k
  - PP-ShiTuV2 feature L2 normalization and exact inner-product retrieval

The C++ runtime does not include PaddleClas training, serving, mobile, or object
detection components. Training and ONNX export remain external data tools.
