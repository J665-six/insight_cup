# Upstream provenance

This directory is a focused subset of
[`deepinsight/insightface`](https://github.com/deepinsight/insightface).

- Upstream tag: `model-zoo`
- Upstream commit: `7fadd420c2351d0ffa8cac403421c1a3ed733365`
- Upstream license: MIT, reproduced in `LICENSE`
- Retained inference code: `FaceAnalysis`, `SCRFD`, `ArcFaceONNX`, five-point
  face alignment, and the model download helpers they require

Project-specific changes are deliberately small:

1. The package is isolated as `insight_cup.vendor.insightface_core` so it
   cannot accidentally import a globally installed InsightFace fork.
2. Model routing accepts only SCRFD detection models and 512-dimensional
   ArcFace recognition models.
3. Model routing forwards application-provided ONNX Runtime session options so
   CPU thread limits and idle-spin policy reach the retained upstream models.
4. Imports, the `Face.sex` attribute helper, and drawing code for landmark/3D,
   attributes, face swap, mask rendering, GUI, and training modules were
   removed.
5. Liveness is not included. It was not part of this pinned upstream tag and
   there is no liveness hook in this package.

The application-specific gallery and YOLO handoff logic lives separately in
`src/insight_cup/face/`; it is not presented as upstream InsightFace code.
