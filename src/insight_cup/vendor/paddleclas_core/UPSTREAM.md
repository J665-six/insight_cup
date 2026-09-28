# Upstream provenance

This directory is a focused deployment subset of
[`PaddlePaddle/PaddleClas`](https://github.com/PaddlePaddle/PaddleClas).

- Upstream branch: `release/2.6`
- Upstream commit: `f1233c18455b8acde4fc42ab0bea575fa06daa8e`
- Upstream license: Apache-2.0, reproduced in `LICENSE`
- Retained behavior: classification resize/normalize/CHW preprocessing,
  `Topk` probability postprocessing, PP-ShiTu feature normalization and Faiss
  flat inner-product gallery search
- Training architecture: upstream `PPLCNetV2_base`; the complete upstream
  checkout is used only by the isolated training workflow

Project-specific changes are deliberately limited:

1. Detection, attributes, serving, mobile, GUI, benchmark and training-engine
   imports were removed from the production runtime. Retrieval keeps only the
   feature normalization and exact inner-product index needed by this project.
2. Paddle-only imports were removed so an officially exported ONNX model can
   run with ONNX Runtime without loading PaddlePaddle in the camera process.
3. Pillow resize constants use the current `Image.Resampling` API.
4. A missing label map and an unsupported resize backend raise explicit
   exceptions instead of continuing with an empty mapping.

The application-specific ONNX session, decision thresholds, YOLO handoff and
dataset grouping code lives in `src/insight_cup/knife/` and
`tools/data/export_knife_dataset.py`. Knife gallery construction lives in
`tools/data/build_knife_retrieval_gallery.py`; these project adapters are not
presented as upstream PaddleClas code.
