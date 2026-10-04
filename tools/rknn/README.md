# RK3588 model conversion

The generated models use RKNN Toolkit2 2.3.2, target RK3588, FP16, and a
fixed batch size of one. They keep preprocessing outside the model.

Convert every model with the locally installed Toolkit2 dependencies:

```bash
PYTHONPATH=../../.cache/insight-cup-rknn/site \
  .venv/bin/python tools/rknn/convert_models.py
```

Convert selected models by passing one or more of `yolo`, `scrfd`, `arcface`,
and `knife`.

The OS-provided RKNN Runtime 1.4.0 cannot load Toolkit2 2.3.2 models. Build
the smoke test against the matching local Runtime and run it on each model.
The smoke test validates that the file exists and is non-empty, then exercises
the same `rknn_init` path mode used by the production runtime without reading
the complete model into an application buffer:

```bash
RKNN_ROOT=../../.cache/insight-cup-rknn/runtime-2.3.2
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  tools/rknn/rknn_smoke.cpp \
  -I"$RKNN_ROOT" -L"$RKNN_ROOT" -Wl,-rpath,"$RKNN_ROOT" -lrknnrt \
  -o build/insight-cup-rknn-smoke
build/insight-cup-rknn-smoke models/rknn/rk3588_fp16/yolo_best.rknn
```

RKNN exposes these model inputs as FP16 NHWC tensors. When using
`rknn_inputs_set`, submit normalized FP32 input with the reported NHWC format;
the runtime performs the FP32-to-FP16 conversion.
