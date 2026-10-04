#!/usr/bin/env python3
"""Convert the project's ONNX models to RK3588 FP16 RKNN models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rknn.api import RKNN


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "models" / "rknn" / "rk3588_fp16"


@dataclass(frozen=True)
class ModelSpec:
    source: Path
    input_name: str
    input_shape: tuple[int, int, int, int]
    output_name: str


MODELS = {
    "yolo": ModelSpec(
        source=REPOSITORY_ROOT / "models" / "yolo" / "best.onnx",
        input_name="images",
        input_shape=(1, 3, 640, 640),
        output_name="yolo_best.rknn",
    ),
    "scrfd": ModelSpec(
        source=(
            REPOSITORY_ROOT
            / "models"
            / "face"
            / "models"
            / "face_only_r50"
            / "det_10g.onnx"
        ),
        input_name="input.1",
        input_shape=(1, 3, 320, 320),
        output_name="scrfd_det_10g.rknn",
    ),
    "arcface": ModelSpec(
        source=(
            REPOSITORY_ROOT
            / "models"
            / "face"
            / "models"
            / "face_only_r50"
            / "w600k_r50.onnx"
        ),
        input_name="input.1",
        input_shape=(1, 3, 112, 112),
        output_name="arcface_w600k_r50.rknn",
    ),
    "knife": ModelSpec(
        source=(
            REPOSITORY_ROOT
            / "models"
            / "knife"
            / "pplcnetv2_base_knife10"
            / "inference.onnx"
        ),
        input_name="x",
        input_shape=(1, 3, 224, 224),
        output_name="knife_pplcnetv2.rknn",
    ),
}


def require_success(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with code {result}")


def convert(name: str, spec: ModelSpec, output_dir: Path, verbose: bool) -> Path:
    if not spec.source.is_file():
        raise FileNotFoundError(spec.source)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / spec.output_name
    print(f"[{name}] {spec.source.relative_to(REPOSITORY_ROOT)} -> {destination}")

    rknn = RKNN(verbose=verbose)
    try:
        require_success(
            rknn.config(target_platform="rk3588", float_dtype="float16"),
            f"{name}: config",
        )
        require_success(
            rknn.load_onnx(
                model=str(spec.source),
                inputs=[spec.input_name],
                input_size_list=[list(spec.input_shape)],
            ),
            f"{name}: load_onnx",
        )
        require_success(
            rknn.build(do_quantization=False),
            f"{name}: build",
        )
        require_success(
            rknn.export_rknn(str(destination)),
            f"{name}: export_rknn",
        )
    finally:
        rknn.release()

    print(f"[{name}] wrote {destination.stat().st_size:,} bytes")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODELS),
        default=list(MODELS),
        help="models to convert (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--verbose", action="store_true", help="enable RKNN logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in args.models:
        convert(name, MODELS[name], args.output_dir.resolve(), args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
