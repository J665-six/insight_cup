"""ONNX adapter for a PaddleClas-exported knife classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from insight_cup.common.crops import square_context_crop
from insight_cup.common.onnx import cpu_session_options
from insight_cup.paths import MODELS_ROOT
from insight_cup.vendor.paddleclas_core import Topk, create_operators


UPSTREAM_COMMIT = "f1233c18455b8acde4fc42ab0bea575fa06daa8e"
DEFAULT_MODEL_DIR = (
    MODELS_ROOT / "knife" / "pplcnetv2_base_knife10"
)
DEFAULT_LABELS = tuple(f"k{index}" for index in range(1, 11))
PREPROCESS_CONFIG = [
    {"ResizeImage": {"size": [224, 224]}},
    {
        "NormalizeImage": {
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "order": "",
            "channel_num": 3,
        }
    },
    {"ToCHWImage": None},
]


@dataclass(frozen=True)
class KnifePrediction:
    class_ids: tuple[int, ...]
    labels: tuple[str, ...]
    scores: tuple[float, ...]
    crop_shape: tuple[int, int]
    score_kind: str = "softmax_confidence"
    class_scores: dict[str, float] = field(default_factory=dict)


def _resolve_providers(provider: str) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    normalized = provider.strip().lower()
    if normalized == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install a compatible "
                "onnxruntime-gpu build or use --provider cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError("provider must be one of: auto, cpu, cuda")


def _load_labels(path: Path) -> tuple[str, ...]:
    labels: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        class_id, separator, label = value.partition(" ")
        if not separator or not class_id.isdigit() or not label:
            raise ValueError(f"Invalid label map line {line_number}: {line!r}")
        labels[int(class_id)] = label.strip().lower()
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("label IDs must be contiguous and start at zero")
    values = tuple(labels[index] for index in range(len(labels)))
    if values[:10] != DEFAULT_LABELS:
        raise ValueError(f"The first ten labels must be {DEFAULT_LABELS}, got {values[:10]}")
    if any(label not in {*DEFAULT_LABELS, "other"} for label in values):
        raise ValueError("Only k1...k10 and optional internal label 'other' are supported")
    return values


class PaddleClasImagePreprocessor:
    """Shared PaddleClas recognition preprocessing for BGR crops."""

    def __init__(self) -> None:
        self._operators = create_operators(PREPROCESS_CONFIG)

    def prepare(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a non-empty BGR image")
        if min(image.shape[:2]) <= 0:
            raise ValueError("image must have positive dimensions")
        original_shape = (int(image.shape[0]), int(image.shape[1]))
        if image.shape[0] != image.shape[1]:
            image = square_context_crop(
                image,
                (0, 0, image.shape[1], image.shape[0]),
                context_scale=1.0,
            )
        value = image[:, :, ::-1]
        for operator in self._operators:
            value = operator(value)
        return np.ascontiguousarray(value, dtype=np.float32), original_shape


class PaddleClasONNXEngine:
    """Run the officially exported PaddleClas classifier on BGR crops."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        provider: str = "auto",
        cpu_threads: int = 4,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for knife recognition") from exc

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.model_path = self.model_dir / "inference.onnx"
        self.labels_path = self.model_dir / "labels.txt"
        self.metadata_path = self.model_dir / "model.json"
        for required in (self.model_path, self.labels_path, self.metadata_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Knife model file not found: {required}. Train/export the model first."
                )

        self.labels = _load_labels(self.labels_path)
        self.model_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.model_metadata.get("output_activation") != "softmax":
            raise ValueError("Knife model must export softmax probabilities")

        options = cpu_session_options(ort, cpu_threads)
        self.providers = _resolve_providers(provider)
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=self.providers,
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise RuntimeError("Knife classifier must have one input and at least one output")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._preprocessor = PaddleClasImagePreprocessor()
        self._topk = Topk(topk=min(2, len(self.labels)), label_list=list(self.labels))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "paddleclas_onnx",
            "upstream_commit": UPSTREAM_COMMIT,
            "model_dir": str(self.model_dir),
            "architecture": self.model_metadata.get("architecture"),
            "labels": list(self.labels),
            "providers": list(self._session.get_providers()),
            "input_size": [224, 224],
        }

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        return self._preprocessor.prepare(image)

    def predict_batch(self, images: Iterable[np.ndarray]) -> list[KnifePrediction]:
        prepared = [self._prepare(image) for image in images]
        if not prepared:
            return []
        batch = np.stack([value[0] for value in prepared]).astype(np.float32, copy=False)
        probabilities = np.asarray(
            self._session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.labels):
            raise RuntimeError(
                "Knife model output shape does not match the configured labels: "
                f"{probabilities.shape} vs {len(self.labels)}"
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Knife model produced non-finite probabilities")
        row_sums = probabilities.sum(axis=1)
        if np.any(probabilities < -1e-5) or not np.allclose(row_sums, 1.0, atol=1e-3):
            raise RuntimeError("Knife model output is not a softmax probability distribution")

        ranked = self._topk(probabilities)
        return [
            KnifePrediction(
                class_ids=tuple(int(value) for value in result["class_ids"]),
                labels=tuple(str(value) for value in result["label_names"]),
                scores=tuple(float(value) for value in result["scores"]),
                crop_shape=prepared[index][1],
                class_scores={
                    label: float(probabilities[index, class_id])
                    for class_id, label in enumerate(self.labels)
                },
            )
            for index, result in enumerate(ranked)
        ]

    def predict(self, image: np.ndarray) -> KnifePrediction:
        return self.predict_batch([image])[0]
