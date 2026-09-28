"""Shared ONNX Runtime CPU session policy for bounded resource usage."""

from __future__ import annotations

from typing import Any


def cpu_session_options(ort: Any, cpu_threads: int) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(cpu_threads))
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options
