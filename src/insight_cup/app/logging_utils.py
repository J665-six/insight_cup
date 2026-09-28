"""Console logging and durable JSONL event journaling."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any


def configure_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("recognition")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    return logger


def _number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


class EventJournal:
    def __init__(self, path: str | Path, logger: logging.Logger) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._logger = logger

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._stream.flush()
        stage2 = event["stage2"]
        result = (
            stage2.get("predicted_class")
            or stage2.get("candidate_class")
            or stage2.get("status")
            or "-"
        )
        self._logger.info(
            "RESULT frame=%s id=%s label=%s probability=%s "
            "second=%s second_probability=%s",
            event["frame"]["index"],
            event["id"],
            result,
            _number(stage2.get("score")),
            stage2.get("second_best_class") or "-",
            _number(stage2.get("second_best_score")),
        )

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)
