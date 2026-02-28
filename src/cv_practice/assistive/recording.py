from __future__ import annotations

import json
import time
from pathlib import Path

from cv_practice.assistive.types import GesturePrediction


class GestureRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._path: Path | None = None
        self.active_label = "unknown"

    @property
    def is_recording(self) -> bool:
        return self._fh is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, label: str = "unknown") -> Path:
        if self._fh is not None:
            return self._path  # type: ignore[return-value]
        self.active_label = label
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = self.output_dir / f"recording_{stamp}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")
        return self._path

    def set_label(self, label: str) -> None:
        self.active_label = label

    def add_frame(self, prediction: GesturePrediction, lm_list: list[list[int]]) -> None:
        if self._fh is None:
            return
        payload = {
            "kind": "frame",
            "timestamp_ms": int(time.perf_counter() * 1000),
            "ground_truth": self.active_label,
            "predicted": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "lm_list": lm_list,
        }
        self._fh.write(json.dumps(payload))
        self._fh.write("\n")

    def add_command(self, command: str) -> None:
        if self._fh is None:
            return
        payload = {
            "kind": "command",
            "timestamp_ms": int(time.perf_counter() * 1000),
            "ground_truth": self.active_label,
            "command": command,
        }
        self._fh.write(json.dumps(payload))
        self._fh.write("\n")

    def stop(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

