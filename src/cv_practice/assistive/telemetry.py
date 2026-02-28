from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cv_practice.assistive.types import CommandEvent, GesturePrediction


@dataclass(slots=True)
class SessionSummary:
    frames: int
    commands: int
    runtime_s: float
    fps_avg: float
    latency_p50_ms: float
    latency_p95_ms: float


class TelemetryLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        self._events_path = self.output_dir / f"session_{session_id}.jsonl"
        self._summary_path = self.output_dir / f"summary_{session_id}.json"
        self._start = time.perf_counter()
        self._frames = 0
        self._commands = 0
        self._latencies_ms: list[float] = []

    def log_frame(self, capture_to_decision_ms: float, prediction: GesturePrediction) -> None:
        self._frames += 1
        self._latencies_ms.append(float(capture_to_decision_ms))
        payload = {
            "kind": "frame",
            "prediction": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "latency_ms": round(capture_to_decision_ms, 3),
            "timestamp_ms": int(time.perf_counter() * 1000),
        }
        self._append(payload)

    def log_command(self, event: CommandEvent, executed: bool) -> None:
        self._commands += 1
        payload = {
            "kind": "command",
            **asdict(event),
            "executed": bool(executed),
            "timestamp_ms": int(time.perf_counter() * 1000),
        }
        self._append(payload)

    def finalize(self) -> SessionSummary:
        runtime_s = max(1e-6, time.perf_counter() - self._start)
        fps_avg = self._frames / runtime_s
        p50 = _percentile(self._latencies_ms, 0.5)
        p95 = _percentile(self._latencies_ms, 0.95)
        summary = SessionSummary(
            frames=self._frames,
            commands=self._commands,
            runtime_s=runtime_s,
            fps_avg=fps_avg,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
        )
        with self._summary_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        return summary

    def _append(self, payload: dict) -> None:
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(len(ordered) - 1, lo + 1)
    if lo == hi:
        return float(ordered[lo])
    t = idx - lo
    return (1.0 - t) * ordered[lo] + t * ordered[hi]
