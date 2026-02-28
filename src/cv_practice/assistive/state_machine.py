from __future__ import annotations

from dataclasses import dataclass

from cv_practice.assistive.types import CommandEvent, GesturePrediction, UserProfile


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(slots=True)
class StateMachineMetrics:
    frames_seen: int = 0
    command_count: int = 0
    prediction_count: int = 0


class GestureStateMachine:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile
        self.locked = True
        self._active_label = "unknown"
        self._label_since_ms = 0
        self._last_emit: dict[str, int] = {}
        self._last_volume_value: float | None = None
        self._pinch_baseline: float | None = None
        self._tap_latched = False
        self.metrics = StateMachineMetrics()

    def _can_emit(self, command: str, now_ms: int, cooldown_ms: int | None = None) -> bool:
        limit = self.profile.cooldown_ms if cooldown_ms is None else cooldown_ms
        last = self._last_emit.get(command, -10_000_000)
        if now_ms - last < limit:
            return False
        self._last_emit[command] = now_ms
        return True

    def _held_long_enough(self, now_ms: int) -> bool:
        return now_ms - self._label_since_ms >= self.profile.hold_ms

    def update(self, prediction: GesturePrediction, t_ms: int) -> list[CommandEvent]:
        self.metrics.frames_seen += 1
        if prediction.label != "unknown":
            self.metrics.prediction_count += 1

        if prediction.label != self._active_label:
            self._active_label = prediction.label
            self._label_since_ms = t_ms
            if prediction.label != "pinch_control":
                self._pinch_baseline = None
            if prediction.label != "two_finger_tap":
                self._tap_latched = False
            return []

        events: list[CommandEvent] = []

        if prediction.label == "open_hand":
            if (
                self.locked
                and prediction.confidence >= 0.72
                and self._held_long_enough(t_ms)
                and self._can_emit("unlock", t_ms)
            ):
                self.locked = False
                events.append(
                    CommandEvent(
                        "unlock",
                        t_ms,
                        confidence=prediction.confidence,
                        reason="open_hold",
                    )
                )

        elif prediction.label == "fist":
            if (
                (not self.locked)
                and prediction.confidence >= 0.72
                and self._held_long_enough(t_ms)
                and self._can_emit("lock", t_ms)
            ):
                self.locked = True
                self._last_volume_value = None
                events.append(
                    CommandEvent(
                        "lock",
                        t_ms,
                        confidence=prediction.confidence,
                        reason="fist_hold",
                    )
                )

        elif prediction.label == "two_finger_tap":
            tap_hold_ms = max(120, self.profile.hold_ms // 2)
            if (
                (not self.locked)
                and (not self._tap_latched)
                and prediction.confidence >= 0.70
                and (t_ms - self._label_since_ms >= tap_hold_ms)
            ):
                if self._can_emit("media_play_pause", t_ms):
                    events.append(
                        CommandEvent(
                            "media_play_pause",
                            t_ms,
                            confidence=prediction.confidence,
                            reason="two_finger_tap_hold",
                        )
                    )
                    self._tap_latched = True

        elif prediction.label == "pinch_control":
            if self.locked:
                return []
            if prediction.confidence < 0.65:
                return []
            if not self._held_long_enough(t_ms):
                return []
            if self._pinch_baseline is None:
                self._pinch_baseline = prediction.features.pinch_ratio
                return []
            if not self._can_emit("volume_set", t_ms, cooldown_ms=self.profile.volume_emit_ms):
                return []

            ratio = prediction.features.pinch_ratio
            norm = (ratio - self.profile.pinch_min) / max(
                1e-6,
                (self.profile.pinch_max - self.profile.pinch_min),
            )
            volume_value = _clamp(1.0 - norm, 0.0, 1.0)
            if self._last_volume_value is not None:
                if abs(volume_value - self._last_volume_value) < self.profile.volume_deadzone:
                    return []
            self._last_volume_value = volume_value
            events.append(
                CommandEvent(
                    "volume_set",
                    t_ms,
                    value=volume_value,
                    confidence=prediction.confidence,
                    reason="pinch_ratio_control",
                )
            )

        if events:
            self.metrics.command_count += len(events)
        return events
