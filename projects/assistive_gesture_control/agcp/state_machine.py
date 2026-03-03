from __future__ import annotations

from dataclasses import dataclass

from .types import CommandEvent, GesturePrediction, UserProfile


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
        self._pinch_session_active = False
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

    def _reset_pinch_session(self) -> None:
        self._pinch_baseline = None
        self._pinch_session_active = False
        self._last_volume_value = None

    @staticmethod
    def _pinch_session_pose_valid(prediction: GesturePrediction) -> bool:
        f = prediction.features
        # Pinch session stays active while index+thumb remain extended and
        # middle/ring/pinky stay non-extended.
        index_extended = f.index_ext_ratio >= 1.18
        thumb_extended = f.thumb_ext_ratio >= 0.95
        middle_non_extended = f.middle_ext_ratio <= 1.34
        ring_non_extended = f.ring_ext_ratio <= 1.34
        pinky_non_extended = f.pinky_ext_ratio <= 1.34
        return (
            index_extended
            and thumb_extended
            and middle_non_extended
            and ring_non_extended
            and pinky_non_extended
        )

    @staticmethod
    def _gesture_required_confidence(prediction: GesturePrediction, label: str, fallback: float) -> float:
        if prediction.debug is None:
            return fallback
        if label == "open_hand":
            return prediction.debug.open_required
        if label == "fist":
            return prediction.debug.fist_required
        if label == "two_finger_tap":
            return prediction.debug.two_finger_required
        if label == "pinch_control":
            return prediction.debug.pinch_required
        return fallback

    def debug_snapshot(self, now_ms: int) -> dict[str, float | int | bool | str | None]:
        hold_elapsed_ms = max(0, now_ms - self._label_since_ms)
        return {
            "locked": self.locked,
            "active_label": self._active_label,
            "hold_elapsed_ms": hold_elapsed_ms,
            "hold_required_ms": self.profile.hold_ms,
            "tap_hold_required_ms": max(120, self.profile.hold_ms // 2),
            "pinch_baseline": self._pinch_baseline,
            "pinch_session_active": self._pinch_session_active,
            "tap_latched": self._tap_latched,
        }

    def _emit_volume_event(
        self,
        prediction: GesturePrediction,
        t_ms: int,
        *,
        reason: str,
    ) -> list[CommandEvent]:
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
        return [
            CommandEvent(
                "volume_set",
                t_ms,
                value=volume_value,
                confidence=prediction.confidence,
                reason=reason,
            )
        ]

    def update(self, prediction: GesturePrediction, t_ms: int) -> list[CommandEvent]:
        self.metrics.frames_seen += 1
        if prediction.label != "unknown":
            self.metrics.prediction_count += 1

        if self._pinch_session_active:
            if self.locked or (not self._pinch_session_pose_valid(prediction)):
                self._reset_pinch_session()
            else:
                events = self._emit_volume_event(
                    prediction,
                    t_ms,
                    reason="pinch_session_control",
                )
                if events:
                    self.metrics.command_count += len(events)
                return events

        if prediction.label != self._active_label:
            self._active_label = prediction.label
            self._label_since_ms = t_ms
            if prediction.label != "pinch_control":
                self._reset_pinch_session()
            if prediction.label != "two_finger_tap":
                self._tap_latched = False
            return []

        events: list[CommandEvent] = []

        if prediction.label == "open_hand":
            unlock_required = self._gesture_required_confidence(prediction, "open_hand", 0.70)
            if (
                self.locked
                and prediction.confidence >= unlock_required
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
            lock_required = self._gesture_required_confidence(prediction, "fist", 0.70)
            if (
                (not self.locked)
                and prediction.confidence >= lock_required
                and self._held_long_enough(t_ms)
                and self._can_emit("lock", t_ms)
            ):
                if prediction.debug is None:
                    if prediction.features.pinch_ratio <= self.profile.lock_pinch_guard_ratio:
                        return []
                else:
                    pinch_like_signature = (
                        prediction.features.pinch_ratio <= self.profile.lock_pinch_guard_ratio
                        and prediction.debug.pinch_conf >= self.profile.lock_pinch_guard_conf
                    )
                    if pinch_like_signature:
                        return []
                self.locked = True
                self._reset_pinch_session()
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
                self._reset_pinch_session()
                return []
            pinch_required = self._gesture_required_confidence(prediction, "pinch_control", 0.70)
            if prediction.confidence < max(0.65, pinch_required):
                return []
            if not self._held_long_enough(t_ms):
                return []
            if not self._pinch_session_pose_valid(prediction):
                self._reset_pinch_session()
                return []
            self._pinch_session_active = True
            events.extend(
                self._emit_volume_event(
                    prediction,
                    t_ms,
                    reason="pinch_ratio_control",
                )
            )

        if events:
            self.metrics.command_count += len(events)
        return events

