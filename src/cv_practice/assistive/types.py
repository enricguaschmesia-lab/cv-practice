from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GestureLabel = Literal[
    "unknown",
    "open_hand",
    "fist",
    "pinch_control",
    "two_finger_tap",
]

CommandName = Literal[
    "lock",
    "unlock",
    "volume_set",
    "volume_toggle_mute",
    "media_play_pause",
    "confirm",
]


@dataclass(slots=True)
class LandmarkFrame:
    timestamp_ms: int
    hand_index: int
    lm_list: list[list[int]]
    frame_size: tuple[int, int]


@dataclass(slots=True)
class GestureFeatures:
    valid: bool
    pinch_ratio: float
    open_ratio: float
    fist_ratio: float
    two_finger_ratio: float
    palm_scale_px: float
    index_tip_y_px: float
    pinch_delta: float = 0.0
    thumb_ext_ratio: float = 0.0
    index_ext_ratio: float = 0.0
    middle_ext_ratio: float = 0.0
    ring_ext_ratio: float = 0.0
    pinky_ext_ratio: float = 0.0


@dataclass(slots=True)
class GesturePrediction:
    label: GestureLabel
    confidence: float
    features: GestureFeatures


@dataclass(slots=True)
class CommandEvent:
    command: CommandName
    timestamp_ms: int
    value: float | None = None
    confidence: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class UserProfile:
    name: str
    hold_ms: int = 300
    cooldown_ms: int = 450
    volume_emit_ms: int = 120
    volume_deadzone: float = 0.03
    pinch_min: float = 0.12
    pinch_max: float = 0.38
    open_min: float = 1.32
    fist_max: float = 0.92
    two_finger_tap_max: float = 0.17
    smoothing_alpha: float = 0.2
    model_path: str = "models/hand_landmarker.task"
    extra: dict[str, float] = field(default_factory=dict)
