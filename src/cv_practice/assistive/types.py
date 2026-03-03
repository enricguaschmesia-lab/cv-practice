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
    debug: "GestureDebugInfo | None" = None


@dataclass(slots=True)
class GestureDebugInfo:
    input_valid: bool
    index_ext_conf: float = 0.0
    index_fold_conf: float = 0.0
    middle_ext_conf: float = 0.0
    middle_fold_conf: float = 0.0
    ring_ext_conf: float = 0.0
    ring_fold_conf: float = 0.0
    pinky_ext_conf: float = 0.0
    pinky_fold_conf: float = 0.0
    thumb_open_conf: float = 0.0
    thumb_fold_conf: float = 0.0
    open_spread_conf: float = 0.0
    fist_non_pinch_conf: float = 0.0
    pinch_not_fist_conf: float = 0.0
    two_finger_close_conf: float = 0.0
    pinch_base_conf: float = 0.0
    pinch_sep_conf: float = 0.0
    open_conf: float = 0.0
    fist_conf: float = 0.0
    open_full_conf: float = 0.0
    fist_full_conf: float = 0.0
    pinch_conf: float = 0.0
    two_finger_conf: float = 0.0
    open_required: float = 0.70
    fist_required: float = 0.70
    pinch_required: float = 0.70
    two_finger_required: float = 0.70
    open_fist_margin_required: float = 0.08
    open_tap_margin_required: float = 0.06
    fist_pinch_margin_required: float = 0.07
    pinch_fist_margin_required: float = 0.04
    two_finger_vs_pinch_ratio_required: float = 0.92


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
    fist_pinch_min: float = 0.30
    open_min: float = 1.32
    fist_max: float = 0.92
    two_finger_tap_max: float = 0.17
    open_two_finger_min: float = 0.24
    open_fist_margin: float = 0.08
    open_tap_margin: float = 0.06
    fist_pinch_margin: float = 0.07
    pinch_fist_margin: float = 0.04
    two_finger_vs_pinch_ratio: float = 0.92
    lock_pinch_guard_ratio: float = 0.22
    lock_pinch_guard_conf: float = 0.58
    smoothing_alpha: float = 0.2
    model_path: str = "models/hand_landmarker.task"
    extra: dict[str, float] = field(default_factory=dict)
