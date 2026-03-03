from __future__ import annotations

from agcp.state_machine import GestureStateMachine
from agcp.types import (
    GestureDebugInfo,
    GestureFeatures,
    GesturePrediction,
    UserProfile,
)


def _pred(
    label: str,
    conf: float = 0.95,
    pinch_ratio: float = 0.2,
    *,
    thumb_ext: float = 1.05,
    index_ext: float = 1.45,
    middle_ext: float = 1.12,
    ring_ext: float = 1.10,
    pinky_ext: float = 1.08,
) -> GesturePrediction:
    feat = GestureFeatures(
        valid=True,
        pinch_ratio=pinch_ratio,
        open_ratio=1.4,
        fist_ratio=0.6,
        two_finger_ratio=0.1,
        palm_scale_px=120.0,
        index_tip_y_px=200.0,
        thumb_ext_ratio=thumb_ext,
        index_ext_ratio=index_ext,
        middle_ext_ratio=middle_ext,
        ring_ext_ratio=ring_ext,
        pinky_ext_ratio=pinky_ext,
    )
    return GesturePrediction(label=label, confidence=conf, features=feat)


def _pred_with_debug(
    label: str,
    *,
    conf: float,
    pinch_ratio: float,
    pinch_conf: float,
    fist_required: float = 0.70,
) -> GesturePrediction:
    pred = _pred(label, conf=conf, pinch_ratio=pinch_ratio)
    pred.debug = GestureDebugInfo(
        input_valid=True,
        fist_required=fist_required,
        pinch_conf=pinch_conf,
    )
    return pred


def test_state_machine_hold_and_cooldown() -> None:
    profile = UserProfile(name="t", hold_ms=200, cooldown_ms=400)
    sm = GestureStateMachine(profile)

    assert sm.locked is True
    assert sm.update(_pred("open_hand"), 1000) == []
    ev = sm.update(_pred("open_hand"), 1300)
    assert len(ev) == 1
    assert ev[0].command == "unlock"
    assert sm.locked is False

    # Still in cooldown, should not fire again.
    assert sm.update(_pred("open_hand"), 1400) == []


def test_state_machine_volume_deadzone_blocks_duplicates() -> None:
    profile = UserProfile(
        name="t",
        hold_ms=100,
        cooldown_ms=250,
        volume_emit_ms=100,
        volume_deadzone=0.1,
        pinch_min=0.1,
        pinch_max=0.7,
    )
    sm = GestureStateMachine(profile)
    sm.locked = False

    assert sm.update(_pred("pinch_control", pinch_ratio=0.2), 1000) == []
    assert sm.update(_pred("pinch_control", pinch_ratio=0.2), 1120) == []
    first = sm.update(_pred("pinch_control", pinch_ratio=0.25), 1240)
    assert len(first) == 1
    assert first[0].command == "volume_set"

    # Small delta below deadzone.
    assert sm.update(_pred("pinch_control", pinch_ratio=0.255), 1360) == []


def test_two_finger_tap_is_latched_until_release() -> None:
    profile = UserProfile(name="t", hold_ms=240, cooldown_ms=250)
    sm = GestureStateMachine(profile)
    sm.locked = False

    assert sm.update(_pred("two_finger_tap"), 1000) == []
    first = sm.update(_pred("two_finger_tap"), 1140)
    assert len(first) == 1
    assert first[0].command == "media_play_pause"

    # Same hold should not retrigger while latched.
    assert sm.update(_pred("two_finger_tap"), 1300) == []
    assert sm.update(_pred("two_finger_tap"), 1500) == []

    # Release to unknown, then new tap can fire again.
    assert sm.update(_pred("unknown", conf=0.0), 1700) == []
    assert sm.update(_pred("two_finger_tap"), 1800) == []
    second = sm.update(_pred("two_finger_tap"), 1950)
    assert len(second) == 1
    assert second[0].command == "media_play_pause"


def test_lock_guard_blocks_fist_when_pinch_signature_is_present() -> None:
    profile = UserProfile(
        name="t",
        hold_ms=150,
        cooldown_ms=250,
        lock_pinch_guard_ratio=0.24,
        lock_pinch_guard_conf=0.55,
    )
    sm = GestureStateMachine(profile)
    sm.locked = False

    assert sm.update(_pred("fist", conf=0.9, pinch_ratio=0.16), 1000) == []
    blocked = sm.update(_pred("fist", conf=0.9, pinch_ratio=0.16), 1200)
    assert blocked == []
    assert sm.locked is False


def test_lock_uses_classifier_fist_required_threshold() -> None:
    profile = UserProfile(name="t", hold_ms=150, cooldown_ms=250)
    sm = GestureStateMachine(profile)
    sm.locked = False

    first = sm.update(
        _pred_with_debug("fist", conf=0.70, pinch_ratio=0.30, pinch_conf=0.20, fist_required=0.70),
        1000,
    )
    second = sm.update(
        _pred_with_debug("fist", conf=0.70, pinch_ratio=0.30, pinch_conf=0.20, fist_required=0.70),
        1200,
    )
    assert first == []
    assert len(second) == 1
    assert second[0].command == "lock"
    assert sm.locked is True


def test_lock_guard_with_debug_requires_ratio_and_pinch_conf() -> None:
    profile = UserProfile(
        name="t",
        hold_ms=150,
        cooldown_ms=250,
        lock_pinch_guard_ratio=0.24,
        lock_pinch_guard_conf=0.55,
    )

    # Low pinch ratio but weak pinch confidence: allow lock.
    sm_allow = GestureStateMachine(profile)
    sm_allow.locked = False
    sm_allow.update(
        _pred_with_debug("fist", conf=0.9, pinch_ratio=0.16, pinch_conf=0.30),
        1000,
    )
    allow = sm_allow.update(
        _pred_with_debug("fist", conf=0.9, pinch_ratio=0.16, pinch_conf=0.30),
        1200,
    )
    assert len(allow) == 1
    assert allow[0].command == "lock"
    assert sm_allow.locked is True

    # Low pinch ratio and strong pinch confidence: block lock.
    sm_block = GestureStateMachine(profile)
    sm_block.locked = False
    sm_block.update(
        _pred_with_debug("fist", conf=0.9, pinch_ratio=0.16, pinch_conf=0.70),
        1000,
    )
    blocked = sm_block.update(
        _pred_with_debug("fist", conf=0.9, pinch_ratio=0.16, pinch_conf=0.70),
        1200,
    )
    assert blocked == []
    assert sm_block.locked is False


def test_pinch_session_persists_after_trigger_when_pose_is_preserved() -> None:
    profile = UserProfile(
        name="t",
        hold_ms=100,
        cooldown_ms=250,
        volume_emit_ms=100,
        volume_deadzone=0.0,
        pinch_min=0.1,
        pinch_max=0.7,
    )
    sm = GestureStateMachine(profile)
    sm.locked = False

    assert sm.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1000) == []
    assert sm.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1120) == []
    events = sm.update(_pred("unknown", conf=0.2, pinch_ratio=0.26), 1240)
    assert len(events) == 1
    assert events[0].command == "volume_set"
    assert bool(sm.debug_snapshot(1240)["pinch_session_active"]) is True


def test_pinch_session_stops_when_non_pinch_pose_change_occurs() -> None:
    profile = UserProfile(
        name="t",
        hold_ms=100,
        cooldown_ms=250,
        volume_emit_ms=100,
        volume_deadzone=0.0,
        pinch_min=0.1,
        pinch_max=0.7,
    )

    # Break by extending one of the other fingers.
    sm_extend = GestureStateMachine(profile)
    sm_extend.locked = False
    sm_extend.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1000)
    sm_extend.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1120)
    assert sm_extend.update(
        _pred("unknown", conf=0.2, pinch_ratio=0.26, middle_ext=1.45),
        1240,
    ) == []
    assert bool(sm_extend.debug_snapshot(1240)["pinch_session_active"]) is False

    # Break by folding index or thumb.
    sm_fold = GestureStateMachine(profile)
    sm_fold.locked = False
    sm_fold.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1000)
    sm_fold.update(_pred("pinch_control", conf=0.9, pinch_ratio=0.14), 1120)
    assert sm_fold.update(
        _pred("unknown", conf=0.2, pinch_ratio=0.26, index_ext=1.05),
        1240,
    ) == []
    assert bool(sm_fold.debug_snapshot(1240)["pinch_session_active"]) is False

