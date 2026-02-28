from __future__ import annotations

from cv_practice.assistive.state_machine import GestureStateMachine
from cv_practice.assistive.types import GestureFeatures, GesturePrediction, UserProfile


def _pred(label: str, conf: float = 0.95, pinch_ratio: float = 0.2) -> GesturePrediction:
    feat = GestureFeatures(
        valid=True,
        pinch_ratio=pinch_ratio,
        open_ratio=1.4,
        fist_ratio=0.6,
        two_finger_ratio=0.1,
        palm_scale_px=120.0,
        index_tip_y_px=200.0,
    )
    return GesturePrediction(label=label, confidence=conf, features=feat)


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
