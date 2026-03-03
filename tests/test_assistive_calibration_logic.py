from __future__ import annotations

from copy import deepcopy

from cv_practice.assistive.calibration import _derive_profile_from_samples
from cv_practice.assistive.types import UserProfile


def _empty_samples() -> dict[str, dict[str, list[float]]]:
    keys = ["open_hand", "fist", "pinch_control", "two_finger_tap"]
    return {k: {"open_ratio": [], "pinch_ratio": [], "two_finger_ratio": []} for k in keys}


def test_calibration_uses_prior_when_pose_samples_are_sparse() -> None:
    prior = UserProfile(name="u")
    samples = _empty_samples()
    samples["open_hand"]["open_ratio"] = [1.7, 1.72, 1.69]
    samples["fist"]["open_ratio"] = [0.88, 0.90, 0.86]
    samples["pinch_control"]["pinch_ratio"] = [0.15, 0.16, 0.14]
    samples["two_finger_tap"]["two_finger_ratio"] = [0.14, 0.13, 0.15]

    out = _derive_profile_from_samples(prior, samples, min_samples_per_pose=16)

    assert out.open_min == prior.open_min
    assert out.fist_max == prior.fist_max
    assert out.pinch_max == prior.pinch_max
    assert out.two_finger_tap_max == prior.two_finger_tap_max


def test_calibration_updates_thresholds_and_preserves_separations() -> None:
    prior = UserProfile(name="u")
    samples = _empty_samples()
    samples["open_hand"]["open_ratio"] = [1.82] * 20
    samples["open_hand"]["two_finger_ratio"] = [0.24] * 20
    samples["fist"]["open_ratio"] = [0.84] * 20
    samples["fist"]["pinch_ratio"] = [0.17] * 20
    samples["pinch_control"]["pinch_ratio"] = [0.10] * 20
    samples["two_finger_tap"]["two_finger_ratio"] = [0.11] * 20

    out = _derive_profile_from_samples(deepcopy(prior), samples, min_samples_per_pose=16)

    assert out.open_min > prior.open_min
    assert out.fist_max <= out.open_min - 0.08
    assert out.pinch_max >= out.pinch_min + 0.03
    assert out.open_two_finger_min >= out.two_finger_tap_max + 0.03
    assert out.fist_pinch_min >= out.pinch_max + 0.02
    assert out.lock_pinch_guard_ratio >= out.pinch_min + 0.02
