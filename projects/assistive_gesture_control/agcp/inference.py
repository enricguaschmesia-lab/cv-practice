from __future__ import annotations

import math

from .types import (
    GestureDebugInfo,
    GestureFeatures,
    GesturePrediction,
    UserProfile,
)


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _pt(lm_list: list[list[int]], idx: int) -> tuple[int, int]:
    return lm_list[idx][1], lm_list[idx][2]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ext_ratio(
    lm_list: list[list[int]],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
) -> float:
    tip = _pt(lm_list, tip_idx)
    pip = _pt(lm_list, pip_idx)
    mcp = _pt(lm_list, mcp_idx)
    return _dist(tip, mcp) / max(1e-6, _dist(pip, mcp))


def extract_features(
    lm_list: list[list[int]],
    profile: UserProfile,
    prev_features: GestureFeatures | None = None,
) -> GestureFeatures:
    if len(lm_list) < 21:
        return GestureFeatures(
            valid=False,
            pinch_ratio=1.0,
            open_ratio=0.0,
            fist_ratio=0.0,
            two_finger_ratio=1.0,
            palm_scale_px=1.0,
            index_tip_y_px=0.0,
            pinch_delta=0.0,
            thumb_ext_ratio=0.0,
            index_ext_ratio=0.0,
            middle_ext_ratio=0.0,
            ring_ext_ratio=0.0,
            pinky_ext_ratio=0.0,
        )

    wrist = _pt(lm_list, 0)
    thumb_tip = _pt(lm_list, 4)
    index_tip = _pt(lm_list, 8)
    middle_tip = _pt(lm_list, 12)
    ring_tip = _pt(lm_list, 16)
    pinky_tip = _pt(lm_list, 20)
    index_mcp = _pt(lm_list, 5)
    middle_mcp = _pt(lm_list, 9)
    ring_mcp = _pt(lm_list, 13)
    pinky_mcp = _pt(lm_list, 17)
    thumb_ext_ratio = _ext_ratio(lm_list, 4, 3, 2)
    index_ext_ratio = _ext_ratio(lm_list, 8, 6, 5)
    middle_ext_ratio = _ext_ratio(lm_list, 12, 10, 9)
    ring_ext_ratio = _ext_ratio(lm_list, 16, 14, 13)
    pinky_ext_ratio = _ext_ratio(lm_list, 20, 18, 17)

    palm_scale = max(1.0, _dist(wrist, index_mcp) + _dist(wrist, pinky_mcp))
    pinch_ratio = _dist(thumb_tip, index_tip) / palm_scale
    two_finger_ratio = _dist(index_tip, middle_tip) / palm_scale

    ext_num = (
        _dist(wrist, index_tip)
        + _dist(wrist, middle_tip)
        + _dist(wrist, ring_tip)
        + _dist(wrist, pinky_tip)
    )
    ext_den = (
        _dist(wrist, index_mcp)
        + _dist(wrist, middle_mcp)
        + _dist(wrist, ring_mcp)
        + _dist(wrist, pinky_mcp)
    )
    open_ratio = ext_num / max(1.0, ext_den)
    fist_ratio = 1.0 / max(0.001, open_ratio)
    pinch_delta = 0.0
    if prev_features is not None:
        pinch_delta = pinch_ratio - prev_features.pinch_ratio

    return GestureFeatures(
        valid=True,
        pinch_ratio=pinch_ratio,
        open_ratio=open_ratio,
        fist_ratio=fist_ratio,
        two_finger_ratio=two_finger_ratio,
        palm_scale_px=palm_scale,
        index_tip_y_px=float(index_tip[1]),
        pinch_delta=pinch_delta,
        thumb_ext_ratio=thumb_ext_ratio,
        index_ext_ratio=index_ext_ratio,
        middle_ext_ratio=middle_ext_ratio,
        ring_ext_ratio=ring_ext_ratio,
        pinky_ext_ratio=pinky_ext_ratio,
    )


def _ratio_conf_above(value: float, threshold: float, span: float = 0.2) -> float:
    return _clamp((value - threshold) / max(1e-6, span), 0.0, 1.0)


def _ratio_conf_below(value: float, threshold: float, span: float = 0.2) -> float:
    return _clamp((threshold - value) / max(1e-6, span), 0.0, 1.0)


def predict_gesture(features: GestureFeatures, profile: UserProfile) -> GesturePrediction:
    debug = GestureDebugInfo(
        input_valid=features.valid,
        open_fist_margin_required=profile.open_fist_margin,
        open_tap_margin_required=profile.open_tap_margin,
        fist_pinch_margin_required=profile.fist_pinch_margin,
        pinch_fist_margin_required=profile.pinch_fist_margin,
        two_finger_vs_pinch_ratio_required=profile.two_finger_vs_pinch_ratio,
    )
    if not features.valid:
        return GesturePrediction("unknown", 0.0, features, debug=debug)

    index_ext_conf = _ratio_conf_above(features.index_ext_ratio, 1.35, 0.28)
    index_fold_conf = _ratio_conf_below(features.index_ext_ratio, 1.24, 0.24)
    middle_ext_conf = _ratio_conf_above(features.middle_ext_ratio, 1.35, 0.28)
    middle_fold_conf = _ratio_conf_below(features.middle_ext_ratio, 1.24, 0.24)
    ring_ext_conf = _ratio_conf_above(features.ring_ext_ratio, 1.30, 0.26)
    ring_fold_conf = _ratio_conf_below(features.ring_ext_ratio, 1.25, 0.25)
    pinky_ext_conf = _ratio_conf_above(features.pinky_ext_ratio, 1.30, 0.26)
    pinky_fold_conf = _ratio_conf_below(features.pinky_ext_ratio, 1.25, 0.25)
    thumb_open_conf = _ratio_conf_above(features.thumb_ext_ratio, 1.20, 0.35)
    thumb_fold_conf = _ratio_conf_below(features.thumb_ext_ratio, 1.10, 0.30)

    # Allow a modest tolerance above calibrated "touching" to account for
    # fingertip length/asymmetry while keeping tap selective.
    two_finger_close_conf = _ratio_conf_below(
        features.two_finger_ratio,
        profile.two_finger_tap_max + 0.025,
        0.10,
    )
    pinch_base_conf = _ratio_conf_below(features.pinch_ratio, profile.pinch_max, 0.18)
    pinch_sep_conf = _ratio_conf_above(
        features.two_finger_ratio,
        profile.two_finger_tap_max * 1.18,
        0.12,
    )
    open_spread_conf = _ratio_conf_above(
        features.two_finger_ratio,
        profile.open_two_finger_min,
        0.12,
    )
    fist_non_pinch_conf = _ratio_conf_above(
        features.pinch_ratio,
        profile.fist_pinch_min,
        0.16,
    )
    open_conf = _ratio_conf_above(features.open_ratio, profile.open_min, 0.35)
    fist_conf = _ratio_conf_below(features.open_ratio, profile.fist_max, 0.22)
    pinch_not_fist_conf = _clamp(1.0 - fist_conf, 0.0, 1.0)
    tap_shape_open_conf = _ratio_conf_below(features.open_ratio, profile.open_min + 0.03, 0.35)
    pinch_open_conf = _ratio_conf_above(features.pinch_ratio, 0.20, 0.18)

    open_full_conf = (
        0.42 * open_conf
        + 0.16 * index_ext_conf
        + 0.16 * middle_ext_conf
        + 0.09 * ring_ext_conf
        + 0.09 * pinky_ext_conf
        + 0.08 * thumb_open_conf
    )
    open_full_conf = _clamp(open_full_conf - 0.18 * (1.0 - open_spread_conf), 0.0, 1.0)
    fist_full_conf = (
        0.30 * fist_conf
        + 0.30 * index_fold_conf
        + 0.24 * middle_fold_conf
        + 0.08 * ring_fold_conf
        + 0.04 * pinky_fold_conf
        + 0.04 * thumb_fold_conf
    )
    fist_full_conf = _clamp(fist_full_conf, 0.0, 1.0)
    pinch_conf = (
        0.66 * pinch_base_conf
        + 0.18 * index_ext_conf
        + 0.10 * pinch_sep_conf
        + 0.06 * pinch_not_fist_conf
    )
    two_finger_conf = (
        0.56 * two_finger_close_conf
        + 0.22 * min(index_ext_conf, middle_ext_conf)
        + 0.10 * min(ring_fold_conf, pinky_fold_conf)
        + 0.07 * pinch_open_conf
        + 0.05 * tap_shape_open_conf
    )
    debug.index_ext_conf = index_ext_conf
    debug.index_fold_conf = index_fold_conf
    debug.middle_ext_conf = middle_ext_conf
    debug.middle_fold_conf = middle_fold_conf
    debug.ring_ext_conf = ring_ext_conf
    debug.ring_fold_conf = ring_fold_conf
    debug.pinky_ext_conf = pinky_ext_conf
    debug.pinky_fold_conf = pinky_fold_conf
    debug.thumb_open_conf = thumb_open_conf
    debug.thumb_fold_conf = thumb_fold_conf
    debug.open_spread_conf = open_spread_conf
    debug.fist_non_pinch_conf = fist_non_pinch_conf
    debug.pinch_not_fist_conf = pinch_not_fist_conf
    debug.two_finger_close_conf = two_finger_close_conf
    debug.pinch_base_conf = pinch_base_conf
    debug.pinch_sep_conf = pinch_sep_conf
    debug.open_conf = open_conf
    debug.fist_conf = fist_conf
    debug.open_full_conf = open_full_conf
    debug.fist_full_conf = fist_full_conf
    debug.pinch_conf = pinch_conf
    debug.two_finger_conf = two_finger_conf

    open_ok = (
        open_full_conf >= debug.open_required
        and open_full_conf > fist_full_conf + profile.open_fist_margin
        and open_full_conf > two_finger_conf + profile.open_tap_margin
    )
    fist_ok = (
        fist_full_conf >= debug.fist_required
        and fist_full_conf > open_full_conf + profile.open_fist_margin
        and fist_full_conf > pinch_conf + profile.fist_pinch_margin
    )
    two_ok = (
        two_finger_conf >= debug.two_finger_required
        and two_finger_conf >= pinch_conf * profile.two_finger_vs_pinch_ratio
    )
    pinch_ok = (
        pinch_conf >= debug.pinch_required
        and pinch_conf > fist_full_conf + profile.pinch_fist_margin
    )

    candidates: list[tuple[str, float]] = []
    if open_ok:
        candidates.append(("open_hand", open_full_conf))
    if fist_ok:
        candidates.append(("fist", fist_full_conf))
    if two_ok:
        candidates.append(("two_finger_tap", two_finger_conf))
    if pinch_ok:
        candidates.append(("pinch_control", pinch_conf))
    if candidates:
        label, conf = max(candidates, key=lambda item: item[1])
        return GesturePrediction(label, conf, features, debug=debug)

    best_conf = max(two_finger_conf, pinch_conf, open_full_conf, fist_full_conf)
    return GesturePrediction("unknown", best_conf, features, debug=debug)

