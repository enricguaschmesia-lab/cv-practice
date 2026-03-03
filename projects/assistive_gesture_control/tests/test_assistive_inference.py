from __future__ import annotations

from agcp.inference import extract_features, predict_gesture
from agcp.types import GestureFeatures, UserProfile


def _make_lm(scale: int = 1) -> list[list[int]]:
    pts = {
        0: (100, 200),
        4: (120, 150),
        5: (140, 180),
        8: (150, 130),
        9: (165, 180),
        12: (170, 130),
        13: (185, 182),
        16: (190, 132),
        17: (205, 185),
        20: (210, 140),
    }
    lm = []
    for idx in range(21):
        x, y = pts.get(idx, (100 + idx * 3, 180 + idx))
        lm.append([idx, x * scale, y * scale])
    return lm


def test_extract_features_scale_stable_ratios() -> None:
    profile = UserProfile(name="t")
    f1 = extract_features(_make_lm(1), profile)
    f2 = extract_features(_make_lm(2), profile)

    assert f1.valid is True
    assert f2.valid is True
    assert abs(f1.pinch_ratio - f2.pinch_ratio) < 0.02
    assert abs(f1.open_ratio - f2.open_ratio) < 0.02
    assert abs(f1.two_finger_ratio - f2.two_finger_ratio) < 0.02


def test_predict_gesture_labels() -> None:
    p = UserProfile(name="t")
    open_feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.45,
        open_ratio=1.58,
        fist_ratio=0.63,
        two_finger_ratio=0.33,
        palm_scale_px=140.0,
        index_tip_y_px=130.0,
        thumb_ext_ratio=1.35,
        index_ext_ratio=1.70,
        middle_ext_ratio=1.72,
        ring_ext_ratio=1.61,
        pinky_ext_ratio=1.55,
    )
    fist_feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.55,
        open_ratio=0.73,
        fist_ratio=1.37,
        two_finger_ratio=0.40,
        palm_scale_px=140.0,
        index_tip_y_px=220.0,
        thumb_ext_ratio=0.95,
        index_ext_ratio=1.08,
        middle_ext_ratio=1.06,
        ring_ext_ratio=1.04,
        pinky_ext_ratio=1.03,
    )
    pinch_feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.14,
        open_ratio=1.12,
        fist_ratio=0.89,
        two_finger_ratio=0.23,
        palm_scale_px=140.0,
        index_tip_y_px=170.0,
        thumb_ext_ratio=1.10,
        index_ext_ratio=1.42,
        middle_ext_ratio=1.18,
        ring_ext_ratio=1.14,
        pinky_ext_ratio=1.12,
    )
    tap_feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.32,
        open_ratio=1.17,
        fist_ratio=0.85,
        two_finger_ratio=0.11,
        palm_scale_px=140.0,
        index_tip_y_px=155.0,
        thumb_ext_ratio=1.02,
        index_ext_ratio=1.66,
        middle_ext_ratio=1.61,
        ring_ext_ratio=1.07,
        pinky_ext_ratio=1.05,
    )

    assert predict_gesture(open_feat, p).label == "open_hand"
    assert predict_gesture(fist_feat, p).label == "fist"
    assert predict_gesture(pinch_feat, p).label == "pinch_control"
    assert predict_gesture(tap_feat, p).label == "two_finger_tap"


def test_two_finger_tap_not_overridden_by_open_hand() -> None:
    p = UserProfile(name="t")
    feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.31,
        open_ratio=1.42,
        fist_ratio=0.70,
        two_finger_ratio=0.10,
        palm_scale_px=140.0,
        index_tip_y_px=145.0,
        thumb_ext_ratio=1.06,
        index_ext_ratio=1.64,
        middle_ext_ratio=1.58,
        ring_ext_ratio=1.08,
        pinky_ext_ratio=1.05,
    )
    pred = predict_gesture(feat, p)
    assert pred.label == "two_finger_tap"
    assert pred.debug is not None
    assert pred.debug.two_finger_conf > pred.debug.open_full_conf


def test_two_finger_close_conf_allows_small_separation_tolerance() -> None:
    p = UserProfile(name="t", two_finger_tap_max=0.13)
    feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.31,
        open_ratio=1.17,
        fist_ratio=0.85,
        two_finger_ratio=0.148,  # Above prior tolerance (+0.015), within new one (+0.025).
        palm_scale_px=140.0,
        index_tip_y_px=155.0,
        thumb_ext_ratio=1.02,
        index_ext_ratio=1.66,
        middle_ext_ratio=1.61,
        ring_ext_ratio=1.07,
        pinky_ext_ratio=1.05,
    )
    pred = predict_gesture(feat, p)
    assert pred.debug is not None
    assert pred.debug.two_finger_close_conf > 0.0


def test_pinch_not_classified_as_fist_when_thumb_index_are_close() -> None:
    p = UserProfile(name="t")
    feat = GestureFeatures(
        valid=True,
        pinch_ratio=0.13,
        open_ratio=0.86,
        fist_ratio=1.16,
        two_finger_ratio=0.24,
        palm_scale_px=140.0,
        index_tip_y_px=185.0,
        thumb_ext_ratio=1.02,
        index_ext_ratio=1.44,
        middle_ext_ratio=1.18,
        ring_ext_ratio=1.08,
        pinky_ext_ratio=1.07,
    )
    pred = predict_gesture(feat, p)
    assert pred.label != "fist"

