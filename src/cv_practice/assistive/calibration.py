from __future__ import annotations

from copy import deepcopy
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from cv_practice.assistive.detector import HandDetector
from cv_practice.assistive.inference import extract_features
from cv_practice.assistive.types import UserProfile


def default_profile(name: str = "default") -> UserProfile:
    return UserProfile(name=name)


def _profile_from_dict(payload: dict) -> UserProfile:
    known = {k: v for k, v in payload.items() if k in UserProfile.__dataclass_fields__}
    profile = UserProfile(**known)
    if "extra" not in payload:
        profile.extra = {}
    return profile


def load_profile(path: str | Path) -> UserProfile:
    p = Path(path)
    if not p.exists():
        return default_profile(p.stem)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _profile_from_dict(payload)


def save_profile(profile: UserProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2)


def _median_or(default: float, samples: list[float]) -> float:
    if not samples:
        return default
    return float(statistics.median(samples))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _has_samples(samples: list[float], min_count: int = 1) -> bool:
    return len(samples) >= min_count


def _robust_median(default: float, samples: list[float]) -> float:
    if not samples:
        return default
    raw_median = float(statistics.median(samples))
    if len(samples) < 5:
        return raw_median

    abs_dev = [abs(v - raw_median) for v in samples]
    mad = float(statistics.median(abs_dev))
    if mad < 1e-6:
        return raw_median

    filtered = [v for v in samples if abs(v - raw_median) <= 2.8 * mad]
    if len(filtered) < max(3, len(samples) // 3):
        return raw_median
    return float(statistics.median(filtered))


def _blend(prior: float, candidate: float, weight: float) -> float:
    w = _clamp(weight, 0.0, 1.0)
    return (1.0 - w) * prior + w * candidate


def _sample_count(samples: dict[str, dict[str, list[float]]], key: str) -> int:
    return len(samples[key]["open_ratio"])


def _derive_profile_from_samples(
    prior: UserProfile,
    samples: dict[str, dict[str, list[float]]],
    *,
    min_samples_per_pose: int = 16,
) -> UserProfile:
    profile = deepcopy(prior)

    open_samples = samples["open_hand"]["open_ratio"]
    fist_samples = samples["fist"]["open_ratio"]
    pinch_samples = samples["pinch_control"]["pinch_ratio"]
    tap_samples = samples["two_finger_tap"]["two_finger_ratio"]
    open_two_samples = samples["open_hand"]["two_finger_ratio"]
    fist_pinch_samples = samples["fist"]["pinch_ratio"]

    open_median = _robust_median(prior.open_min, open_samples)
    fist_median = _robust_median(prior.fist_max, fist_samples)
    pinch_median = _robust_median(prior.pinch_max, pinch_samples)
    tap_median = _robust_median(prior.two_finger_tap_max, tap_samples)
    open_two_median = _robust_median(prior.open_two_finger_min, open_two_samples)
    fist_pinch_median = _robust_median(prior.fist_pinch_min, fist_pinch_samples)

    primary_weight = 0.42
    separation_weight = 0.35

    if _has_samples(open_samples, min_samples_per_pose):
        open_candidate = _clamp(open_median * 0.90, 1.25, 1.75)
        profile.open_min = _blend(prior.open_min, open_candidate, primary_weight)

    if _has_samples(fist_samples, min_samples_per_pose):
        fist_candidate = _clamp(fist_median * 1.10, 0.82, 1.12)
        profile.fist_max = _blend(prior.fist_max, fist_candidate, primary_weight)

    if _has_samples(pinch_samples, min_samples_per_pose):
        pinch_candidate = _clamp(pinch_median * 1.18, max(prior.pinch_min + 0.03, 0.10), 0.45)
        profile.pinch_max = _blend(prior.pinch_max, pinch_candidate, primary_weight)

    if _has_samples(tap_samples, min_samples_per_pose):
        tap_candidate = _clamp(tap_median * 1.12, 0.10, 0.24)
        profile.two_finger_tap_max = _blend(prior.two_finger_tap_max, tap_candidate, primary_weight)

    if _has_samples(open_two_samples, min_samples_per_pose) and _has_samples(
        tap_samples, min_samples_per_pose
    ):
        open_sep = (open_two_median + tap_median) / 2.0
        open_two_candidate = _clamp(
            max(profile.two_finger_tap_max + 0.03, open_sep),
            0.14,
            0.34,
        )
        profile.open_two_finger_min = _blend(
            prior.open_two_finger_min,
            open_two_candidate,
            separation_weight,
        )

    if _has_samples(fist_pinch_samples, min_samples_per_pose) and _has_samples(
        pinch_samples, min_samples_per_pose
    ):
        pinch_sep = (fist_pinch_median + pinch_median) / 2.0
        fist_pinch_candidate = _clamp(
            max(profile.pinch_max + 0.02, pinch_sep),
            0.10,
            0.50,
        )
        profile.fist_pinch_min = _blend(
            prior.fist_pinch_min,
            fist_pinch_candidate,
            separation_weight,
        )

    profile.open_min = _clamp(profile.open_min, 1.25, 1.75)
    profile.fist_max = _clamp(profile.fist_max, 0.82, 1.12)
    if profile.fist_max >= profile.open_min - 0.08:
        profile.fist_max = max(0.82, profile.open_min - 0.10)

    profile.pinch_max = _clamp(profile.pinch_max, max(profile.pinch_min + 0.03, 0.10), 0.45)
    profile.two_finger_tap_max = _clamp(profile.two_finger_tap_max, 0.10, 0.24)
    profile.open_two_finger_min = _clamp(
        max(profile.open_two_finger_min, profile.two_finger_tap_max + 0.03),
        0.14,
        0.34,
    )
    profile.fist_pinch_min = _clamp(
        max(profile.fist_pinch_min, profile.pinch_max + 0.02),
        0.10,
        0.50,
    )

    lock_hi = max(profile.pinch_min + 0.03, profile.fist_pinch_min - 0.005)
    profile.lock_pinch_guard_ratio = _clamp(
        min(profile.fist_pinch_min - 0.01, profile.pinch_max + 0.03),
        profile.pinch_min + 0.02,
        lock_hi,
    )
    profile.lock_pinch_guard_conf = _clamp(prior.lock_pinch_guard_conf, 0.50, 0.75)

    total_samples = sum(_sample_count(samples, key) for key in samples.keys())
    profile.extra["calibration_samples"] = float(total_samples)
    profile.extra["open_pose_samples"] = float(_sample_count(samples, "open_hand"))
    profile.extra["fist_pose_samples"] = float(_sample_count(samples, "fist"))
    profile.extra["pinch_pose_samples"] = float(_sample_count(samples, "pinch_control"))
    profile.extra["tap_pose_samples"] = float(_sample_count(samples, "two_finger_tap"))
    profile.extra["open_two_finger_median"] = float(open_two_median)
    profile.extra["fist_pinch_median"] = float(fist_pinch_median)
    return profile


def run_calibration(
    camera_id: int = 0,
    model_path: str = "models/hand_landmarker.task",
    base_profile: UserProfile | None = None,
) -> UserProfile:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for calibration.")

    prompts = [
        ("open_hand", "Open hand and hold steady. Press SPACE to capture."),
        ("fist", "Close fist and hold steady. Press SPACE to capture."),
        ("pinch_control", "Pinch thumb and index. Press SPACE to capture."),
        ("two_finger_tap", "Extend index+middle together. Press SPACE to capture."),
    ]
    samples: dict[str, dict[str, list[float]]] = {
        k: {"open_ratio": [], "pinch_ratio": [], "two_finger_ratio": []}
        for k, _ in prompts
    }
    prompt_idx = 0
    collecting_warmup_until = 0.0
    collecting_until = 0.0
    collecting_key: str | None = None
    prior_profile = deepcopy(base_profile) if base_profile is not None else default_profile("user")

    with HandDetector(model_path=model_path, num_hands=1) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, _, lm_list = detector.find_hands(frame, hand_num=0, flip=True)
            feat = extract_features(lm_list, default_profile("calib"))

            if prompt_idx >= len(prompts):
                break

            key, text = prompts[prompt_idx]
            now = time.perf_counter()

            if (
                collecting_key == key
                and collecting_until > now
                and now >= collecting_warmup_until
                and feat.valid
            ):
                samples[key]["open_ratio"].append(float(feat.open_ratio))
                samples[key]["pinch_ratio"].append(float(feat.pinch_ratio))
                samples[key]["two_finger_ratio"].append(float(feat.two_finger_ratio))
            elif collecting_key == key and collecting_until > 0.0 and now >= collecting_until:
                collecting_key = None
                collecting_warmup_until = 0.0
                collecting_until = 0.0
                prompt_idx += 1
                continue

            cv2.rectangle(frame, (20, 20), (1240, 110), (15, 15, 15), -1)
            cv2.putText(frame, f"Calibration {prompt_idx + 1}/{len(prompts)}", (40, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (220, 220, 220), 2, cv2.LINE_AA)
            if collecting_key == key and collecting_until > now:
                remaining_s = max(0.0, collecting_until - now)
                count = _sample_count(samples, key)
                cv2.putText(
                    frame,
                    f"Capturing... {remaining_s:0.1f}s  n={count}",
                    (790, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (120, 250, 120),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(frame, "Q to cancel", (1080, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("AGCP Calibration", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" ") and collecting_key is None:
                collecting_warmup_until = time.perf_counter() + 0.25
                collecting_until = time.perf_counter() + 1.55
                collecting_key = key

    cap.release()
    cv2.destroyAllWindows()

    return _derive_profile_from_samples(prior_profile, samples)
