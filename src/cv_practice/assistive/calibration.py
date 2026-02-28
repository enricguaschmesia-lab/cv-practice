from __future__ import annotations

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


def run_calibration(
    camera_id: int = 0,
    model_path: str = "models/hand_landmarker.task",
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
    samples: dict[str, list[float]] = {k: [] for k, _ in prompts}
    prompt_idx = 0
    collecting_until = 0.0

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

            if collecting_until > now and feat.valid:
                if key == "open_hand":
                    samples[key].append(feat.open_ratio)
                elif key == "fist":
                    # Fist threshold uses open_ratio upper bound in inference.
                    samples[key].append(feat.open_ratio)
                elif key == "pinch_control":
                    samples[key].append(feat.pinch_ratio)
                elif key == "two_finger_tap":
                    samples[key].append(feat.two_finger_ratio)

            cv2.rectangle(frame, (20, 20), (1240, 110), (15, 15, 15), -1)
            cv2.putText(frame, f"Calibration {prompt_idx + 1}/{len(prompts)}", (40, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q to cancel", (1080, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow("AGCP Calibration", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                collecting_until = time.perf_counter() + 1.2
                time.sleep(1.25)
                prompt_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    profile = default_profile("user")
    open_median = _median_or(profile.open_min, samples["open_hand"])
    fist_median = _median_or(profile.fist_max, samples["fist"])
    pinch_median = _median_or(profile.pinch_max, samples["pinch_control"])
    two_finger_median = _median_or(profile.two_finger_tap_max, samples["two_finger_tap"])

    profile.open_min = max(1.05, open_median * 0.92)
    profile.fist_max = min(1.10, fist_median * 1.12)
    profile.pinch_max = min(0.55, max(profile.pinch_min + 0.04, pinch_median * 1.18))
    profile.two_finger_tap_max = min(0.35, two_finger_median * 1.12)
    if profile.fist_max >= profile.open_min:
        profile.fist_max = max(0.85, profile.open_min - 0.12)
    profile.extra["calibration_samples"] = float(sum(len(v) for v in samples.values()))
    return profile
