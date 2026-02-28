from __future__ import annotations

import time
from pathlib import Path

import cv2

from cv_practice.assistive.actions import ActionExecutor
from cv_practice.assistive.calibration import load_profile, run_calibration, save_profile
from cv_practice.assistive.config import load_assistive_config
from cv_practice.assistive.detector import HandDetector
from cv_practice.assistive.inference import extract_features, predict_gesture
from cv_practice.assistive.recording import GestureRecorder
from cv_practice.assistive.state_machine import GestureStateMachine
from cv_practice.assistive.telemetry import TelemetryLogger
from cv_practice.assistive.types import GestureFeatures


def _draw_hud(
    frame,
    *,
    fps: float,
    prediction_label: str,
    confidence: float,
    locked: bool,
    last_event: str,
    recording: bool,
    rec_label: str,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (24, 24), (w - 24, 180), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (24, 24), (w - 24, 180), (240, 240, 240), 2)

    state_color = (0, 220, 110) if not locked else (0, 80, 255)
    cv2.putText(
        frame,
        f"AGCP {'LOCKED' if locked else 'UNLOCKED'}",
        (45, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        state_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Gesture: {prediction_label:>16}  conf={confidence:0.2f}",
        (45, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Last command: {last_event}  |  FPS: {fps:0.1f}",
        (45, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    rec_text = f"REC {'ON' if recording else 'OFF'} ({rec_label})"
    rec_color = (50, 60, 255) if recording else (170, 170, 170)
    cv2.putText(
        frame,
        rec_text,
        (45, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        rec_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Keys: Q quit | C calibrate | R record | 1-5 labels (idle/open/fist/pinch/tap)",
        (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def _open_camera(cfg):
    cap = cv2.VideoCapture(cfg.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    return cap


def run_assistive_app(config_path: str | None = None) -> None:
    cfg = load_assistive_config(config_path)
    profile = load_profile(cfg.profile_path)
    profile.model_path = cfg.model_path
    machine = GestureStateMachine(profile)
    executor = ActionExecutor()
    telemetry = TelemetryLogger(cfg.output_dir)
    recorder = GestureRecorder(Path(cfg.output_dir) / "recordings")

    cap = _open_camera(cfg)
    detector = HandDetector(model_path=profile.model_path, num_hands=1)

    prev_t = time.perf_counter()
    fps_ema = 0.0
    fps_alpha = 0.1
    prev_features: GestureFeatures | None = None
    last_event = "none"
    label_map = {
        ord("1"): "idle",
        ord("2"): "open_hand",
        ord("3"): "fist",
        ord("4"): "pinch_control",
        ord("5"): "two_finger_tap",
    }

    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame, _, lm_list = detector.find_hands(frame, hand_num=0, flip=True)
            features = extract_features(lm_list, profile, prev_features)
            prediction = predict_gesture(features, profile)
            prev_features = features
            now_ms = int(time.perf_counter() * 1000)
            events = machine.update(prediction, now_ms)

            for event in events:
                executed = executor.execute(event)
                telemetry.log_command(event, executed=executed)
                recorder.add_command(event.command)
                last_event = f"{event.command} ({'ok' if executed else 'fail'})"

            decision_ms = (time.perf_counter() - frame_start) * 1000.0
            telemetry.log_frame(decision_ms, prediction)
            recorder.add_frame(prediction, lm_list)

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = (
                inst_fps
                if fps_ema == 0.0
                else (1 - fps_alpha) * fps_ema + fps_alpha * inst_fps
            )

            _draw_hud(
                frame,
                fps=fps_ema,
                prediction_label=prediction.label,
                confidence=prediction.confidence,
                locked=machine.locked,
                last_event=last_event,
                recording=recorder.is_recording,
                rec_label=recorder.active_label,
            )
            cv2.imshow("Assistive Gesture Control Platform", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if recorder.is_recording:
                    recorder.stop()
                else:
                    recorder.start("idle")
            if key == ord("c"):
                was_recording = recorder.is_recording
                if was_recording:
                    recorder.stop()

                detector.close()
                cap.release()
                cv2.destroyAllWindows()
                try:
                    new_profile = run_calibration(cfg.camera_id, profile.model_path)
                    sample_count = float(new_profile.extra.get("calibration_samples", 0.0))
                    if sample_count > 0:
                        new_profile.name = profile.name
                        new_profile.model_path = profile.model_path
                        profile = new_profile
                        save_profile(profile, cfg.profile_path)
                        machine = GestureStateMachine(profile)
                        prev_features = None
                        last_event = "profile_recalibrated"
                    else:
                        last_event = "calibration_cancelled_or_empty"
                except Exception as exc:
                    last_event = f"calibration_failed ({exc.__class__.__name__})"
                finally:
                    cap = _open_camera(cfg)
                    detector = HandDetector(model_path=profile.model_path, num_hands=1)
                    prev_t = time.perf_counter()
                    if was_recording:
                        recorder.start(recorder.active_label)
            if key in label_map and recorder.is_recording:
                recorder.set_label(label_map[key])

    finally:
        detector.close()
        cap.release()
        recorder.stop()
        summary = telemetry.finalize()
        cv2.destroyAllWindows()
    print(
        "AGCP session summary: "
        f"frames={summary.frames}, commands={summary.commands}, "
        f"fps_avg={summary.fps_avg:.1f}, latency_p95_ms={summary.latency_p95_ms:.1f}"
    )
