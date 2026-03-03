from __future__ import annotations

import time
from pathlib import Path

import cv2

from cv_practice.assistive.actions import ActionExecutor
from cv_practice.assistive.calibration import (
    default_profile,
    load_profile,
    run_calibration,
    save_profile,
)
from cv_practice.assistive.config import load_assistive_config
from cv_practice.assistive.detector import HandDetector
from cv_practice.assistive.inference import extract_features, predict_gesture
from cv_practice.assistive.recording import GestureRecorder
from cv_practice.assistive.state_machine import GestureStateMachine
from cv_practice.assistive.telemetry import TelemetryLogger
from cv_practice.assistive.types import GestureFeatures

_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def _draw_hand_debug(frame, lm_list: list[list[int]]) -> None:
    if len(lm_list) < 21:
        return

    pts: dict[int, tuple[int, int]] = {}
    for idx, x, y in lm_list:
        pts[idx] = (x, y)

    for start, end in _HAND_CONNECTIONS:
        if start in pts and end in pts:
            cv2.line(frame, pts[start], pts[end], (80, 210, 255), 2, cv2.LINE_AA)

    highlight = {0, 4, 8, 12, 16, 20}
    for idx, x, y in lm_list:
        r = 5 if idx in highlight else 3
        color = (40, 255, 120) if idx in highlight else (220, 220, 220)
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        if idx in highlight:
            cv2.putText(
                frame,
                str(idx),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


def _draw_debug_panel(frame, prediction, profile, machine_debug: dict[str, float | int | bool | str | None]) -> None:
    h, w = frame.shape[:2]
    panel_w = min(470, max(320, int(w * 0.37)))
    x0 = w - panel_w - 18
    y0 = 196
    x1 = w - 18
    y1 = h - 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (235, 235, 235), 1)

    feat = prediction.features
    dbg = prediction.debug
    hold_elapsed = int(machine_debug.get("hold_elapsed_ms", 0))
    hold_required = int(machine_debug.get("hold_required_ms", profile.hold_ms))
    tap_required = int(machine_debug.get("tap_hold_required_ms", max(120, profile.hold_ms // 2)))
    pinch_baseline = machine_debug.get("pinch_baseline")

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("DEBUG VIEW", (255, 255, 255)),
        (
            f"selected={prediction.label} conf={prediction.confidence:.2f}",
            (235, 235, 235),
        ),
        (
            f"locked={machine_debug.get('locked')} active={machine_debug.get('active_label')}",
            (225, 225, 225),
        ),
        (f"hold={hold_elapsed}/{hold_required}ms  tap_hold={tap_required}ms", (215, 215, 215)),
        (
            f"pinch_baseline={'None' if pinch_baseline is None else f'{float(pinch_baseline):.3f}'}",
            (215, 215, 215),
        ),
        ("", (215, 215, 215)),
        ("Measures (live)", (255, 255, 255)),
        (f"palm_scale_px: {feat.palm_scale_px:.2f}", (230, 230, 230)),
        (
            f"open_ratio: {feat.open_ratio:.3f} (>= {profile.open_min:.3f})",
            (230, 230, 230),
        ),
        (
            f"pinch_ratio: {feat.pinch_ratio:.3f} (<= {profile.pinch_max:.3f})",
            (230, 230, 230),
        ),
        (
            f"fist_non_pinch: pinch_ratio >= {profile.fist_pinch_min:.3f}",
            (230, 230, 230),
        ),
        (
            f"two_finger_ratio: {feat.two_finger_ratio:.3f} (<= {profile.two_finger_tap_max:.3f})",
            (230, 230, 230),
        ),
        (
            f"open_spread: two_finger_ratio >= {profile.open_two_finger_min:.3f}",
            (230, 230, 230),
        ),
        (f"pinch_range: [{profile.pinch_min:.3f}, {profile.pinch_max:.3f}]", (230, 230, 230)),
        (
            f"ext thumb/index/mid: {feat.thumb_ext_ratio:.2f} {feat.index_ext_ratio:.2f} {feat.middle_ext_ratio:.2f}",
            (230, 230, 230),
        ),
        (
            f"ext ring/pinky: {feat.ring_ext_ratio:.2f} {feat.pinky_ext_ratio:.2f}",
            (230, 230, 230),
        ),
        (f"pinch_delta: {feat.pinch_delta:+.4f}", (230, 230, 230)),
    ]

    if dbg is not None:
        open_ok = dbg.open_full_conf >= dbg.open_required
        fist_ok = dbg.fist_full_conf >= dbg.fist_required
        pinch_ok = dbg.pinch_conf >= dbg.pinch_required
        tap_ok = dbg.two_finger_conf >= dbg.two_finger_required
        open_margin = dbg.open_full_conf - dbg.fist_full_conf
        fist_margin = dbg.fist_full_conf - dbg.open_full_conf
        tap_vs_pinch = dbg.two_finger_conf / max(1e-6, dbg.pinch_conf)
        lines.extend(
            [
                ("", (215, 215, 215)),
                ("Gesture confidence (live / required)", (255, 255, 255)),
                (
                    f"open_hand: {dbg.open_full_conf:.2f} / {dbg.open_required:.2f}",
                    (95, 235, 120) if open_ok else (235, 220, 95),
                ),
                (
                    f"fist: {dbg.fist_full_conf:.2f} / {dbg.fist_required:.2f}",
                    (95, 235, 120) if fist_ok else (235, 220, 95),
                ),
                (
                    f"pinch_control: {dbg.pinch_conf:.2f} / {dbg.pinch_required:.2f}",
                    (95, 235, 120) if pinch_ok else (235, 220, 95),
                ),
                (
                    f"two_finger_tap: {dbg.two_finger_conf:.2f} / {dbg.two_finger_required:.2f}",
                    (95, 235, 120) if tap_ok else (235, 220, 95),
                ),
                (
                    f"open-vs-fist margin: {open_margin:+.2f} (need > +{dbg.open_fist_margin_required:.2f})",
                    (230, 230, 230),
                ),
                (
                    (
                        "open-vs-tap margin: "
                        f"{(dbg.open_full_conf - dbg.two_finger_conf):+.2f} "
                        f"(need > +{dbg.open_tap_margin_required:.2f})"
                    ),
                    (230, 230, 230),
                ),
                (
                    f"fist-vs-open margin: {fist_margin:+.2f} (need > +{dbg.open_fist_margin_required:.2f})",
                    (230, 230, 230),
                ),
                (
                    (
                        "fist-vs-pinch margin: "
                        f"{(dbg.fist_full_conf - dbg.pinch_conf):+.2f} "
                        f"(need > +{dbg.fist_pinch_margin_required:.2f})"
                    ),
                    (230, 230, 230),
                ),
                (
                    (
                        "pinch-vs-fist margin: "
                        f"{(dbg.pinch_conf - dbg.fist_full_conf):+.2f} "
                        f"(need > +{dbg.pinch_fist_margin_required:.2f})"
                    ),
                    (230, 230, 230),
                ),
                (
                    f"tap/pinch ratio: {tap_vs_pinch:.2f} (need >= {dbg.two_finger_vs_pinch_ratio_required:.2f})",
                    (230, 230, 230),
                ),
                ("", (215, 215, 215)),
                ("Component confidence", (255, 255, 255)),
                (f"index_ext={dbg.index_ext_conf:.2f}  middle_ext={dbg.middle_ext_conf:.2f}", (225, 225, 225)),
                (f"index_fold={dbg.index_fold_conf:.2f}  middle_fold={dbg.middle_fold_conf:.2f}", (225, 225, 225)),
                (f"ring_ext={dbg.ring_ext_conf:.2f}  pinky_ext={dbg.pinky_ext_conf:.2f}", (225, 225, 225)),
                (f"ring_fold={dbg.ring_fold_conf:.2f}  pinky_fold={dbg.pinky_fold_conf:.2f}", (225, 225, 225)),
                (f"thumb_open={dbg.thumb_open_conf:.2f}  thumb_fold={dbg.thumb_fold_conf:.2f}", (225, 225, 225)),
                (
                    f"open_spread={dbg.open_spread_conf:.2f}  fist_non_pinch={dbg.fist_non_pinch_conf:.2f}",
                    (225, 225, 225),
                ),
                (f"pinch_not_fist={dbg.pinch_not_fist_conf:.2f}", (225, 225, 225)),
                (
                    f"two_finger_close={dbg.two_finger_close_conf:.2f}  pinch_base={dbg.pinch_base_conf:.2f}",
                    (225, 225, 225),
                ),
                (f"pinch_sep={dbg.pinch_sep_conf:.2f}", (225, 225, 225)),
            ]
        )

    lines.extend(
        [
            ("", (215, 215, 215)),
            ("Action gates", (255, 255, 255)),
            (
                f"unlock/lock conf>=0.72 hold>={profile.hold_ms}ms",
                (220, 220, 220),
            ),
            (
                (
                    "lock guards: "
                    f"pinch_ratio>{profile.lock_pinch_guard_ratio:.3f} "
                    f"and pinch_conf<{profile.lock_pinch_guard_conf:.2f}"
                ),
                (220, 220, 220),
            ),
            (
                f"tap conf>=0.70 hold>={tap_required}ms",
                (220, 220, 220),
            ),
            (
                f"pinch conf>=0.65 hold>={profile.hold_ms}ms",
                (220, 220, 220),
            ),
            (
                f"volume_emit={profile.volume_emit_ms}ms deadzone={profile.volume_deadzone:.3f}",
                (220, 220, 220),
            ),
        ]
    )

    y = y0 + 24
    for text, color in lines:
        if y > y1 - 10:
            break
        cv2.putText(
            frame,
            text,
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 17


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
    debug_enabled: bool,
    using_calibration: bool,
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
    rec_text = (
        f"REC {'ON' if recording else 'OFF'} ({rec_label})  |  "
        f"PROFILE {'CALIB' if using_calibration else 'DEFAULT'}"
    )
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
        (
            "Keys: Q quit | C calibrate | X profile(calib/default) | R record | D debug "
            f"({'ON' if debug_enabled else 'OFF'}) | 1-5 labels (idle/open/fist/pinch/tap)"
        ),
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
    calibrated_profile = load_profile(cfg.profile_path)
    calibrated_profile.model_path = cfg.model_path
    default_runtime_profile = default_profile("defaults")
    default_runtime_profile.model_path = cfg.model_path
    use_calibration_profile = True
    profile = calibrated_profile
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
    debug_enabled = bool(cfg.show_debug)
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

            if debug_enabled:
                _draw_hand_debug(frame, lm_list)
                machine_debug = machine.debug_snapshot(now_ms)
                _draw_debug_panel(frame, prediction, profile, machine_debug)

            _draw_hud(
                frame,
                fps=fps_ema,
                prediction_label=prediction.label,
                confidence=prediction.confidence,
                locked=machine.locked,
                last_event=last_event,
                recording=recorder.is_recording,
                rec_label=recorder.active_label,
                debug_enabled=debug_enabled,
                using_calibration=use_calibration_profile,
            )
            cv2.imshow("Assistive Gesture Control Platform", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("x"), ord("X")):
                use_calibration_profile = not use_calibration_profile
                profile = calibrated_profile if use_calibration_profile else default_runtime_profile
                locked_state = machine.locked
                machine = GestureStateMachine(profile)
                machine.locked = locked_state
                prev_features = None
                last_event = (
                    "calibration_profile_enabled"
                    if use_calibration_profile
                    else "default_profile_enabled"
                )
            if key in (ord("d"), ord("D")):
                debug_enabled = not debug_enabled
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
                    new_profile = run_calibration(
                        cfg.camera_id,
                        calibrated_profile.model_path,
                        base_profile=calibrated_profile,
                    )
                    sample_count = float(new_profile.extra.get("calibration_samples", 0.0))
                    if sample_count > 0:
                        new_profile.name = calibrated_profile.name
                        new_profile.model_path = calibrated_profile.model_path
                        calibrated_profile = new_profile
                        save_profile(calibrated_profile, cfg.profile_path)

                        if use_calibration_profile:
                            locked_state = machine.locked
                            profile = calibrated_profile
                            machine = GestureStateMachine(profile)
                            machine.locked = locked_state
                            prev_features = None
                            last_event = "profile_recalibrated"
                        else:
                            last_event = "calibration_saved_default_mode_active"
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
