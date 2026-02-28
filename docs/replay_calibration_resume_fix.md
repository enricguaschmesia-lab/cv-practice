# Replay Spec: AGCP Calibration Resume Fix

This file captures the **exact code changes** introduced for the prompt:

> "After calibration, code execution stops. However, it should continue with the new calibrated parameters. Carry out the necessary code modifications to make sure this functions this way."

Use this spec in another conversation to reproduce the same behavior change.

## Scope
- File modified: `src/cv_practice/assistive/app.py`
- Functional goal:
1. Recalibration (`C` key) must not terminate app execution.
2. App must resume live loop after calibration with updated profile when valid.
3. Camera/detector resources must be safely reinitialized.
4. Cancelled/empty calibration must not overwrite active profile.

## Apply-Patch (Exact Replay)
```patch
*** Begin Patch
*** Update File: src/cv_practice/assistive/app.py
@@
 def _draw_hud(
@@
     )
 
 
+def _open_camera(cfg):
+    cap = cv2.VideoCapture(cfg.camera_id)
+    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
+    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
+    if not cap.isOpened():
+        raise RuntimeError("Could not open webcam.")
+    return cap
+
+
 def run_assistive_app(config_path: str | None = None) -> None:
 @@
-    cap = cv2.VideoCapture(cfg.camera_id)
-    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
-    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
-    if not cap.isOpened():
-        raise RuntimeError("Could not open webcam.")
+    cap = _open_camera(cfg)
+    detector = HandDetector(model_path=profile.model_path, num_hands=1)
 @@
-    with HandDetector(model_path=profile.model_path, num_hands=1) as detector:
+    try:
         while True:
 @@
             if key == ord("r"):
                 if recorder.is_recording:
                     recorder.stop()
                 else:
                     recorder.start("idle")
             if key == ord("c"):
-                new_profile = run_calibration(cfg.camera_id, profile.model_path)
-                new_profile.name = profile.name
-                profile = new_profile
-                save_profile(profile, cfg.profile_path)
-                machine = GestureStateMachine(profile)
-                prev_features = None
-                last_event = "profile_recalibrated"
+                was_recording = recorder.is_recording
+                if was_recording:
+                    recorder.stop()
+
+                detector.close()
+                cap.release()
+                cv2.destroyAllWindows()
+                try:
+                    new_profile = run_calibration(cfg.camera_id, profile.model_path)
+                    sample_count = float(new_profile.extra.get("calibration_samples", 0.0))
+                    if sample_count > 0:
+                        new_profile.name = profile.name
+                        new_profile.model_path = profile.model_path
+                        profile = new_profile
+                        save_profile(profile, cfg.profile_path)
+                        machine = GestureStateMachine(profile)
+                        prev_features = None
+                        last_event = "profile_recalibrated"
+                    else:
+                        last_event = "calibration_cancelled_or_empty"
+                except Exception as exc:
+                    last_event = f"calibration_failed ({exc.__class__.__name__})"
+                finally:
+                    cap = _open_camera(cfg)
+                    detector = HandDetector(model_path=profile.model_path, num_hands=1)
+                    prev_t = time.perf_counter()
+                    if was_recording:
+                        recorder.start(recorder.active_label)
             if key in label_map and recorder.is_recording:
                 recorder.set_label(label_map[key])
-
-    cap.release()
-    recorder.stop()
-    summary = telemetry.finalize()
-    cv2.destroyAllWindows()
+    finally:
+        detector.close()
+        cap.release()
+        recorder.stop()
+        summary = telemetry.finalize()
+        cv2.destroyAllWindows()
     print(
         "AGCP session summary: "
         f"frames={summary.frames}, commands={summary.commands}, "
         f"fps_avg={summary.fps_avg:.1f}, latency_p95_ms={summary.latency_p95_ms:.1f}"
     )
*** End Patch
```

## Behavioral Expectations After Replay
1. Pressing `C` temporarily pauses runtime, runs calibration, then returns to the live app loop.
2. If calibration produced samples, the new calibrated profile is persisted and activated immediately.
3. If calibration is cancelled/empty, current profile remains active.
4. If calibration throws, app recovers and continues with previous profile.
5. Resource lifecycle is safe (`detector.close()`, `cap.release()`, reopen camera/detector, final cleanup in `finally`).

## Minimal Verification
1. Start app.
2. Trigger calibration with `C`.
3. Complete at least one calibration step and exit calibration.
4. Confirm app window returns and continues running (no termination).
5. Confirm `last_event` shows one of:
- `profile_recalibrated`
- `calibration_cancelled_or_empty`
- `calibration_failed (<ExceptionName>)`
