from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, TypeAlias

import cv2
import mediapipe as mp

import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)).")

# FPS smoothing (EMA)
prev_t = time.perf_counter()
fps_ema = 0.0
fps_alpha = 0.10

with htm.HandDetector(model_path="models/hand_landmarker.task", num_hands=2) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema == 0.0 else (1 - fps_alpha) * fps_ema + fps_alpha * inst_fps

        frame, _, lmList = detector.find_hands(frame, hand_num=0, draw=True, flip=True, highlight_index_tip=True)

        # Example: print landmark list when available
        # if lmList:
        #     print(lmList[4])

        htm.draw_fps(frame, fps_ema)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()