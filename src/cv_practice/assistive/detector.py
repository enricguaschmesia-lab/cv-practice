from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandDetector:
    """Small wrapper around MediaPipe Tasks hand landmarker."""

    def __init__(
        self,
        model_path: str,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.6,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: VisionRunningMode = VisionRunningMode.VIDEO,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self._last_timestamp_ms = 0
        self._running_mode = running_mode
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector: Optional[HandLandmarker] = HandLandmarker.create_from_options(opts)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ts_ms(self) -> int:
        now = int(time.perf_counter() * 1000)
        if now <= self._last_timestamp_ms:
            now = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now
        return now

    def find_hands(self, frame_bgr, hand_num: int = 0, flip: bool = True):
        if self._detector is None:
            raise RuntimeError("Detector was closed.")

        out = cv2.flip(frame_bgr, 1) if flip else frame_bgr
        frame_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._running_mode != VisionRunningMode.VIDEO:
            raise ValueError("Only VIDEO running mode is supported.")

        result = self._detector.detect_for_video(mp_image, self._ts_ms())
        lm_list: list[list[int]] = []
        if result.hand_landmarks and 0 <= hand_num < len(result.hand_landmarks):
            hand = result.hand_landmarks[hand_num]
            h, w = out.shape[:2]
            for idx, lm in enumerate(hand):
                lm_list.append([idx, int(lm.x * w), int(lm.y * h)])
        return out, result, lm_list

