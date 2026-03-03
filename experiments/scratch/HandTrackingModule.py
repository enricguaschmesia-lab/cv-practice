"""
hand_detector.py

MediaPipe Hands (Tasks API) wrapper designed to be imported and reused.

Requires:
  - mediapipe >= 0.10.30 (Tasks API)
  - opencv-python

Model file:
  - models/hand_landmarker.task (relative to your working directory)
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, TypeAlias

import cv2
import mediapipe as mp


# ---- MediaPipe Tasks aliases ----
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode: TypeAlias = mp.tasks.vision.RunningMode


# ---- Hand skeleton topology (for drawing) ----
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def draw_fps(frame_bgr, fps: float) -> None:
    """Small, subtle FPS overlay in the top-left corner."""
    text = f"{fps:.1f} FPS"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    pad = 8

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 12, 12  # top-left anchor

    x1, y1 = x, y
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)

    tx = x + pad
    ty = y + pad + th
    cv2.putText(frame_bgr, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hand_landmarks(
    frame_bgr,
    hand_landmarks_list,
    *,
    highlight_index_tip: bool = False,
) -> None:
    """
    Draw connections + points for one or more hands.

    hand_landmarks_list can be:
      - list of hands: [ [lm0..lm20], [lm0..lm20], ... ]
      - a single hand (list of 21 lms): [lm0..lm20]
    """
    # If user passed a single hand (list of 21 landmark objects), wrap it.
    if hand_landmarks_list and hasattr(hand_landmarks_list[0], "x"):
        hands = [hand_landmarks_list]
    else:
        hands = hand_landmarks_list

    h, w = frame_bgr.shape[:2]

    for hand_landmarks in hands:
        # Connections
        for a, b in HAND_CONNECTIONS:
            la = hand_landmarks[a]
            lb = hand_landmarks[b]
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 2)

        # Points
        for idx, lm in enumerate(hand_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)

            # index fingertip highlight (optional)
            if highlight_index_tip and idx == 8:
                cv2.circle(frame_bgr, (x, y), 9, (255, 0, 0), -1)
            else:
                cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)


class HandDetector:
    """
    Reusable hand detector based on MediaPipe Tasks (HandLandmarker).

    Typical usage:
        from hand_detector import HandDetector

        with HandDetector(model_path="models/hand_landmarker.task") as detector:
            frame_out, result, lm_list = detector.find_hands(frame, flip=True, draw=True)
    """

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: VisionRunningMode = VisionRunningMode.VIDEO,
    ):
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.running_mode = running_mode

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: '{self.model_path}'.\n"
                f"Expected a .task file (e.g., models/hand_landmarker.task)."
            )

        self._landmarker: Optional[HandLandmarker] = None
        self._last_timestamp_ms = 0
        self._create_landmarker()

    def _create_landmarker(self) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _timestamp_ms(self) -> int:
        """Monotonic timestamp (ms), required for VIDEO mode."""
        ts = int(time.perf_counter() * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    @staticmethod
    def landmarks_to_pixels(hand_landmarks, image_shape) -> List[Tuple[int, int]]:
        """Convert one hand's normalized landmarks to pixel (x, y) coordinates."""
        h, w = image_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    def detect(self, frame_bgr, timestamp_ms: Optional[int] = None):
        """Run the hand landmarker on one BGR frame and return the raw MediaPipe result."""
        if self._landmarker is None:
            raise RuntimeError("HandDetector is closed. Create a new instance or use it within a 'with' block.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode == VisionRunningMode.VIDEO:
            ts = self._timestamp_ms() if timestamp_ms is None else timestamp_ms
            return self._landmarker.detect_for_video(mp_image, ts)

        raise ValueError(f"Unsupported running_mode for this wrapper: {self.running_mode}")

    def find_hands(
        self,
        frame_bgr,
        hand_num: int = 0,
        draw: bool = True,
        flip: bool = False,
        highlight_index_tip: bool = False,
    ):
        """
        Optionally flip, run detection, optionally draw.

        Returns:
            (output_frame_bgr, result, lmList)

        lmList format:
            [[idx, x_px, y_px], ...] for the selected hand_num.
            Empty list if no hands or invalid hand_num.
        """
        lmList: List[List[int]] = []

        out = cv2.flip(frame_bgr, 1) if flip else frame_bgr
        result = self.detect(out)

        if result.hand_landmarks and 0 <= hand_num < len(result.hand_landmarks):
            hand = result.hand_landmarks[hand_num]  # list of 21 landmarks

            if draw:
                draw_hand_landmarks(out, hand, highlight_index_tip=highlight_index_tip)

            # Build lmList for this hand (all 21 landmarks)
            h, w = out.shape[:2]
            for idx, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])

        return out, result, lmList


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    # FPS smoothing (EMA)
    prev_t = time.perf_counter()
    fps_ema = 0.0
    fps_alpha = 0.10

    with HandDetector(model_path="models/hand_landmarker.task", num_hands=2) as detector:
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

            draw_fps(frame, fps_ema)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()