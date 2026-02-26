import cv2
import time
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

# Same topology as classic MediaPipe Hands (used only for drawing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm base
]

def draw_fps(frame_bgr, fps: float):

    text = f"{fps:.1f} FPS"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    pad = 8

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 12, 12  # top-left anchor

    # Background rectangle (semi-transparent)
    x1, y1 = x, y
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Text
    tx = x + pad
    ty = y + pad + th
    cv2.putText(frame_bgr, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_hand_landmarks(frame_bgr, hand_landmarks_list):
    h, w = frame_bgr.shape[:2]

    for hand_landmarks in hand_landmarks_list:
        # Draw connections first
        for a, b in HAND_CONNECTIONS:
            la = hand_landmarks[a]
            lb = hand_landmarks[b]
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 2)

        # Draw points
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0)

# FPS smoothing (EMA) to reduce jitter
prev_t = time.perf_counter()
fps_ema = 0.0
fps_alpha = 0.10

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema == 0.0 else (1 - fps_alpha) * fps_ema + fps_alpha * inst_fps

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.perf_counter() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for id, lm in enumerate(hand_landmarks):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # print(f"Landmark {id}: ({cx}, {cy})")
                    if id == 8:  # Index fingertip
                        cv2.circle(frame, (cx, cy), 9, (255, 0, 0), -1)

            draw_hand_landmarks(frame, result.hand_landmarks)
            

        draw_fps(frame, fps_ema)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()