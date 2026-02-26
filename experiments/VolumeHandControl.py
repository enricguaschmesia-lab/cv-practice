import math
import time

import cv2
import HandTrackingModule as htm
from pycaw.pycaw import AudioUtilities


# -----------------------------
# Config
# -----------------------------
W_CAM, H_CAM = 640, 480
CAM_INDEX = 0

# Gesture calibration: ratio = pinch_distance / palm_scale
PINCH_MIN = 0.12
PINCH_MAX = 0.85

# Gate volume control with a "lock" gesture (extend hand)
ENABLE_GATING = True
LOCK_FRAMES = 4
UNLOCK_FRAMES = 2
ENABLE_THRESH = 0.95   # ratio must be below to unlock
DISABLE_THRESH = 1.05  # ratio must be above to lock

# Volume smoothing (applied ONLY when control is enabled)
VOL_SMOOTH_ALPHA = 0.20

# UI toggles
SHOW_SCANLINES = True
SHOW_GESTURE_DEBUG = False


# -----------------------------
# Neon UI (BGR colors)
# -----------------------------
NEON_CYAN = (255, 255, 0)
NEON_MAGENTA = (255, 0, 255)
NEON_PURPLE = (200, 0, 255)
NEON_GREEN = (0, 255, 120)
INK = (10, 10, 18)
WHITE = (245, 245, 245)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def inv_lerp(a: float, b: float, v: float) -> float:
    if abs(b - a) < 1e-9:
        return 0.0
    return (v - a) / (b - a)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def dist(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def neon_text(img, text, org, color=WHITE, scale=0.6, thickness=2) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    glow = img.copy()
    cv2.putText(glow, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness + 4, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.12, img, 0.88, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def neon_panel(img, x1, y1, x2, y2, border_color=NEON_CYAN) -> None:
    panel = img.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), INK, -1, cv2.LINE_AA)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)

    border = img.copy()
    cv2.rectangle(border, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
    cv2.addWeighted(border, 0.80, img, 0.20, 0, img)

    glow = img.copy()
    cv2.rectangle(glow, (x1, y1), (x2, y2), border_color, 8, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.08, img, 0.92, 0, img)


def neon_glow_line(img, p1, p2, color, thickness=2) -> None:
    for extra_t, alpha in [(10, 0.08), (6, 0.14), (3, 0.22)]:
        ov = img.copy()
        cv2.line(ov, p1, p2, color, thickness + extra_t, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def neon_glow_circle(img, center, radius, color) -> None:
    for extra_r, alpha in [(10, 0.08), (6, 0.14), (3, 0.22)]:
        ov = img.copy()
        cv2.circle(ov, center, radius + extra_r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


def scanlines(img, spacing=7) -> None:
    h, w = img.shape[:2]
    ov = img.copy()
    for y in range(0, h, spacing):
        cv2.line(ov, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(ov, 0.10, img, 0.90, 0, img)


def draw_hud(img, vol_scalar: float, enabled: bool, fps: float) -> None:
    x1, y1, x2, y2 = 18, 18, 180, 250
    neon_panel(img, x1, y1, x2, y2, border_color=NEON_PURPLE)

    neon_text(img, "VOLUME", (x1 + 16, y1 + 34), color=NEON_CYAN, scale=0.7, thickness=2)

    status = "CONTROL" if enabled else "LOCKED"
    neon_text(img, status, (x1 + 16, y1 + 62), color=(NEON_GREEN if enabled else NEON_MAGENTA), scale=0.55, thickness=2)

    bx1, by1, bx2, by2 = x1 + 18, y1 + 82, x1 + 52, y2 - 18
    neon_panel(img, bx1, by1, bx2, by2, border_color=NEON_CYAN)

    t = clamp(vol_scalar, 0.0, 1.0)
    fill_y = int(lerp(by2 - 6, by1 + 6, t))
    fill = img.copy()
    cv2.rectangle(fill, (bx1 + 6, fill_y), (bx2 - 6, by2 - 6), NEON_CYAN, -1, cv2.LINE_AA)
    cv2.addWeighted(fill, 0.55, img, 0.45, 0, img)

    neon_text(img, f"{int(round(t * 100)):3d}%", (x1 + 74, y1 + 140), color=WHITE, scale=0.95, thickness=2)
    neon_text(img, f"{fps:4.1f} FPS", (x1 + 74, y1 + 180), color=NEON_MAGENTA, scale=0.55, thickness=2)


def get_endpoint_volume():
    return AudioUtilities.GetSpeakers().EndpointVolume


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAM)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    detector = htm.HandDetector(min_hand_detection_confidence=0.75)

    endpoint = get_endpoint_volume()
    current_vol = float(endpoint.GetMasterVolumeLevelScalar())
    target_vol = current_vol

    prev_t = time.perf_counter()
    fps_ema = 0.0
    fps_alpha = 0.10

    enabled_state = True
    lock_count = 0
    unlock_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=4)

        frame, _, lm_list = detector.find_hands(frame, flip=True, draw=False)

        if lm_list:
            thumb = (lm_list[4][1], lm_list[4][2])
            index = (lm_list[8][1], lm_list[8][2])
            wrist = (lm_list[0][1], lm_list[0][2])
            middle_tip = (lm_list[12][1], lm_list[12][2])
            pinky_base = (lm_list[17][1], lm_list[17][2])

            palm_scale = max(1.0, dist(wrist, pinky_base) * 1.7)
            pinch_ratio = dist(thumb, index) / palm_scale

            if ENABLE_GATING:
                gate_ratio = dist(wrist, middle_tip) / palm_scale
                enabled_raw = (gate_ratio < DISABLE_THRESH) if enabled_state else (gate_ratio < ENABLE_THRESH)

                if enabled_raw:
                    unlock_count += 1
                    lock_count = 0
                else:
                    lock_count += 1
                    unlock_count = 0

                if enabled_state and lock_count >= LOCK_FRAMES:
                    enabled_state = False
                    current_vol = float(endpoint.GetMasterVolumeLevelScalar())
                    target_vol = current_vol
                elif (not enabled_state) and unlock_count >= UNLOCK_FRAMES:
                    enabled_state = True
            else:
                enabled_state = True

            cx, cy = (thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2
            neon_glow_circle(frame, thumb, 8, NEON_CYAN)
            neon_glow_circle(frame, index, 8, NEON_CYAN)
            neon_glow_circle(frame, (cx, cy), 6, NEON_MAGENTA)
            neon_glow_line(frame, thumb, index, NEON_MAGENTA, thickness=2)

            t = clamp(inv_lerp(PINCH_MIN, PINCH_MAX, pinch_ratio), 0.0, 1.0)

            if enabled_state:
                target_vol = t
                current_vol = (1 - VOL_SMOOTH_ALPHA) * current_vol + VOL_SMOOTH_ALPHA * target_vol
                current_vol = float(clamp(current_vol, 0.0, 1.0))
                endpoint.SetMasterVolumeLevelScalar(current_vol, None)

            if SHOW_GESTURE_DEBUG:
                neon_text(frame, f"pinch:{pinch_ratio:.2f}", (20, H_CAM - 40), color=NEON_CYAN, scale=0.55, thickness=2)

        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema == 0.0 else (1 - fps_alpha) * fps_ema + fps_alpha * inst_fps

        draw_hud(frame, current_vol, enabled_state, fps_ema)

        if SHOW_SCANLINES:
            scanlines(frame)

        cv2.imshow("Neon Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()