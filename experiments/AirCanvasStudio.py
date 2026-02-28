import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import HandTrackingModule as htm


# -----------------------------
# Config
# -----------------------------
W_CAM, H_CAM = 1280, 720
CAM_INDEX = 0

DRAW_THICKNESS = 8
ERASER_THICKNESS = 45
SMOOTH_ALPHA = 0.40

PINCH_DRAW_THRESH = 0.22
ACTION_COOLDOWN_SEC = 0.35
MAX_UNDO_STEPS = 15
STRAND_ERASE_RADIUS = 60.0
CLAP_CLEAR_DIST_RATIO = 0.65
CLAP_CLEAR_COOLDOWN_SEC = 0.9

SAVE_DIR = "experiments/captures"

# iOS-inspired color palette (BGR)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "Blue": (255, 149, 0),    # System Blue
    "Green": (76, 217, 100),  # System Green
    "Red": (59, 59, 255),     # System Red
    "Yellow": (0, 204, 255),  # System Yellow
    "Purple": (255, 100, 175),# System Purple
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def pt(lm_list: List[List[int]], idx: int) -> Tuple[int, int]:
    return lm_list[idx][1], lm_list[idx][2]


def hand_to_lm_list(hand_landmarks, image_shape) -> List[List[int]]:
    h, w = image_shape[:2]
    out: List[List[int]] = []
    for idx, lm in enumerate(hand_landmarks):
        out.append([idx, int(lm.x * w), int(lm.y * h)])
    return out


def draw_rounded_rect(img, pt1, pt2, color, thickness, r, line_type=cv2.LINE_AA):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, line_type)

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, line_type)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, line_type)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, line_type)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, line_type)


def fill_rounded_rect(img, pt1, pt2, color, r, line_type=cv2.LINE_AA):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, line_type)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, line_type)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, line_type)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, line_type)
    
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)


def draw_glass_panel(frame, pt1, pt2, alpha=0.6, color=(20, 20, 20), radius=20):
    overlay = frame.copy()
    fill_rounded_rect(overlay, pt1, pt2, color, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    draw_rounded_rect(frame, pt1, pt2, (100, 100, 100), 1, radius)


def make_buttons(frame_w: int, frame_h: int) -> List[dict]:
    labels = list(COLORS.keys()) + ["Eraser", "Undo", "Clear", "Save"]
    n = len(labels)
    
    margin = 15
    button_w = 75
    button_h = 45
    
    total_w = n * button_w + (n + 1) * margin
    start_x = (frame_w - total_w) // 2
    
    dock_h = 70
    y_center = 50  # Top position with margin
    
    buttons = []
    for i, label in enumerate(labels):
        x1 = start_x + margin + i * (button_w + margin)
        y1 = y_center - button_h // 2
        x2 = x1 + button_w
        y2 = y1 + button_h
        buttons.append({"label": label, "rect": (x1, y1, x2, y2)})
    
    dock_rect = (start_x, y_center - dock_h // 2, start_x + total_w, y_center + dock_h // 2)
    return buttons, dock_rect


def draw_buttons(
    frame,
    buttons: List[dict],
    dock_rect: Tuple[int, int, int, int],
    hover_label: Optional[str],
    active_mode: str,
    active_color_name: str,
) -> None:
    # Draw Dock
    draw_glass_panel(frame, (dock_rect[0], dock_rect[1]), (dock_rect[2], dock_rect[3]), alpha=0.4)

    for b in buttons:
        label = b["label"]
        x1, y1, x2, y2 = b["rect"]
        is_active = False

        if label in COLORS:
            color = COLORS[label]
            is_active = (label == active_color_name and active_mode == "draw")
            text_color = (255, 255, 255)
            if is_active:
                fill_rounded_rect(frame, (x1, y1), (x2, y2), color, 12)
                draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, 12)
            else:
                fill_rounded_rect(frame, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), color, 10)
        else:
            is_active = (label == "Eraser" and active_mode == "eraser")
            if is_active:
                fill_rounded_rect(frame, (x1, y1), (x2, y2), (120, 120, 120), 12)
                draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, 12)
            else:
                fill_rounded_rect(frame, (x1, y1), (x2, y2), (40, 40, 40), 12)
            
            text_color = (240, 240, 240)

        if label == hover_label and not is_active:
            draw_rounded_rect(frame, (x1, y1), (x2, y2), (200, 200, 200), 1, 12)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(frame, label, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)


def clone_strokes(strokes: List[dict]) -> List[dict]:
    return [
        {
            "mode": s["mode"],
            "color": s["color"],
            "thickness": s["thickness"],
            "points": list(s["points"]),
        }
        for s in strokes
    ]


def push_undo(undo_stack: List[List[dict]], strokes: List[dict]) -> None:
    undo_stack.append(clone_strokes(strokes))
    if len(undo_stack) > MAX_UNDO_STEPS:
        undo_stack.pop(0)


def finger_states(lm_list: List[List[int]]) -> Dict[str, bool]:
    thumb_tip = pt(lm_list, 4)
    thumb_ip = pt(lm_list, 3)
    ring_mcp = pt(lm_list, 14)

    thumb_extended = dist(thumb_tip, ring_mcp) > dist(thumb_ip, ring_mcp) * 1.08

    return {
        "thumb": thumb_extended,
        "index": lm_list[8][2] < lm_list[6][2],
        "middle": lm_list[12][2] < lm_list[10][2],
        "ring": lm_list[16][2] < lm_list[14][2],
        "pinky": lm_list[20][2] < lm_list[18][2],
    }


def get_hover_label(pointer: Tuple[int, int], buttons: List[dict]) -> Optional[str]:
    px, py = pointer
    for b in buttons:
        x1, y1, x2, y2 = b["rect"]
        if x1 <= px <= x2 and y1 <= py <= y2:
            return b["label"]
    return None


def execute_action(
    label: str,
    canvas: np.ndarray,
    strokes: List[dict],
    undo_stack: List[List[dict]],
    active_mode: str,
    active_color_name: str,
) -> Tuple[str, str, str]:
    message = ""

    if label in COLORS:
        active_mode = "draw"
        active_color_name = label
        message = f"Selected {label}"
    elif label == "Eraser":
        active_mode = "eraser"
        message = "Eraser Mode"
    elif label == "Undo":
        if undo_stack:
            strokes[:] = undo_stack.pop()
            render_canvas(canvas, strokes)
            message = "Action Undone"
        else:
            message = "Nothing to undo"
    elif label == "Clear":
        push_undo(undo_stack, strokes)
        strokes.clear()
        canvas[:] = 0
        message = "Canvas Cleared"
    elif label == "Save":
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = f"air_canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, canvas)
        message = f"Saved to {filename}"

    return active_mode, active_color_name, message


def render_canvas(canvas: np.ndarray, strokes: List[dict]) -> None:
    canvas[:] = 0
    for s in strokes:
        pts = s["points"]
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], s["color"], s["thickness"], cv2.LINE_AA)


def remove_nearest_draw_stroke(strokes: List[dict], center: Tuple[int, int], radius: float = STRAND_ERASE_RADIUS) -> bool:
    cx, cy = center
    best_idx = -1
    best_d = float("inf")

    for i, s in enumerate(strokes):
        if s["mode"] != "draw":
            continue
        for px, py in s["points"]:
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_idx = i

    if best_idx >= 0 and best_d <= radius:
        strokes.pop(best_idx)
        return True
    return False


def draw_fps_pill(frame_bgr, fps: float) -> None:
    text = f"{fps:.0f} FPS"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    
    pad_x, pad_y = 12, 6
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    x1, y1 = W_CAM - w - 20, H_CAM - h - 40  # Moved to bottom-right
    x2, y2 = x1 + w, y1 + h
    
    draw_glass_panel(frame_bgr, (x1, y1), (x2, y2), alpha=0.5, radius=h // 2)
    cv2.putText(frame_bgr, text, (x1 + pad_x, y1 + pad_y + th), font, scale, (200, 200, 200), thick, cv2.LINE_AA)


def draw_status_pill(frame, message, duration_remaining):
    if duration_remaining <= 0:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(message, font, scale, thick)
    
    pad_x, pad_y = 20, 10
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    x1, y1 = (W_CAM - w) // 2, H_CAM - h - 100 # Moved to bottom center
    x2, y2 = x1 + w, y1 + h
    
    alpha = min(0.7, duration_remaining * 2.0)
    draw_glass_panel(frame, (x1, y1), (x2, y2), alpha=alpha, radius=h // 2)
    cv2.putText(frame, message, (x1 + pad_x, y1 + pad_y + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAM)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    canvas: Optional[np.ndarray] = None
    buttons: Optional[List[dict]] = None
    dock_rect: Optional[Tuple[int, int, int, int]] = None
    undo_stack: List[List[dict]] = []
    strokes: List[dict] = []

    active_mode = "draw"
    active_color_name = "Blue"

    smooth_pointer_1: Optional[Tuple[int, int]] = None
    prev_draw_point: Optional[Tuple[int, int]] = None
    stroke_active = False
    current_stroke: Optional[dict] = None

    action_cooldown_primary = 0.0
    action_cooldown_secondary = 0.0
    strand_erase_cooldown_until = 0.0
    clap_clear_cooldown_until = 0.0
    status_message = ""
    status_until = 0.0

    prev_t = time.perf_counter()
    fps_ema = 0.0
    fps_alpha = 0.10

    with htm.HandDetector(model_path="models/hand_landmarker.task", num_hands=2) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, result, _ = detector.find_hands(frame, hand_num=0, draw=False, flip=True)

            if canvas is None:
                h, w = frame.shape[:2]
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                buttons, dock_rect = make_buttons(w, h)

            assert canvas is not None
            assert buttons is not None
            assert dock_rect is not None

            hover_label = None
            hand_lm_lists: List[List[List[int]]] = []
            if result and result.hand_landmarks:
                hand_lm_lists = [hand_to_lm_list(hand, frame.shape) for hand in result.hand_landmarks]

            primary = hand_lm_lists[0] if len(hand_lm_lists) >= 1 else None
            secondary = hand_lm_lists[1] if len(hand_lm_lists) >= 2 else None

            # Clap to clear: both hands open and close together.
            if primary and secondary:
                p_fingers = finger_states(primary)
                s_fingers = finger_states(secondary)
                p_open = p_fingers["index"] and p_fingers["middle"] and p_fingers["ring"] and p_fingers["pinky"]
                s_open = s_fingers["index"] and s_fingers["middle"] and s_fingers["ring"] and s_fingers["pinky"]

                if p_open and s_open:
                    p_center = pt(primary, 9)
                    s_center = pt(secondary, 9)
                    p_scale = max(40.0, dist(pt(primary, 0), pt(primary, 5)) + dist(pt(primary, 0), pt(primary, 17)))
                    s_scale = max(40.0, dist(pt(secondary, 0), pt(secondary, 5)) + dist(pt(secondary, 0), pt(secondary, 17)))
                    clap_thresh = CLAP_CLEAR_DIST_RATIO * ((p_scale + s_scale) * 0.5)
                    now_sec = time.perf_counter()

                    if dist(p_center, s_center) < clap_thresh and now_sec >= clap_clear_cooldown_until:
                        push_undo(undo_stack, strokes)
                        strokes.clear()
                        canvas[:] = 0
                        status_message = "Canvas Cleared (Clap)"
                        status_until = now_sec + 1.6
                        clap_clear_cooldown_until = now_sec + CLAP_CLEAR_COOLDOWN_SEC
                        stroke_active = False
                        current_stroke = None
                        prev_draw_point = None

            # Secondary hand logic
            if secondary:
                s_fingers = finger_states(secondary)
                s_wrist = pt(secondary, 0)
                s_thumb = pt(secondary, 4)
                s_index = pt(secondary, 8)
                s_index_mcp = pt(secondary, 5)
                s_pinky_mcp = pt(secondary, 17)
                s_palm = max(40.0, dist(s_wrist, s_index_mcp) + dist(s_wrist, s_pinky_mcp))
                s_pinch_ratio = dist(s_thumb, s_index) / s_palm

                s_hover = get_hover_label(s_index, buttons)
                s_click = s_fingers["index"] and s_pinch_ratio < PINCH_DRAW_THRESH
                now_sec = time.perf_counter()

                if s_hover:
                    hover_label = s_hover
                if s_hover and s_click and now_sec >= action_cooldown_secondary:
                    active_mode, active_color_name, status_message = execute_action(
                        s_hover, canvas, strokes, undo_stack, active_mode, active_color_name
                    )
                    status_until = now_sec + 2.0
                    action_cooldown_secondary = now_sec + ACTION_COOLDOWN_SEC

                cv2.circle(frame, s_index, 6, (255, 255, 255), -1, cv2.LINE_AA)

            if primary:
                fingers = finger_states(primary)
                wrist = pt(primary, 0)
                thumb_tip = pt(primary, 4)
                index_tip = pt(primary, 8)
                index_mcp = pt(primary, 5)
                pinky_mcp = pt(primary, 17)

                palm_scale = max(40.0, dist(wrist, index_mcp) + dist(wrist, pinky_mcp))
                pinch_ratio = dist(thumb_tip, index_tip) / palm_scale

                if smooth_pointer_1 is None:
                    smooth_pointer_1 = index_tip
                else:
                    sx = int((1.0 - SMOOTH_ALPHA) * smooth_pointer_1[0] + SMOOTH_ALPHA * index_tip[0])
                    sy = int((1.0 - SMOOTH_ALPHA) * smooth_pointer_1[1] + SMOOTH_ALPHA * index_tip[1])
                    smooth_pointer_1 = (sx, sy)

                p_hover = get_hover_label(smooth_pointer_1, buttons)
                if p_hover:
                    hover_label = p_hover

                open_hand_pose = fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]
                draw_pose = fingers["index"] and (not fingers["middle"])
                strand_eraser_pose = (
                    fingers["thumb"]
                    and fingers["index"]
                    and fingers["middle"]
                    and (not fingers["ring"])
                    and (not fingers["pinky"])
                )
                eraser_pose = (
                    fingers["index"]
                    and fingers["middle"]
                    and (not fingers["ring"])
                    and (not fingers["pinky"])
                    and (not strand_eraser_pose)
                )
                pinch_draw = pinch_ratio < PINCH_DRAW_THRESH and fingers["index"]

                in_toolbar = p_hover is not None
                now_sec = time.perf_counter()
                p_click = fingers["index"] and pinch_ratio < PINCH_DRAW_THRESH

                if in_toolbar and p_click and now_sec >= action_cooldown_primary:
                    active_mode, active_color_name, status_message = execute_action(
                        p_hover, canvas, strokes, undo_stack, active_mode, active_color_name
                    )
                    status_until = now_sec + 2.0
                    action_cooldown_primary = now_sec + ACTION_COOLDOWN_SEC

                # Strand Erase
                if strand_eraser_pose and (not in_toolbar) and now_sec >= strand_erase_cooldown_until:
                    push_undo(undo_stack, strokes)
                    if remove_nearest_draw_stroke(strokes, smooth_pointer_1):
                        render_canvas(canvas, strokes)
                        status_message = "Strand Erased"
                    else:
                        status_message = "No Strand Nearby"
                    status_until = now_sec + 1.5
                    strand_erase_cooldown_until = now_sec + ACTION_COOLDOWN_SEC

                mode_now = active_mode
                if eraser_pose:
                    mode_now = "eraser"

                should_draw = (draw_pose or pinch_draw or eraser_pose) and (not in_toolbar) and (not strand_eraser_pose)
                action_active = should_draw or strand_eraser_pose or (in_toolbar and p_click)

                # Drawing logic
                # Dock is at bottom, so avoid drawing when pointer is in dock area
                in_dock = (dock_rect[0] <= smooth_pointer_1[0] <= dock_rect[2] and 
                           dock_rect[1] <= smooth_pointer_1[1] <= dock_rect[3])

                if should_draw and not in_dock:
                    desired_mode = "draw" if mode_now == "draw" else "erase"
                    desired_color = COLORS[active_color_name] if mode_now == "draw" else (0, 0, 0)
                    desired_thickness = DRAW_THICKNESS if mode_now == "draw" else ERASER_THICKNESS

                    if not stroke_active:
                        push_undo(undo_stack, strokes)
                        stroke_active = True
                        prev_draw_point = smooth_pointer_1
                        current_stroke = {
                            "mode": desired_mode,
                            "color": desired_color,
                            "thickness": desired_thickness,
                            "points": [smooth_pointer_1],
                        }
                    elif current_stroke is not None:
                        style_changed = (
                            current_stroke["mode"] != desired_mode
                            or current_stroke["color"] != desired_color
                            or current_stroke["thickness"] != desired_thickness
                        )
                        if style_changed:
                            if len(current_stroke["points"]) > 1:
                                strokes.append(current_stroke)
                            anchor = prev_draw_point if prev_draw_point is not None else smooth_pointer_1
                            current_stroke = {
                                "mode": desired_mode,
                                "color": desired_color,
                                "thickness": desired_thickness,
                                "points": [anchor],
                            }

                    if prev_draw_point is not None and current_stroke is not None:
                        cv2.line(
                            canvas,
                            prev_draw_point,
                            smooth_pointer_1,
                            current_stroke["color"],
                            current_stroke["thickness"],
                            cv2.LINE_AA,
                        )
                        current_stroke["points"].append(smooth_pointer_1)
                    prev_draw_point = smooth_pointer_1
                else:
                    if stroke_active and current_stroke is not None and len(current_stroke["points"]) > 1:
                        strokes.append(current_stroke)
                    current_stroke = None
                    stroke_active = False
                    prev_draw_point = None

                # Cursor Rendering
                cursor_color = COLORS[active_color_name] if mode_now == "draw" else (220, 220, 220)
                if strand_eraser_pose:
                    cv2.circle(frame, smooth_pointer_1, 15, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, smooth_pointer_1, 15, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, smooth_pointer_1, 2, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    if open_hand_pose and (not action_active):
                        overlay = frame.copy()
                        cv2.circle(overlay, smooth_pointer_1, 12, cursor_color, 2, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    else:
                        cv2.circle(frame, smooth_pointer_1, 10, cursor_color, 2, cv2.LINE_AA)
                        cv2.circle(frame, smooth_pointer_1, 2, cursor_color, -1, cv2.LINE_AA)
                cv2.circle(frame, index_tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                stroke_active = False
                prev_draw_point = None
                smooth_pointer_1 = None

            frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0.0)

            # Draw UI
            draw_buttons(frame, buttons, dock_rect, hover_label, active_mode, active_color_name)
            
            now = time.perf_counter()
            draw_status_pill(frame, status_message, status_until - now)

            dt = now - prev_t
            prev_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = inst_fps if fps_ema == 0.0 else (1 - fps_alpha) * fps_ema + fps_alpha * inst_fps
            draw_fps_pill(frame, fps_ema)

            # Subtle hint text at the very bottom
            cv2.putText(
                frame,
                "Draw: Index | Erase: Index+Middle | Strand: Thumb+Index+Middle | Save/Undo in Dock",
                (20, H_CAM - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Air Canvas Studio", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
