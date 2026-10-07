"""Real-time OpenCV overlay: hand skeletons + a professional status panel."""

import cv2

from hand_tracking import HAND_CONNECTIONS
from hand_tracking.landmarks import INDEX_TIP, TIP_IDS, WRIST
from hand_tracking.handedness import hand_role
from utils.constants import MOUSE, UNKNOWN

HAND_COLORS = {"Left": (0, 220, 0), "Right": (255, 120, 0)}
DEFAULT_COLOR = (0, 200, 255)
UNKNOWN_COLOR = (160, 160, 160)


def _draw_hand_skeleton(frame, points, color):
    for conn in HAND_CONNECTIONS:
        start = tuple(points[conn.start].astype(int))
        end = tuple(points[conn.end].astype(int))
        cv2.line(frame, start, end, color, 2)
    for idx, (x, y) in enumerate(points.astype(int)):
        radius = 6 if idx in TIP_IDS else 4
        cv2.circle(frame, (int(x), int(y)), radius, color, cv2.FILLED)
        cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)


def _draw_text_box(frame, text, center_x, top_y, color, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(2, min(center_x - tw // 2, frame.shape[1] - tw - 4))
    y = max(th + 8, top_y)
    cv2.rectangle(frame, (x - 5, y - th - 6), (x + tw + 5, y + 5), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hands(frame, hands_by_label, mode):
    """Draw skeletons + per-hand gesture labels."""
    for label, h in hands_by_label.items():
        pts = h["points"]
        _draw_hand_skeleton(frame, pts, h["color"])
        wx, wy = pts[WRIST].astype(int)
        _draw_text_box(frame, f'{label} {h["score"]:.2f}', wx, wy + 18, h["color"], scale=0.6)
        top_y = int(min(p[1] for p in pts)) - 14
        gcolor = h["color"] if h["gesture"] != UNKNOWN else UNKNOWN_COLOR
        _draw_text_box(frame, f'{h["gesture"]}  {int(h["confidence"]*100)}%',
                       wx, top_y, gcolor, scale=0.75)
        if label == "Left" and mode == MOUSE:
            tx, ty = pts[INDEX_TIP].astype(int)
            cv2.circle(frame, (tx, ty), 12, (0, 255, 255), 2)


def draw_ui(frame, hands_by_label, fps, mode, locked, control_on, last_action,
            show_action, cursor_pos=None, hands_detected=0):
    """Draw the full status panel and hand overlays."""
    draw_hands(frame, hands_by_label, mode)

    title = "AI TOUCHLESS COMPUTER CONTROL"
    lines = [
        f"MODE: {mode}    CONTROL: {'LOCKED' if locked else 'UNLOCKED'}"
        f"    [{'ON' if control_on else 'OFF'}]",
        "",
    ]
    for label in ("Left", "Right"):
        h = hands_by_label.get(label)
        if h:
            lines.append(f'{label.upper()} HAND [{hand_role(label, mode)}]: '
                         f'{h["gesture"]}  {int(h["confidence"]*100)}%')
        else:
            lines.append(f'{label.upper()} HAND: NONE   Confidence: --')
    lines.append("")
    lines.append(f'ACTION: {last_action if show_action else "-"}')
    lines.append(f'FPS: {fps:5.1f}    HANDS: {hands_detected}')
    if cursor_pos:
        lines.append(f'CURSOR: {cursor_pos[0]}, {cursor_pos[1]}')
    lines.append("")
    lines.append("[N]ormal [M]ouse [P]resent  [C]ontrol  [R]eset  [Q]uit")

    x, y0, lh, w = 12, 30, 24, 620
    header_h = 30
    panel_h = header_h + lh * len(lines) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, 6), (x + w, panel_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, title, (x, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 215, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (x - 4, y0 + 2), (x + w - 4, y0 + 2), (0, 215, 255), 1)

    for i, line in enumerate(lines):
        if not line:
            continue
        col = (255, 255, 255)
        if line.startswith("MODE:"):
            col = (0, 160, 255) if locked else (0, 255, 0)
        elif line.startswith("ACTION:"):
            col = (0, 255, 255)
        elif line.startswith("FPS:"):
            col = (200, 200, 200)
        cv2.putText(frame, line, (x, y0 + header_h + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
