"""AI-Powered Touchless Human-Computer Interaction System.

Real-time, camera-driven computer control using MediaPipe hand tracking and
geometry-based gesture recognition. This is the application entry point that
wires together the modular packages:

    hand_tracking -> gesture -> control -> ui

Pipeline per frame
------------------
    Camera -> MediaPipe 21-landmark detection -> per-hand temporal smoothing
    -> gesture classification + confidence -> majority-vote gesture smoothing
    -> control lock gate -> stability/cooldown gate -> action dispatch
    -> OpenCV overlay (+ optional web dashboard)

Safety
------
The system starts CONTROL LOCKED. No computer action (media, volume, mouse,
scroll, screenshot, desktop) executes until Open Palm is held steadily to
unlock, and computer control can be killed at any time with the 'c' key.

Keys
----
    N  Normal mode      M  Mouse mode      P  Presentation mode
    C  Toggle computer control ON/OFF
    R  Reset smoothing / history / cooldowns
    Q / ESC  Quit
"""

import sys
import time

import cv2

from control import ActionController, MouseController, ScrollController
from control import platform_input as pin
from gesture import (
    GestureEngine,
    GestureSmoother,
    GestureThresholds,
    process_gesture_event,
)
from hand_tracking import (
    LandmarkSmoother,
    build_from_config,
    detect_hands,
    hand_count,
    track_landmarks,
)
from hand_tracking.landmarks import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, distance
from ui import HAND_COLORS, DashboardState, draw_ui, run_dashboard
from utils import FpsCounter, get_logger, load_config, setup_logging
from utils.constants import (
    LEFT,
    MOUSE,
    NORMAL,
    OPEN_PALM,
    PEACE,
    PRESENTATION,
    RIGHT,
)

log = get_logger("main")


class ModeManager:
    """Holds the active control mode and handles mode-switch keys."""

    KEY_MAP = {"m": MOUSE, "p": PRESENTATION, "n": NORMAL}

    def __init__(self, mode=NORMAL):
        self.mode = mode

    def handle_key(self, key_char: str) -> bool:
        new = self.KEY_MAP.get(str(key_char).lower())
        if new is None:
            return False
        self.mode = new
        return True


def _open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {index}). Check that a camera is "
            "connected and not in use by another application.")
    log.info("Camera initialized (index %s)", index)
    return cap


def run(cfg=None):
    cfg = cfg or load_config()
    setup_logging(cfg.get("logging.level", "INFO"),
                  cfg.get("logging.file"), cfg.root)

    # --- Camera + model ---------------------------------------------------- #
    try:
        cap = _open_camera(cfg.get("camera_index", 0))
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    try:
        landmarker = build_from_config(cfg)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        cap.release()
        return 1

    # --- Components -------------------------------------------------------- #
    g = cfg.get("gesture", {}) or {}
    m = cfg.get("movement", {}) or {}
    thresholds = GestureThresholds.from_config(cfg)
    engine = GestureEngine(thresholds=thresholds)

    fps_counter = FpsCounter()
    landmark_smoother = LandmarkSmoother(alpha=g.get("landmark_smoothing_alpha", 0.55))
    gesture_smoother = GestureSmoother(window=g.get("smoothing_window", 9),
                                       min_votes=g.get("min_votes", 5))
    actions = ActionController(cfg)
    mouse = MouseController.from_config(cfg)
    scroller = ScrollController.from_config(cfg)
    mode_mgr = ModeManager()

    min_conf = thresholds.confidence_threshold
    window_name = cfg.get("ui.window_name", "AI Touchless Computer Control")
    action_display = cfg.get("ui.action_display_seconds", 2.0)

    # --- Optional dashboard ------------------------------------------------ #
    dash = DashboardState()
    if cfg.get("dashboard.enabled", False):
        run_dashboard(dash, cfg.get("dashboard.host", "127.0.0.1"),
                      cfg.get("dashboard.port", 8000))
    dash.update(running=True)

    slideshow_active = False
    cursor_pos = None
    prev_mode = mode_mgr.mode
    frame_count = 0
    screen_w, screen_h = pin.screen_size()

    log.info("System ready. Starting LOCKED - hold Open Palm to unlock. "
             "[N/M/P] modes, [C] control, [R] reset, [Q] quit.")
    print("Touchless control started. Hold OPEN PALM to unlock. "
          "[N]ormal [M]ouse [P]resent  [C] control  [R] reset  [Q]/ESC quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Failed to read frame from webcam - stopping.")
            break

        frame = cv2.flip(frame, 1)  # mirrored selfie view
        height, width = frame.shape[:2]
        now = time.time()

        try:
            timestamp_ms = int(time.monotonic() * 1000) + frame_count
            frame_count += 1
            result = detect_hands(landmarker, frame, timestamp_ms)

            hands_by_label = {}
            active_labels = []
            for idx in range(hand_count(result)):
                pts, label, score = track_landmarks(result, idx, width, height,
                                                    landmark_smoother)
                active_labels.append(label)
                det = engine.detect(pts)
                gesture, conf = _smooth(gesture_smoother, label,
                                        det["gesture"], det["confidence"])
                if label not in hands_by_label or conf > hands_by_label[label]["confidence"]:
                    hands_by_label[label] = {
                        "points": pts, "color": HAND_COLORS.get(label, (0, 200, 255)),
                        "score": score, "fingers": det["fingers"],
                        "gesture": gesture, "confidence": conf,
                    }
            landmark_smoother.prune(active_labels)
            gesture_smoother.prune(active_labels)

            left = hands_by_label.get(LEFT)
            right = hands_by_label.get(RIGHT)
            mode = mode_mgr.mode

            cursor_pos, slideshow_active = _process_frame(
                mode=mode, left=left, right=right, actions=actions, mouse=mouse,
                scroller=scroller, now=now, width=width, height=height,
                screen_w=screen_w, screen_h=screen_h, min_conf=min_conf,
                slideshow_active=slideshow_active, dash=dash)

            fps = fps_counter.tick()
            show_action = (now - actions.last_action_time) < action_display
            draw_ui(frame, hands_by_label, fps, mode, actions.lock.locked,
                    actions.enabled, actions.last_action, show_action,
                    cursor_pos, hands_detected=len(hands_by_label))

            dash.update(mode=mode, locked=actions.lock.locked,
                        control_on=actions.enabled, fps=fps,
                        hands=len(hands_by_label),
                        left=_hand_snapshot(left), right=_hand_snapshot(right),
                        last_action=actions.last_action if show_action else "-")
        except Exception as exc:
            log.warning("Frame processing error (%s: %s)", type(exc).__name__, exc)
            draw_ui(frame, {}, fps_counter.tick(), mode_mgr.mode,
                    actions.lock.locked, actions.enabled, actions.last_action,
                    False, None, hands_detected=0)

        cv2.imshow(window_name, frame)
        if _handle_key(cv2.waitKey(1) & 0xFF, actions, mode_mgr, mouse, scroller,
                       landmark_smoother, gesture_smoother):
            break
        if mode_mgr.mode != prev_mode:
            log.info("Mode -> %s", mode_mgr.mode)
            mouse.reset()
            scroller.reset()
            prev_mode = mode_mgr.mode

    _shutdown(cap, landmarker, mouse, dash)
    return 0


def _smooth(smoother, label, name, conf):
    smoother.push(label, name, conf)
    return smoother.stable(label, name, conf)


def _hand_snapshot(hand):
    if not hand:
        return {"gesture": "NONE", "confidence": 0.0}
    return {"gesture": hand["gesture"], "confidence": hand["confidence"]}


def _static_action(actions, hand, mode, now, dash):
    """Resolve + gate + execute a static (posture) gesture for one hand."""
    if not hand:
        actions.request(None, 0.0, now)
        return
    intent = process_gesture_event(hand["gesture"], mode)
    if not intent or intent[1] == "lock":
        # Open-Palm lock is handled by ControlLock, not dispatched here.
        actions.request(None, 0.0, now)
        return
    label, kind, arg = intent
    if kind == "presentation" and arg == "toggle":
        arg = "stop" if _slideshow.get("active") else "start"
    key = (kind, arg)
    if actions.request(key, hand["confidence"], now):
        result = actions.execute(kind, arg, label=label, gesture=hand["gesture"], now=now)
        if result:
            dash.push_event(hand["gesture"], label)
            if kind == "presentation" and arg in ("start", "stop"):
                _slideshow["active"] = (arg == "start")


# Slideshow state lives here so presentation toggle can track start/stop.
_slideshow = {"active": False}


def _try_scroll(actions, hand, scroller, now, height, dash):
    """Two-finger (Victory) directional scroll for one hand."""
    if not hand or hand["gesture"] != PEACE or hand["confidence"] < actions.min_conf:
        scroller.reset()
        return False
    if not actions.can_act():
        scroller.reset()
        return False
    pts = hand["points"]
    # Use the wrist as the scroll reference point (normalised).
    x_norm = pts[WRIST][0]
    y_norm = pts[WRIST][1] / height
    direction = scroller.update(x_norm, y_norm, now)
    if direction and actions.fire_now(("scroll", direction), hand["confidence"], now):
        result = actions.execute("scroll", direction,
                                 label=f"Scroll {direction.capitalize()}",
                                 gesture=PEACE, now=now)
        if result:
            dash.push_event(PEACE, f"Scroll {direction.capitalize()}")
    return True


def _process_frame(mode, left, right, actions, mouse, scroller, now, width,
                   height, screen_w, screen_h, min_conf, slideshow_active, dash):
    """Route the two hands to their control channels for one frame.

    Returns (cursor_pos, slideshow_active).
    """
    cursor_pos = None

    # --- Control lock: Open Palm held steadily (NORMAL / MOUSE modes) ------- #
    open_palm = False
    palm_conf = 0.0
    for hand in (right, left):
        if hand and hand["gesture"] == OPEN_PALM and mode in (NORMAL, MOUSE):
            open_palm = True
            palm_conf = max(palm_conf, hand["confidence"])
    actions.lock.update(open_palm, palm_conf)

    # --- Presentation mode: left hand drives slides ------------------------ #
    if mode == PRESENTATION:
        mouse.reset()
        scroller.reset()
        _slideshow["active"] = slideshow_active
        _static_action(actions, left, PRESENTATION, now, dash)
        return cursor_pos, _slideshow["active"]

    # --- NORMAL / MOUSE: right hand = media/system, left hand = mouse ------- #
    media_hand = right if right else (left if mode == NORMAL else None)

    if mode == MOUSE:
        # Right hand keeps media/system control; left hand is the pointer.
        _static_action(actions, right, NORMAL, now, dash)
        if _try_scroll(actions, left, scroller, now, height, dash):
            mouse.release_if_down()
        elif left and actions.can_act():
            pts = left["points"]
            hand_size = max(distance(pts[MIDDLE_MCP], pts[WRIST]), 1e-6)
            pinch_ratio = distance(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size
            nx = pts[INDEX_TIP][0] / width
            ny = pts[INDEX_TIP][1] / height
            ev = mouse.update(True, nx, ny, pinch_ratio, now, screen_w, screen_h)
            cursor_pos = ev["cursor"]
            if ev["double"]:
                actions.last_action = "Double Click [mouse]"
                actions.last_action_time = now
                dash.push_event("Pinch", "Double Click")
            elif ev["clicked"]:
                actions.last_action = "Left Click [mouse]"
                actions.last_action_time = now
                dash.push_event("Pinch", "Left Click")
            elif ev["dropped"]:
                actions.last_action = "Drop [mouse]"
                actions.last_action_time = now
        else:
            mouse.update(False, 0, 0, 1.0, now, screen_w, screen_h)
            scroller.reset()
    else:  # NORMAL
        mouse.reset()
        if not _try_scroll(actions, media_hand, scroller, now, height, dash):
            _static_action(actions, media_hand, NORMAL, now, dash)

    return cursor_pos, slideshow_active


def _handle_key(key, actions, mode_mgr, mouse, scroller, landmark_smoother,
                gesture_smoother) -> bool:
    """Process a keypress. Returns True when the app should quit."""
    if key in (ord("q"), 27):
        return True
    if key in (ord("c"), ord("C")):
        actions.enabled = not actions.enabled
        actions.reset()
        mouse.reset()
        log.info("Computer control %s", "ENABLED" if actions.enabled else "DISABLED")
        print(f"Computer control {'ENABLED' if actions.enabled else 'DISABLED'}.")
        return False
    if key in (ord("r"), ord("R")):
        landmark_smoother.reset()
        gesture_smoother.reset()
        actions.reset()
        mouse.reset()
        scroller.reset()
        log.info("Smoothing, history and cooldowns reset")
        return False
    ch = chr(key) if 32 <= key < 127 else ""
    if ch and mode_mgr.handle_key(ch):
        mouse.reset()
        scroller.reset()
    return False


def _shutdown(cap, landmarker, mouse, dash):
    try:
        mouse.reset()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass
    try:
        landmarker.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
    dash.update(running=False)
    log.info("Shutdown complete")


def main():
    return run(load_config())


if __name__ == "__main__":
    raise SystemExit(main())
