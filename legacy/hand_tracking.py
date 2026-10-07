"""Touchless computer control using MediaPipe Hands (Tasks API).

Builds on the gesture-recognition project to add touchless mouse, scrolling,
double-click, presentation control, two-hand operation, control modes and
application launching. All existing functionality (21-landmark detection,
9-gesture recognition, media/volume/brightness control) is preserved.

Architecture
------------
    Camera -> MediaPipe Hand Detection -> 21 Landmarks -> Gesture Recognition
    -> Gesture Smoothing / Confidence -> Mode Manager -> Command Controller
        |- Media Controller      |- Volume Controller   |- Brightness Controller
        |- Mouse Controller      |- Scroll Controller    |- Presentation Controller
        |- Application Controller

Gesture -> command (NORMAL mode, right hand / media hand)
---------------------------------------------------------
    Thumbs Up -> Play (only)   Thumbs Down -> Stop      Open Palm -> Pause
    Victory   -> Next Track    Closed Fist -> Prev Track
    Pointing Up -> Volume Up   Pinch -> Volume Down
    OK Sign -> Brightness Up   I Love You -> Brightness Down

MOUSE mode (left hand)
----------------------
    Move index finger      -> move cursor
    Pinch thumb+index      -> left click (hold + move = drag, release = drop)
    Two quick pinches      -> double click
    Open Palm + move hand  -> scroll up / down

PRESENTATION mode (left hand)
-----------------------------
    Victory -> next slide    Closed Fist -> prev slide
    Open Palm -> start/stop slideshow (F5 / Esc)

Controls
--------
    m  Mouse mode   p  Presentation mode   n  Normal mode
    c  toggle computer control ON/OFF
    r  reset smoothing / history / cooldown
    q / ESC  quit

Notes / limitations
-------------------
- Thumbs Up triggers Play only; Thumbs Down triggers Stop only. There is no
  Resume command. Windows has a single combined Play/Pause media key
  (VK_MEDIA_PLAY_PAUSE) and no separate Play or Pause virtual key, so Play
  (Thumbs Up) and Pause (Open Palm) both send that toggle; Stop uses
  VK_MEDIA_STOP.
- Right hand is the media hand; left hand is the mouse/presentation hand.
- Brightness needs WMI/DDC-CI support; failures are handled gracefully.
- Application launching is provided by application_controller(); no gesture is
  bound to it by default (see APP_GESTURES) to avoid changing the 9 mappings.

Model
-----
Requires ``hand_landmarker.task`` next to this script. Download it with:
    curl -L -o hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import ctypes
import os
import platform
import subprocess
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

# Optional brightness backend. Imported defensively so the app runs even if the
# library is missing or the display does not support brightness control.
try:
    import screen_brightness_control as sbc
except Exception:  # pragma: no cover - environment dependent
    sbc = None

IS_WINDOWS = platform.system() == "Windows"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")

# Gesture names.
FIST = "Closed Fist"
OPEN_PALM = "Open Palm"
THUMBS_UP = "Thumbs Up"
THUMBS_DOWN = "Thumbs Down"
PEACE = "Victory"
POINTING = "Pointing Up"
OK_SIGN = "OK Sign"
PINCH = "Pinch"
I_LOVE_YOU = "I Love You"
UNKNOWN = "Unknown"

# Modes.
NORMAL = "NORMAL"
MOUSE = "MOUSE"
PRESENTATION = "PRESENTATION"

# Windows virtual-key codes: media keys.
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
# Presentation / navigation keys.
VK_ESCAPE = 0x1B
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_F5 = 0x74

# Mouse SendInput flags.
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120

# Gesture -> (label, kind, arg) for the NORMAL media/volume/brightness set.
GESTURE_ACTIONS = {
    THUMBS_UP:   ("Play",        "media", "play"),
    OPEN_PALM:   ("Pause",       "media", "pause"),
    THUMBS_DOWN: ("Stop",        "media", "stop"),
    PEACE:       ("Next Track",  "media", "next"),
    FIST:        ("Prev Track",  "media", "prev"),
    POINTING:    ("Volume Up",   "volume", "up"),
    PINCH:       ("Volume Down", "volume", "down"),
    OK_SIGN:     ("Brightness Up",   "brightness", "up"),
    I_LOVE_YOU:  ("Brightness Down", "brightness", "down"),
}

# Presentation-mode commands keyed by gesture (resolved in the main loop so the
# Open Palm toggle can track slideshow state).
PRESENT_NEXT = ("Next Slide", "presentation", "next")
PRESENT_PREV = ("Prev Slide", "presentation", "prev")

# Application launcher commands (shell strings). Extend as needed.
APP_COMMANDS = {
    "chrome": 'start "" chrome',
    "vscode": 'start "" code',
    "calculator": 'start "" calc',
    "explorer": 'explorer',
}
# Optional gesture -> app binding. Empty by default so the 9 core mappings are
# not altered. Example: {I_LOVE_YOU: "calculator"}.
APP_GESTURES = {}

# --- Tunables --------------------------------------------------------------- #
BRIGHTNESS_STEP = 10        # percent per brightness command
VOLUME_PRESSES = 2          # OS volume increments per command

# Gesture geometry thresholds.
FINGER_EXTEND_RATIO = 1.10
THUMB_EXTEND_RATIO = 1.05
RATIO_SPREAD = 0.45
OK_PINCH_MAX = 0.28
OK_PINCH_SCALE = 0.45
THUMB_DIR_MIN = 0.10
THUMB_DIR_SCALE = 0.60

# Gesture history smoothing.
GESTURE_WINDOW = 9
GESTURE_MIN_VOTES = 5

# Command triggering (stability + cooldown).
GESTURE_HOLD_FRAMES = 8      # frames a gesture must stay stable before firing
ACTION_COOLDOWN_SEC = 1.2    # min seconds between two fired discrete commands
MIN_ACTION_CONF = 0.60       # ignore predictions below this confidence
ACTION_DISPLAY_SEC = 2.0     # how long the "Action" text stays on screen

# Mouse control.
CURSOR_REGION = (0.15, 0.12, 0.85, 0.88)  # active camera region -> full screen
CURSOR_ALPHA = 0.40         # cursor EMA smoothing (lower = smoother)
CURSOR_DEADZONE = 2         # px; ignore sub-pixel jitter
CLICK_PINCH_MAX = 0.30      # thumb-index pinch ratio that counts as a click
CLICK_DEBOUNCE_FRAMES = 2   # pinch must hold this many frames (anti-accidental)
CLICK_MAX_DUR = 0.5         # s; a pinch shorter than this is a click, else drag
CLICK_MOVE_MAX = 30         # px; movement beyond this while pinched = drag
DOUBLE_CLICK_WINDOW = 0.5   # s between two clicks to count as a double click
DOUBLE_CLICK_RADIUS = 50    # px; clicks must be this close to double-click

# Scroll control.
SCROLL_THRESHOLD = 0.10     # normalized wrist travel to trigger one scroll
SCROLL_COOLDOWN = 0.30      # s between scroll steps
SCROLL_STEPS = 3            # wheel notches per scroll command

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
HAND_COLORS = {"Left": (0, 220, 0), "Right": (255, 120, 0)}
DEFAULT_COLOR = (0, 200, 255)
UNKNOWN_COLOR = (160, 160, 160)
SMOOTHING_ALPHA = 0.55


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) -
                                np.asarray(b, dtype=float)))


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _certainty(ratio, threshold):
    return _clamp(abs(ratio - threshold) / RATIO_SPREAD)


# --------------------------------------------------------------------------- #
# Windows input layer (ctypes SendInput) - keyboard & mouse
# --------------------------------------------------------------------------- #

if IS_WINDOWS:
    _PUL = ctypes.POINTER(ctypes.c_ulong)

    class _KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", _PUL)]

    class _HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]

    class _MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", _PUL)]

    class _InputUnion(ctypes.Union):
        _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput),
                    ("hi", _HardwareInput)]

    class _Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", _InputUnion)]


def _press_vk(vk: int) -> bool:
    """Send a key down+up for a virtual-key code via SendInput (Windows)."""
    if not IS_WINDOWS:
        return False
    KEYEVENTF_KEYUP = 0x0002
    try:
        user32 = ctypes.windll.user32
        for flags in (0, KEYEVENTF_KEYUP):
            extra = ctypes.c_ulong(0)
            union = _InputUnion()
            union.ki = _KeyBdInput(vk, 0, flags, 0, ctypes.pointer(extra))
            cmd = _Input(ctypes.c_ulong(INPUT_KEYBOARD), union)
            user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
        return True
    except Exception:
        return False


def _send_mouse(dx: int, dy: int, data: int, flags: int) -> bool:
    """Send a single mouse input event via SendInput (Windows)."""
    if not IS_WINDOWS:
        return False
    try:
        user32 = ctypes.windll.user32
        extra = ctypes.c_ulong(0)
        union = _InputUnion()
        union.mi = _MouseInput(dx, dy, data & 0xFFFFFFFF, flags, 0,
                               ctypes.pointer(extra))
        cmd = _Input(ctypes.c_ulong(INPUT_MOUSE), union)
        user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
        return True
    except Exception:
        return False


def _screen_size():
    if not IS_WINDOWS:
        return 1920, 1080
    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def _mouse_move_abs(x_px: int, y_px: int) -> bool:
    """Move the cursor to absolute screen pixel coordinates."""
    w, h = _screen_size()
    nx = int(_clamp(x_px / max(w, 1)) * 65535)
    ny = int(_clamp(y_px / max(h, 1)) * 65535)
    return _send_mouse(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)


def _mouse_left_down() -> bool:
    return _send_mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN)


def _mouse_left_up() -> bool:
    return _send_mouse(0, 0, 0, MOUSEEVENTF_LEFTUP)


def _mouse_wheel(delta: int) -> bool:
    return _send_mouse(0, 0, delta, MOUSEEVENTF_WHEEL)


# --------------------------------------------------------------------------- #
# Command controllers
# --------------------------------------------------------------------------- #

def media_controller(command: str) -> str:
    """System-level media control: play | pause | stop | next | prev.

    There is intentionally no "resume" command. Windows exposes a single
    combined Play/Pause media key (VK_MEDIA_PLAY_PAUSE) and no separate Play or
    Pause virtual keys, so both "play" and "pause" send that toggle key. The
    player plays when stopped and pauses when playing.
    """
    vk_map = {"play": VK_MEDIA_PLAY_PAUSE, "pause": VK_MEDIA_PLAY_PAUSE,
              "stop": VK_MEDIA_STOP, "next": VK_MEDIA_NEXT_TRACK,
              "prev": VK_MEDIA_PREV_TRACK}
    vk = vk_map.get(command)
    if vk is None:
        return "Media: unknown command"
    return "Media: sent" if _press_vk(vk) else "Media: unavailable (non-Windows)"


def volume_controller(direction: str, step_presses: int = VOLUME_PRESSES) -> str:
    """System volume: up | down | mute."""
    if direction == "mute":
        return "Volume: mute" if _press_vk(VK_VOLUME_MUTE) else \
            "Volume: unavailable (non-Windows)"
    vk = VK_VOLUME_UP if direction == "up" else VK_VOLUME_DOWN
    sent = False
    for _ in range(max(1, step_presses)):
        sent |= _press_vk(vk)
    if not sent:
        return "Volume: unavailable (non-Windows)"
    return f"Volume: {direction} x{step_presses}"


def brightness_controller(direction: str, step: int = BRIGHTNESS_STEP) -> str:
    """Adjust brightness by ``step`` percent, clamped 0..100, no crash."""
    if sbc is None:
        return "Brightness: library missing"
    try:
        current = sbc.get_brightness()
        cur_val = int(current[0] if isinstance(current, (list, tuple))
                      else current)
        target = cur_val + step if direction == "up" else cur_val - step
        target = int(_clamp(target, 0, 100))
        sbc.set_brightness(target)
        return f"Brightness: {target}% ({direction})"
    except Exception as exc:
        return f"Brightness: unsupported ({type(exc).__name__})"


def mouse_controller(action: str, x: int = 0, y: int = 0) -> str:
    """Low-level mouse ops: move | click | down | up. Used by MouseController."""
    if not IS_WINDOWS:
        return "Mouse: unavailable (non-Windows)"
    if action == "move":
        _mouse_move_abs(int(x), int(y))
        return f"Mouse: move {int(x)},{int(y)}"
    if action == "click":
        _mouse_left_down(); _mouse_left_up()
        return "Mouse: click"
    if action == "down":
        _mouse_left_down(); return "Mouse: down"
    if action == "up":
        _mouse_left_up(); return "Mouse: up"
    return "Mouse: unknown action"


def scroll_controller(direction: str, steps: int = SCROLL_STEPS) -> str:
    """Scroll the mouse wheel up/down by ``steps`` notches."""
    if not IS_WINDOWS:
        return "Scroll: unavailable (non-Windows)"
    delta = WHEEL_DELTA * max(1, steps) * (1 if direction == "up" else -1)
    return f"Scroll: {direction} x{steps}" if _mouse_wheel(delta) else \
        "Scroll: failed"


def presentation_controller(command: str) -> str:
    """Presentation control: next | prev | start | stop."""
    vk_map = {"next": VK_RIGHT, "prev": VK_LEFT,
              "start": VK_F5, "stop": VK_ESCAPE}
    vk = vk_map.get(command)
    if vk is None:
        return "Presentation: unknown command"
    return "Presentation: sent" if _press_vk(vk) else \
        "Presentation: unavailable (non-Windows)"


def application_controller(app: str) -> str:
    """Launch a common application by name. Never raises."""
    cmd = APP_COMMANDS.get(str(app).lower())
    if not cmd:
        return f"App: unknown '{app}'"
    if not IS_WINDOWS:
        return "App: unavailable (non-Windows)"
    try:
        subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"App: launched {app}"
    except Exception as exc:
        return f"App: failed ({type(exc).__name__})"


def _dispatch(kind: str, arg: str) -> str:
    if kind == "media":
        return media_controller(arg)
    if kind == "volume":
        return volume_controller(arg)
    if kind == "brightness":
        return brightness_controller(arg)
    if kind == "presentation":
        return presentation_controller(arg)
    if kind == "application":
        return application_controller(arg)
    if kind == "scroll":
        return scroll_controller(arg)
    if kind == "mouse":
        return mouse_controller(arg)
    return "no-op"


def execute_action(command) -> str:
    """Run a command. Accepts a gesture string (NORMAL set) or a
    (kind, arg, label) tuple. Returns a human-readable result and never raises.
    """
    try:
        if isinstance(command, str):
            if command in APP_GESTURES:
                kind, arg, label = "application", APP_GESTURES[command], \
                    f"Open {APP_GESTURES[command]}"
            elif command in GESTURE_ACTIONS:
                label, kind, arg = GESTURE_ACTIONS[command]
            else:
                return ""
        else:
            kind, arg, label = command
        return f"{label} [{_dispatch(kind, arg)}]"
    except Exception as exc:
        return f"Action error ({type(exc).__name__})"


# --------------------------------------------------------------------------- #
# Cooldown / stability gate
# --------------------------------------------------------------------------- #

class CommandGate:
    """Enforces 'stable for N frames' + a cooldown before a command fires."""

    def __init__(self, hold_frames=GESTURE_HOLD_FRAMES,
                 cooldown=ACTION_COOLDOWN_SEC):
        self.hold_frames = hold_frames
        self.cooldown = cooldown
        self.current = None
        self.hold = 0
        self.last_fire = -cooldown

    def request(self, cmd, now: float) -> bool:
        if cmd is None:
            self.current = None
            self.hold = 0
            return False
        if cmd == self.current:
            self.hold += 1
        else:
            self.current = cmd
            self.hold = 1
        if self.hold >= self.hold_frames and (now - self.last_fire) >= self.cooldown:
            self.last_fire = now
            self.hold = 0
            return True
        return False

    def reset(self):
        self.current = None
        self.hold = 0
        self.last_fire = -self.cooldown


def handle_cooldown(gate: CommandGate, cmd, confidence: float, now: float) -> bool:
    """Gate a command by confidence threshold, stability, and cooldown."""
    if cmd is None or confidence < MIN_ACTION_CONF:
        gate.request(None, now)
        return False
    return gate.request(cmd, now)


# --------------------------------------------------------------------------- #
# FPS
# --------------------------------------------------------------------------- #

class FpsCounter:
    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else \
                self._alpha * inst + (1 - self._alpha) * self._fps
        return self._fps


# --------------------------------------------------------------------------- #
# Landmark tracking & smoothing
# --------------------------------------------------------------------------- #

class LandmarkSmoother:
    """Per-hand EMA over landmark pixel coordinates to reduce jitter."""

    def __init__(self, alpha=SMOOTHING_ALPHA):
        self._alpha = alpha
        self._history = {}

    def smooth(self, label, points):
        prev = self._history.get(label)
        if prev is None or prev.shape != points.shape:
            smoothed = points.copy()
        else:
            smoothed = self._alpha * points + (1 - self._alpha) * prev
        self._history[label] = smoothed
        return smoothed

    def prune(self, active_labels):
        for label in list(self._history.keys()):
            if label not in active_labels:
                del self._history[label]

    def reset(self):
        self._history.clear()


def smooth_landmarks(smoother: LandmarkSmoother, label: str, points):
    return smoother.smooth(label, points)


def track_landmarks(result, hand_index, width, height, smoother):
    """Get one hand's landmarks and apply temporal smoothing."""
    raw, label, score = get_landmarks(result, hand_index, width, height)
    return smooth_landmarks(smoother, label, raw), label, score


# --------------------------------------------------------------------------- #
# Gesture history smoothing
# --------------------------------------------------------------------------- #

class GestureSmoother:
    """Majority vote over a rolling window so labels do not flicker."""

    def __init__(self, window=GESTURE_WINDOW, min_votes=GESTURE_MIN_VOTES):
        self._window = window
        self._min_votes = min_votes
        self._history = {}

    def push(self, label, name, conf):
        self._history.setdefault(label, deque(maxlen=self._window)) \
            .append((name, conf))

    def stable(self, label, fallback_name, fallback_conf):
        hist = self._history.get(label)
        if not hist:
            return fallback_name, fallback_conf
        counts = Counter(n for n, _ in hist)
        top_name, top_votes = counts.most_common(1)[0]
        name = top_name if top_votes >= self._min_votes else fallback_name
        confs = [c for n, c in hist if n == name] or [fallback_conf]
        return name, float(sum(confs) / len(confs))

    def prune(self, active_labels):
        for label in list(self._history.keys()):
            if label not in active_labels:
                del self._history[label]

    def reset(self):
        self._history.clear()


def smooth_gesture(smoother, label, name, conf):
    smoother.push(label, name, conf)
    return smoother.stable(label, name, conf)


# --------------------------------------------------------------------------- #
# Finger states, confidence, gesture recognition
# --------------------------------------------------------------------------- #

def _safe_ratio(tip, pip, wrist):
    denom = _distance(pip, wrist)
    return 0.0 if denom < 1e-6 else _distance(tip, wrist) / denom


def calculate_finger_states(points):
    wrist = points[WRIST]
    ratios = {
        "thumb": _safe_ratio(points[THUMB_TIP], points[THUMB_IP], wrist),
        "index": _safe_ratio(points[INDEX_TIP], points[INDEX_PIP], wrist),
        "middle": _safe_ratio(points[MIDDLE_TIP], points[MIDDLE_PIP], wrist),
        "ring": _safe_ratio(points[RING_TIP], points[RING_PIP], wrist),
        "pinky": _safe_ratio(points[PINKY_TIP], points[PINKY_PIP], wrist),
    }
    states = {
        "thumb": ratios["thumb"] > THUMB_EXTEND_RATIO,
        "index": ratios["index"] > FINGER_EXTEND_RATIO,
        "middle": ratios["middle"] > FINGER_EXTEND_RATIO,
        "ring": ratios["ring"] > FINGER_EXTEND_RATIO,
        "pinky": ratios["pinky"] > FINGER_EXTEND_RATIO,
    }
    return {"states": states, "ratios": ratios}


def count_fingers(finger_info) -> int:
    return sum(1 for k in FINGER_KEYS if finger_info["states"][k])


def _derived(points, finger_info):
    """Compute shared geometry used by classification and confidence."""
    st = finger_info["states"]
    ratios = finger_info["ratios"]
    hand_size = max(_distance(points[MIDDLE_MCP], points[WRIST]), 1e-6)
    pinch = _distance(points[THUMB_TIP], points[INDEX_TIP]) / hand_size
    vertical = (points[WRIST][1] - points[THUMB_TIP][1]) / hand_size
    cert = {k: _certainty(ratios[k],
                          THUMB_EXTEND_RATIO if k == "thumb" else FINGER_EXTEND_RATIO)
            for k in FINGER_KEYS}
    fingers_ext = st["index"] + st["middle"] + st["ring"] + st["pinky"]
    return st, cert, hand_size, pinch, vertical, fingers_ext


def _classify(points, finger_info) -> str:
    st, _, _, pinch, vertical, fingers_ext = _derived(points, finger_info)
    thumb, index, middle, ring, pinky = (st[k] for k in FINGER_KEYS)

    if pinch < OK_PINCH_MAX and middle and ring and pinky:
        return OK_SIGN
    if thumb and pinch < OK_PINCH_MAX and not (middle and ring and pinky):
        return PINCH
    if thumb and fingers_ext == 0 and abs(vertical) > THUMB_DIR_MIN:
        return THUMBS_UP if vertical > 0 else THUMBS_DOWN
    if fingers_ext == 0 and not thumb:
        return FIST
    if fingers_ext == 4 and thumb:
        return OPEN_PALM
    if thumb and index and pinky and not middle and not ring:
        return I_LOVE_YOU
    if index and middle and not ring and not pinky:
        return PEACE
    if index and not middle and not ring and not pinky:
        return POINTING
    return UNKNOWN


def calculate_confidence(name: str, points, finger_info) -> float:
    """Geometric confidence in [0,1] for a classified gesture."""
    if name == UNKNOWN:
        return 0.0
    _, cert, _, pinch, vertical, _ = _derived(points, finger_info)

    def avg(keys):
        return sum(cert[k] for k in keys) / len(keys)

    if name in (OK_SIGN, PINCH):
        pinch_conf = _clamp(1.0 - pinch / OK_PINCH_SCALE)
        base = avg(("middle", "ring", "pinky")) if name == OK_SIGN \
            else avg(("thumb",))
        return _clamp(0.5 * base + 0.5 * pinch_conf, 0.4, 1.0)
    if name in (THUMBS_UP, THUMBS_DOWN):
        dir_conf = _clamp(abs(vertical) / THUMB_DIR_SCALE)
        return _clamp(0.5 * avg(("index", "middle", "ring", "pinky")) +
                      0.5 * dir_conf, 0.4, 1.0)
    if name in (FIST, OPEN_PALM, I_LOVE_YOU):
        return _clamp(avg(FINGER_KEYS), 0.4, 1.0)
    if name in (PEACE, POINTING):
        return _clamp(avg(("index", "middle", "ring", "pinky")), 0.4, 1.0)
    return 0.0


def recognize_gesture(points, finger_info):
    """Geometry-based classifier. Returns (name, confidence in [0,1])."""
    name = _classify(points, finger_info)
    return name, calculate_confidence(name, points, finger_info)


def detect_gesture(points):
    """Landmarks -> {gesture, confidence, fingers, finger_info}."""
    finger_info = calculate_finger_states(points)
    gesture, confidence = recognize_gesture(points, finger_info)
    return {"gesture": gesture, "confidence": confidence,
            "fingers": count_fingers(finger_info), "finger_info": finger_info}


# --------------------------------------------------------------------------- #
# Detection wrappers
# --------------------------------------------------------------------------- #

def build_hand_landmarker(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def get_landmarks(result, hand_index, width, height):
    landmarks = result.hand_landmarks[hand_index]
    points = np.zeros((len(landmarks), 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        points[i] = (lm.x * width, lm.y * height)
    handedness = result.handedness[hand_index][0]
    return points, handedness.category_name, float(handedness.score)


# --------------------------------------------------------------------------- #
# Mode manager
# --------------------------------------------------------------------------- #

class ModeManager:
    """Holds the active control mode and handles mode-switch keys."""

    KEY_MAP = {"m": MOUSE, "p": PRESENTATION, "n": NORMAL}

    def __init__(self):
        self.mode = NORMAL

    def handle_key(self, key_char: str) -> bool:
        new = self.KEY_MAP.get(str(key_char).lower())
        if new is None:
            return False
        self.mode = new
        return True


def mode_manager(manager: ModeManager, key_char: str) -> str:
    """Apply a mode key; returns the (possibly unchanged) current mode."""
    manager.handle_key(key_char)
    return manager.mode


# --------------------------------------------------------------------------- #
# Mouse controller (cursor / click / drag / double-click state machine)
# --------------------------------------------------------------------------- #

class MouseController:
    """Maps the index fingertip to the screen and turns pinch into clicks.

    - Cursor position is EMA-smoothed with a deadzone to avoid shaking.
    - Pinch onset  -> LEFT button down.
    - Pinch hold + movement -> drag.
    - Pinch release -> LEFT up (short/low-move = click, else drop).
    - Two quick clicks at ~same spot -> double click (also delivered to the OS
      because real down/up events are sent at the same position).
    """

    def __init__(self, alpha=CURSOR_ALPHA, deadzone=CURSOR_DEADZONE):
        self.alpha = alpha
        self.deadzone = deadzone
        self.sx = None
        self.sy = None
        self.button_down = False
        self.down_time = 0.0
        self.down_pos = (0, 0)
        self.pinch_frames = 0
        self.last_click_time = -1e9
        self.last_click_pos = (0, 0)

    def _target(self, nx, ny, w, h):
        x0, y0, x1, y1 = CURSOR_REGION
        rx = _clamp((nx - x0) / (x1 - x0))
        ry = _clamp((ny - y0) / (y1 - y0))
        return rx * w, ry * h

    def release_if_down(self, now):
        if self.button_down:
            _mouse_left_up()
            self.button_down = False

    def reset(self, now=0.0):
        self.release_if_down(now)
        self.sx = self.sy = None
        self.pinch_frames = 0

    def update(self, present, nx, ny, pinch_ratio, now, w, h):
        ev = {"cursor": None, "clicked": False, "double": False,
              "dragging": False, "dropped": False}
        if not present:
            self.release_if_down(now)
            self.sx = self.sy = None
            return ev

        tx, ty = self._target(nx, ny, w, h)
        if self.sx is None:
            self.sx, self.sy = tx, ty
        else:
            cx = self.alpha * tx + (1 - self.alpha) * self.sx
            cy = self.alpha * ty + (1 - self.alpha) * self.sy
            if abs(cx - self.sx) > self.deadzone or abs(cy - self.sy) > self.deadzone:
                self.sx, self.sy = cx, cy
        _mouse_move_abs(int(self.sx), int(self.sy))
        ev["cursor"] = (int(self.sx), int(self.sy))

        if pinch_ratio < CLICK_PINCH_MAX:
            self.pinch_frames += 1
        else:
            self.pinch_frames = 0
        stable_pinch = self.pinch_frames >= CLICK_DEBOUNCE_FRAMES

        if stable_pinch and not self.button_down:
            _mouse_left_down()
            self.button_down = True
            self.down_time = now
            self.down_pos = (self.sx, self.sy)
        elif (not stable_pinch) and self.button_down:
            _mouse_left_up()
            self.button_down = False
            dur = now - self.down_time
            moved = _distance((self.sx, self.sy), self.down_pos)
            pos = (self.sx, self.sy)
            if dur <= CLICK_MAX_DUR and moved <= CLICK_MOVE_MAX:
                ev["clicked"] = True
                if (now - self.last_click_time) <= DOUBLE_CLICK_WINDOW and \
                        _distance(pos, self.last_click_pos) <= DOUBLE_CLICK_RADIUS:
                    ev["double"] = True
                    self.last_click_time = -1e9
                else:
                    self.last_click_time = now
                    self.last_click_pos = pos
            else:
                ev["dropped"] = True
        if self.button_down:
            ev["dragging"] = True
        return ev


# --------------------------------------------------------------------------- #
# Scroll controller (vertical hand motion while in the scroll posture)
# --------------------------------------------------------------------------- #

class ScrollController:
    """Triggers discrete scroll steps when the wrist travels past a threshold."""

    def __init__(self, threshold=SCROLL_THRESHOLD, cooldown=SCROLL_COOLDOWN):
        self.threshold = threshold
        self.cooldown = cooldown
        self.ref = None
        self.last = -1e9

    def reset_ref(self):
        self.ref = None

    def update(self, wrist_y_norm, now):
        if self.ref is None:
            self.ref = wrist_y_norm
            return None
        dy = self.ref - wrist_y_norm  # positive => hand moved up
        if abs(dy) < self.threshold:
            return None
        self.ref = wrist_y_norm
        if (now - self.last) < self.cooldown:
            return None
        self.last = now
        return "up" if dy > 0 else "down"


# --------------------------------------------------------------------------- #
# Drawing / UI
# --------------------------------------------------------------------------- #

def _draw_hand_skeleton(frame, points, color):
    for conn in HAND_CONNECTIONS:
        start = tuple(points[conn.start].astype(int))
        end = tuple(points[conn.end].astype(int))
        cv2.line(frame, start, end, color, 2)
    tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    for idx, (x, y) in enumerate(points.astype(int)):
        radius = 6 if idx in tips else 4
        cv2.circle(frame, (int(x), int(y)), radius, color, cv2.FILLED)
        cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)


def _draw_text_box(frame, text, center_x, top_y, color, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(2, min(center_x - tw // 2, frame.shape[1] - tw - 4))
    y = max(th + 8, top_y)
    cv2.rectangle(frame, (x - 5, y - th - 6), (x + tw + 5, y + 5),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _hand_role(label, mode):
    if label == "Right":
        return "Media"
    # Left hand role depends on mode.
    if mode == MOUSE:
        return "Mouse"
    if mode == PRESENTATION:
        return "Slides"
    return "Media*"  # acts as media only when the right hand is absent


def draw_ui(frame, hands_by_label, fps, mode, control_on, last_action,
            show_action, cursor_pos=None):
    """Draw per-hand overlays and the global status panel."""
    for label, h in hands_by_label.items():
        pts = h["points"]
        _draw_hand_skeleton(frame, pts, h["color"])
        wx, wy = pts[WRIST].astype(int)
        _draw_text_box(frame, f'{label} {h["score"]:.2f}', wx, wy + 18,
                       h["color"], scale=0.6)
        top_y = int(min(p[1] for p in pts)) - 14
        gcolor = h["color"] if h["gesture"] != UNKNOWN else UNKNOWN_COLOR
        _draw_text_box(frame,
                       f'{h["gesture"]}  {int(h["confidence"]*100)}%',
                       wx, top_y, gcolor, scale=0.75)
        # In mouse mode mark the controlling fingertip.
        if label == "Left" and mode == MOUSE:
            tx, ty = pts[INDEX_TIP].astype(int)
            cv2.circle(frame, (tx, ty), 12, (0, 255, 255), 2)

    lines = [
        f"FPS: {fps:5.1f}   Mode: {mode}   Control: {'ON' if control_on else 'OFF'}",
        f"Action: {last_action if show_action else '-'}",
    ]
    for label in ("Left", "Right"):
        h = hands_by_label.get(label)
        if h:
            lines.append(f'{label} Hand: {h["gesture"]} '
                         f'({int(h["confidence"]*100)}%) {h["fingers"]}f '
                         f'[{_hand_role(label, mode)}]')
        else:
            lines.append(f'{label} Hand: --- (not detected)')
    if cursor_pos:
        lines.append(f"Cursor: {cursor_pos[0]}, {cursor_pos[1]}")
    lines.append("[C]ontrol [M]ouse [P]resent [N]ormal [R]eset [Q]uit")

    x, y0, lh, w = 12, 26, 26, 560
    h_panel = y0 + lh * (len(lines) - 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, 6), (x + w, h_panel), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        col = (255, 255, 255)
        if i == 0:
            col = (0, 255, 0) if control_on else (0, 160, 255)
        cv2.putText(frame, line, (x, y0 + i * lh), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0). Check that a camera is "
              "connected and not in use by another application.", file=sys.stderr)
        return 1

    try:
        landmarker = build_hand_landmarker()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        cap.release()
        return 1

    fps_counter = FpsCounter()
    landmark_smoother = LandmarkSmoother()
    gesture_smoother = GestureSmoother()
    media_gate = CommandGate()
    present_gate = CommandGate()
    mouse = MouseController()
    scroller = ScrollController()
    mode_mgr = ModeManager()

    control_on = True
    slideshow_active = False
    last_action = ""
    last_action_time = 0.0
    cursor_pos = None
    prev_mode = mode_mgr.mode

    print("Touchless control started. Modes: [m]ouse [p]resentation [n]ormal. "
          "[c] toggle control, [r] reset, [q]/ESC quit.")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame from webcam.", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
        height, width = frame.shape[:2]
        now = time.time()

        try:
            timestamp_ms = int(time.monotonic() * 1000) + frame_count
            frame_count += 1
            result = detect_hands(landmarker, frame, timestamp_ms)

            hands_by_label = {}
            active_labels = []
            for idx in range(len(result.hand_landmarks)):
                pts, label, score = track_landmarks(result, idx, width, height,
                                                    landmark_smoother)
                active_labels.append(label)
                det = detect_gesture(pts)
                gesture, conf = smooth_gesture(gesture_smoother, label,
                                               det["gesture"], det["confidence"])
                # Keep the most confident instance if a label repeats.
                if label not in hands_by_label or conf > hands_by_label[label]["confidence"]:
                    hands_by_label[label] = {
                        "points": pts, "color": HAND_COLORS.get(label, DEFAULT_COLOR),
                        "score": score, "fingers": det["fingers"],
                        "gesture": gesture, "confidence": conf,
                    }
            landmark_smoother.prune(active_labels)
            gesture_smoother.prune(active_labels)

            left = hands_by_label.get("Left")
            right = hands_by_label.get("Right")
            mode = mode_mgr.mode

            def set_action(text):
                nonlocal last_action, last_action_time
                last_action = text
                last_action_time = now

            # --- Media channel: right hand (or left in NORMAL if right absent) --
            media_hand = right if right else (left if mode == NORMAL else None)
            if control_on and media_hand:
                cmd = GESTURE_ACTIONS.get(media_hand["gesture"])
                app = APP_GESTURES.get(media_hand["gesture"])
                if app:
                    key = ("application", app)
                elif cmd:
                    key = (cmd[1], cmd[2])
                else:
                    key = None
                if handle_cooldown(media_gate, key, media_hand["confidence"], now):
                    if app:
                        set_action(execute_action(("application", app, f"Open {app}")))
                    else:
                        set_action(execute_action(media_hand["gesture"]))
            else:
                media_gate.request(None, now)

            # --- Mode channel: left hand ----------------------------------------
            cursor_pos = None
            if control_on and left and mode == PRESENTATION:
                mouse.reset(now)
                scroller.reset_ref()
                g = left["gesture"]
                if g == PEACE:
                    pcmd = PRESENT_NEXT
                elif g == FIST:
                    pcmd = PRESENT_PREV
                elif g == OPEN_PALM:
                    arg = "stop" if slideshow_active else "start"
                    pcmd = ("Start/Stop Presentation", "presentation", arg)
                else:
                    pcmd = None
                key = (pcmd[1], pcmd[2]) if pcmd else None
                if handle_cooldown(present_gate, key, left["confidence"], now):
                    set_action(execute_action(pcmd))
                    if pcmd and pcmd[2] in ("start", "stop"):
                        slideshow_active = (pcmd[2] == "start")
            elif control_on and left and mode == MOUSE:
                present_gate.request(None, now)
                pts = left["points"]
                hand_size = max(_distance(pts[MIDDLE_MCP], pts[WRIST]), 1e-6)
                pinch_ratio = _distance(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size
                wrist_y = pts[WRIST][1] / height
                nx = pts[INDEX_TIP][0] / width
                ny = pts[INDEX_TIP][1] / height

                if left["gesture"] == OPEN_PALM and left["confidence"] >= MIN_ACTION_CONF:
                    # Scroll posture: pause cursor, release any held button.
                    mouse.release_if_down(now)
                    direction = scroller.update(wrist_y, now)
                    if direction:
                        set_action(f"Scroll {direction.capitalize()} "
                                   f"[{scroll_controller(direction)}]")
                else:
                    scroller.reset_ref()
                    ev = mouse.update(True, nx, ny, pinch_ratio, now,
                                      *_screen_size())
                    cursor_pos = ev["cursor"]
                    if ev["double"]:
                        set_action("Double Click [mouse]")
                    elif ev["clicked"]:
                        set_action("Left Click [mouse]")
                    elif ev["dropped"]:
                        set_action("Drop [mouse]")
            else:
                # Normal mode left hand (when right present) or no left hand.
                present_gate.request(None, now)
                if mode == MOUSE:
                    mouse.update(False, 0, 0, 1.0, now, *_screen_size())
                else:
                    mouse.reset(now)
                scroller.reset_ref()

            fps = fps_counter.tick()
            draw_ui(frame, hands_by_label, fps, mode, control_on, last_action,
                    (now - last_action_time) < ACTION_DISPLAY_SEC, cursor_pos)
        except Exception as exc:
            # Never crash the loop because of a gesture/action failure.
            print(f"WARNING: frame processing error ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            draw_ui(frame, {}, fps_counter.tick(), mode_mgr.mode, control_on,
                    last_action, False, None)

        cv2.imshow("Touchless Computer Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("c"), ord("C")):
            control_on = not control_on
            media_gate.reset(); present_gate.reset(); mouse.reset(time.time())
            print(f"Computer control {'ENABLED' if control_on else 'DISABLED'}.")
        elif key in (ord("r"), ord("R")):
            landmark_smoother.reset(); gesture_smoother.reset()
            media_gate.reset(); present_gate.reset(); mouse.reset(time.time())
            scroller.reset_ref()
            print("Smoothing, history, and cooldowns reset.")
        else:
            # Mode-switch keys: m / p / n (ignored for any other key).
            ch = chr(key) if 32 <= key < 127 else ""
            if ch and mode_manager(mode_mgr, ch) != prev_mode:
                print(f"Mode -> {mode_mgr.mode}")
            prev_mode = mode_mgr.mode

    mouse.reset(time.time())
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
