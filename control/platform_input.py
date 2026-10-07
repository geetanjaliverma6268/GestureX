"""Low-level OS input layer (Windows SendInput via ctypes) + screenshot capture.

Isolated so the higher-level controllers (media, mouse, system) never call the
Win32 API directly, and so the whole layer can be mocked in tests. Every function
degrades gracefully and returns a bool instead of raising on unsupported
platforms.
"""

import ctypes
import os
import platform
import time

IS_WINDOWS = platform.system() == "Windows"

# --- Virtual-key codes ------------------------------------------------------ #
VK_ESCAPE = 0x1B
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_F5 = 0x74
VK_LWIN = 0x5B
VK_D = 0x44
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

# --- SendInput constants ---------------------------------------------------- #
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120

# Allow tests / non-Windows hosts to neutralise all real side effects.
ENABLED = True


def set_enabled(flag: bool):
    """Globally enable/disable real OS input (used by the test-suite)."""
    global ENABLED
    ENABLED = bool(flag)


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
        _fields_ = [("ki", _KeyBdInput), ("mi", _MouseInput), ("hi", _HardwareInput)]

    class _Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", _InputUnion)]


def _clamp01(v):
    return max(0.0, min(1.0, v))


def press_vk(vk: int) -> bool:
    """Send a key down+up for a virtual-key code."""
    if not IS_WINDOWS or not ENABLED:
        return False
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


def send_combo(*vks) -> bool:
    """Press a key combination (all down in order, then up in reverse)."""
    if not IS_WINDOWS or not ENABLED or not vks:
        return False
    ok = True
    try:
        for vk in vks:
            ok &= _raw_key(vk, 0)
        for vk in reversed(vks):
            ok &= _raw_key(vk, KEYEVENTF_KEYUP)
        return ok
    except Exception:
        return False


def _raw_key(vk, flags) -> bool:
    user32 = ctypes.windll.user32
    extra = ctypes.c_ulong(0)
    union = _InputUnion()
    union.ki = _KeyBdInput(vk, 0, flags, 0, ctypes.pointer(extra))
    cmd = _Input(ctypes.c_ulong(INPUT_KEYBOARD), union)
    user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
    return True


def _send_mouse(dx: int, dy: int, data: int, flags: int) -> bool:
    if not IS_WINDOWS or not ENABLED:
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


def screen_size():
    if not IS_WINDOWS:
        return 1920, 1080
    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def mouse_move_abs(x_px: int, y_px: int) -> bool:
    w, h = screen_size()
    nx = int(_clamp01(x_px / max(w, 1)) * 65535)
    ny = int(_clamp01(y_px / max(h, 1)) * 65535)
    return _send_mouse(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)


def mouse_left_down() -> bool:
    return _send_mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN)


def mouse_left_up() -> bool:
    return _send_mouse(0, 0, 0, MOUSEEVENTF_LEFTUP)


def mouse_wheel(delta: int) -> bool:
    return _send_mouse(0, 0, delta, MOUSEEVENTF_WHEEL)


def capture_screen(save_path: str) -> str:
    """Grab the full screen and save to ``save_path``. Returns the path on
    success, or an empty string if capture is unavailable."""
    if not ENABLED:
        return ""
    try:
        from PIL import ImageGrab
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img = ImageGrab.grab()
        img.save(save_path)
        return save_path
    except Exception:
        return ""


def timestamped_name(prefix="screenshot", ext="png", when=None):
    when = when or time.localtime()
    return f"{prefix}_{time.strftime('%Y-%m-%d_%H-%M-%S', when)}.{ext}"
