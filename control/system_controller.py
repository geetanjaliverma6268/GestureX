"""System controls: volume, brightness, screenshot and show-desktop."""

import os

from . import platform_input as pin
from utils.logger import get_logger

log = get_logger("system")

try:
    import screen_brightness_control as sbc
except Exception:  # pragma: no cover - environment dependent
    sbc = None


def volume_controller(direction: str, step_presses: int = 2) -> str:
    """System volume: up | down | mute."""
    direction = str(direction).lower()
    if direction == "mute":
        ok = pin.press_vk(pin.VK_VOLUME_MUTE)
        log.info("Volume: mute (%s)", "sent" if ok else "unavailable")
        return "Volume: mute" if ok else "Volume: unavailable (non-Windows)"
    vk = pin.VK_VOLUME_UP if direction == "up" else pin.VK_VOLUME_DOWN
    sent = False
    for _ in range(max(1, step_presses)):
        sent |= pin.press_vk(vk)
    if not sent:
        return "Volume: unavailable (non-Windows)"
    log.info("Volume: %s x%d", direction, step_presses)
    return f"Volume: {direction} x{step_presses}"


def brightness_controller(direction: str, step: int = 10) -> str:
    """Adjust brightness by ``step`` percent, clamped 0..100. Never raises."""
    if sbc is None:
        return "Brightness: library missing"
    try:
        current = sbc.get_brightness()
        cur_val = int(current[0] if isinstance(current, (list, tuple)) else current)
        target = cur_val + step if direction == "up" else cur_val - step
        target = int(max(0, min(100, target)))
        sbc.set_brightness(target)
        log.info("Brightness: %d%% (%s)", target, direction)
        return f"Brightness: {target}% ({direction})"
    except Exception as exc:
        log.warning("Brightness unsupported: %s", type(exc).__name__)
        return f"Brightness: unsupported ({type(exc).__name__})"


def screenshot_controller(directory: str = "screenshots") -> str:
    """Capture the full screen to ``directory`` with a timestamped filename."""
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, pin.timestamped_name())
        saved = pin.capture_screen(path)
        if saved:
            log.info("Screenshot saved: %s", saved)
            return f"Screenshot: {os.path.basename(saved)}"
        return "Screenshot: capture unavailable"
    except Exception as exc:
        return f"Screenshot: failed ({type(exc).__name__})"


def desktop_controller() -> str:
    """Show the desktop (Windows: Win+D)."""
    ok = pin.send_combo(pin.VK_LWIN, pin.VK_D)
    log.info("Show desktop (%s)", "sent" if ok else "unavailable")
    return "Show Desktop: sent" if ok else "Show Desktop: unavailable (non-Windows)"


def scroll_controller(direction: str, steps: int = 3) -> str:
    """Scroll the mouse wheel up/down by ``steps`` notches."""
    delta = pin.WHEEL_DELTA * max(1, steps) * (1 if direction == "up" else -1)
    ok = pin.mouse_wheel(delta)
    if ok:
        log.info("Scroll: %s x%d", direction, steps)
        return f"Scroll: {direction} x{steps}"
    return "Scroll: unavailable (non-Windows)"
