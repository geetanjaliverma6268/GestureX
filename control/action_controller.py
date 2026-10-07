"""Centralised action controller + control-lock safety system.

:class:`ActionController` is the single choke point through which every computer
action flows. It enforces, in order:

    1. computer control is enabled          (keyboard 'c')
    2. the system is UNLOCKED               (ControlLock safety)
    3. the gesture confidence is above threshold
    4. the intent has been stable for N frames
    5. the specific action is not on cooldown

Only when all five pass does ``execute`` dispatch to a concrete controller. This
is what guarantees no action fires while locked, and no command repeats without
its cooldown elapsing.
"""

from collections import deque

from utils.logger import get_logger

from .media_controller import media_controller
from .presentation_controller import presentation_controller
from .system_controller import (
    brightness_controller,
    desktop_controller,
    screenshot_controller,
    scroll_controller,
    volume_controller,
)

log = get_logger("action")


class ControlLock:
    """Safety lock. The system starts LOCKED; nothing executes until unlocked.

    Unlock/lock is toggled by holding Open Palm steadily for ``hold_frames``; the
    gesture must be released before it can toggle again (edge-triggered).
    """

    def __init__(self, hold_frames=12, min_conf=0.60):
        self.locked = True
        self.hold_frames = hold_frames
        self.min_conf = min_conf
        self._hold = 0
        self._armed = True

    def update(self, open_palm_present: bool, confidence: float = 1.0) -> bool:
        """Feed the current Open-Palm state; returns True on a toggle this frame."""
        if open_palm_present and confidence >= self.min_conf:
            self._hold += 1
            if self._armed and self._hold >= self.hold_frames:
                self.locked = not self.locked
                self._armed = False
                self._hold = 0
                log.info("Control %s (open-palm hold)",
                         "LOCKED" if self.locked else "UNLOCKED")
                return True
        else:
            self._hold = 0
            self._armed = True
        return False

    def toggle(self) -> bool:
        """Manual toggle (keyboard). Returns the new locked state."""
        self.locked = not self.locked
        self._hold = 0
        self._armed = True
        log.info("Control %s (manual)", "LOCKED" if self.locked else "UNLOCKED")
        return self.locked

    def reset(self):
        self._hold = 0
        self._armed = True


class ActionController:
    """Gates and dispatches every computer action."""

    def __init__(self, cfg=None, lock: ControlLock = None):
        g = (cfg.get("gesture", {}) if cfg else {}) or {}
        c = (cfg.get("cooldown", {}) if cfg else {}) or {}
        s = (cfg.get("system", {}) if cfg else {}) or {}
        m = (cfg.get("movement", {}) if cfg else {}) or {}

        self.min_conf = g.get("confidence_threshold", 0.60)
        self.hold_frames = g.get("hold_frames", 8)
        self.default_cooldown = c.get("action_seconds", 1.2)

        self.lock = lock or ControlLock(hold_frames=c.get("lock_hold_frames", 12),
                                        min_conf=self.min_conf)
        self.enabled = True  # computer control master switch ('c' key)

        self.volume_presses = s.get("volume_presses", 2)
        self.brightness_step = s.get("brightness_step", 10)
        self.scroll_steps = m.get("scroll_steps", 3)
        self.screenshot_dir = cfg.path(s.get("screenshot_dir", "screenshots")) \
            if cfg else "screenshots"

        # Per-kind cooldown overrides (seconds).
        self.cooldowns = {
            "screenshot": c.get("screenshot_seconds", 2.0),
            "desktop": c.get("desktop_seconds", 2.0),
            "scroll": c.get("scroll_seconds", 0.30),
        }
        self.last_fire = {}
        self._current = None
        self._hold = 0

        # History for the UI / dashboard.
        self.last_action = ""
        self.last_action_time = 0.0
        self.history = deque(maxlen=50)

    # --- cooldown API ------------------------------------------------------- #
    def _kind(self, key):
        return key[0] if isinstance(key, (tuple, list)) else key

    def set_cooldown(self, key, seconds):
        self.cooldowns[self._kind(key)] = seconds

    def _cooldown_for(self, key):
        return self.cooldowns.get(self._kind(key), self.default_cooldown)

    def is_on_cooldown(self, key, now):
        return (now - self.last_fire.get(key, -1e9)) < self._cooldown_for(key)

    def reset(self):
        self.last_fire.clear()
        self._current = None
        self._hold = 0

    # --- stability + safety gate ------------------------------------------- #
    def request(self, key, confidence, now) -> bool:
        """Return True when ``key`` should fire this frame."""
        if key is None or confidence < self.min_conf or not self.can_act():
            # Reset the stability counter so a gesture cannot be "pre-charged"
            # while locked and then fire the instant the system is unlocked.
            self._current = None
            self._hold = 0
            return False
        if key == self._current:
            self._hold += 1
        else:
            self._current = key
            self._hold = 1

        if self._hold >= self.hold_frames and not self.is_on_cooldown(key, now):
            self.last_fire[key] = now
            self._hold = 0
            return True
        return False

    # --- dispatch ----------------------------------------------------------- #
    def can_act(self) -> bool:
        return self.enabled and not self.lock.locked

    def fire_now(self, key, confidence, now) -> bool:
        """Fire a *dynamic* one-shot event (e.g. two-finger scroll) that is
        already debounced by its own motion detector. Enforces only the safety
        gate (enabled + unlocked), the confidence threshold and the cooldown -
        not the multi-frame stability counter used by static postures."""
        if confidence < self.min_conf or not self.can_act():
            return False
        if self.is_on_cooldown(key, now):
            return False
        self.last_fire[key] = now
        return True

    def execute(self, kind, arg, label=None, gesture=None, now=None):
        """Run an action. Refuses (returns '') while disabled or locked."""
        if not self.can_act():
            return ""
        try:
            result = self._dispatch(kind, arg)
        except Exception as exc:  # never crash the loop
            log.warning("Action error %s/%s: %s", kind, arg, type(exc).__name__)
            return f"Action error ({type(exc).__name__})"
        text = f"{label or kind.title()} [{result}]"
        self.last_action = text
        if now is not None:
            self.last_action_time = now
        self.history.appendleft({
            "time": now, "gesture": gesture or "", "action": label or kind,
            "detail": result,
        })
        log.info("Action executed: %s (%s)", label or kind, result)
        return text

    def _dispatch(self, kind, arg):
        if kind == "media":
            return media_controller(arg)
        if kind == "volume":
            return volume_controller(arg, self.volume_presses)
        if kind == "brightness":
            return brightness_controller(arg, self.brightness_step)
        if kind == "presentation":
            return presentation_controller(arg)
        if kind == "screenshot":
            return screenshot_controller(self.screenshot_dir)
        if kind == "desktop":
            return desktop_controller()
        if kind == "scroll":
            return scroll_controller(arg, self.scroll_steps)
        return "no-op"
