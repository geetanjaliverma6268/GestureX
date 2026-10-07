"""Virtual mouse (cursor / click / drag / double-click) and two-finger scroll.

Both are state machines that convert smoothed landmark geometry into discrete,
debounced OS events. Neither fires while the system is locked - that gate lives
in :class:`control.action_controller.ActionController`.
"""

from gesture.direction_detector import DirectionDetector
from hand_tracking.landmarks import clamp, distance

from . import platform_input as pin


class MouseController:
    """Maps the index fingertip to the screen and turns pinch into clicks.

    - Cursor position is EMA-smoothed with a deadzone to avoid shaking.
    - Pinch onset  -> LEFT button down.
    - Pinch hold + movement -> drag.
    - Pinch release -> LEFT up (short/low-move = click, else drop).
    - Two quick clicks at ~same spot -> double click.
    """

    def __init__(self, cursor_region=(0.15, 0.12, 0.85, 0.88), alpha=0.40,
                 deadzone=2, click_pinch_max=0.30, click_debounce_frames=2,
                 click_max_duration=0.5, click_move_max=30,
                 double_click_window=0.5, double_click_radius=50):
        self.cursor_region = tuple(cursor_region)
        self.alpha = alpha
        self.deadzone = deadzone
        self.click_pinch_max = click_pinch_max
        self.click_debounce_frames = click_debounce_frames
        self.click_max_duration = click_max_duration
        self.click_move_max = click_move_max
        self.double_click_window = double_click_window
        self.double_click_radius = double_click_radius

        self.sx = None
        self.sy = None
        self.button_down = False
        self.down_time = 0.0
        self.down_pos = (0, 0)
        self.pinch_frames = 0
        self.last_click_time = -1e9
        self.last_click_pos = (0, 0)

    @classmethod
    def from_config(cls, cfg):
        m = cfg.get("movement", {}) or {}
        return cls(
            cursor_region=m.get("cursor_region", (0.15, 0.12, 0.85, 0.88)),
            alpha=m.get("cursor_alpha", 0.40),
            deadzone=m.get("cursor_deadzone", 2),
            click_pinch_max=m.get("click_pinch_max", 0.30),
            click_debounce_frames=m.get("click_debounce_frames", 2),
            click_max_duration=m.get("click_max_duration", 0.5),
            click_move_max=m.get("click_move_max", 30),
            double_click_window=m.get("double_click_window", 0.5),
            double_click_radius=m.get("double_click_radius", 50),
        )

    def _target(self, nx, ny, w, h):
        x0, y0, x1, y1 = self.cursor_region
        rx = clamp((nx - x0) / (x1 - x0))
        ry = clamp((ny - y0) / (y1 - y0))
        return rx * w, ry * h

    def release_if_down(self):
        if self.button_down:
            pin.mouse_left_up()
            self.button_down = False

    def reset(self):
        self.release_if_down()
        self.sx = self.sy = None
        self.pinch_frames = 0

    def update(self, present, nx, ny, pinch_ratio, now, w, h):
        ev = {"cursor": None, "clicked": False, "double": False,
              "dragging": False, "dropped": False}
        if not present:
            self.release_if_down()
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
        pin.mouse_move_abs(int(self.sx), int(self.sy))
        ev["cursor"] = (int(self.sx), int(self.sy))

        if pinch_ratio < self.click_pinch_max:
            self.pinch_frames += 1
        else:
            self.pinch_frames = 0
        stable_pinch = self.pinch_frames >= self.click_debounce_frames

        if stable_pinch and not self.button_down:
            pin.mouse_left_down()
            self.button_down = True
            self.down_time = now
            self.down_pos = (self.sx, self.sy)
        elif (not stable_pinch) and self.button_down:
            pin.mouse_left_up()
            self.button_down = False
            dur = now - self.down_time
            moved = distance((self.sx, self.sy), self.down_pos)
            pos = (self.sx, self.sy)
            if dur <= self.click_max_duration and moved <= self.click_move_max:
                ev["clicked"] = True
                if (now - self.last_click_time) <= self.double_click_window and \
                        distance(pos, self.last_click_pos) <= self.double_click_radius:
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


class ScrollController:
    """Two-finger directional scroll.

    A *static* two-finger posture must not scroll continuously: only genuine
    travel of the hand past ``threshold`` produces one direction event, and a
    cooldown prevents a burst of wheel notches from a single motion.
    """

    def __init__(self, threshold=0.10, cooldown=0.30, steps=3):
        self.detector = DirectionDetector(threshold=threshold, axis_ratio=1.0)
        self.cooldown = cooldown
        self.steps = steps
        self.last = -1e9

    @classmethod
    def from_config(cls, cfg):
        m = cfg.get("movement", {}) or {}
        c = cfg.get("cooldown", {}) or {}
        return cls(threshold=m.get("scroll_threshold", 0.10),
                   cooldown=c.get("scroll_seconds", 0.30),
                   steps=m.get("scroll_steps", 3))

    def reset(self):
        self.detector.reset()

    def update(self, x_norm, y_norm, now):
        """Feed the current normalised hand position; return 'up'/'down'/None
        (horizontal directions are ignored for scrolling)."""
        direction = self.detector.update(x_norm, y_norm)
        if direction not in ("up", "down"):
            return None
        if (now - self.last) < self.cooldown:
            return None
        self.last = now
        return direction

    def execute(self, direction):
        return pin.mouse_wheel(pin.WHEEL_DELTA * self.steps *
                               (1 if direction == "up" else -1))
