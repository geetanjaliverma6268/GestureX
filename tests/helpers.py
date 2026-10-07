"""Test helpers: build pipeline-shaped hand dicts and drive main._process_frame
headlessly with all real OS input disabled."""

import os
import sys

# Make the project root importable regardless of the test runner's cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from control import platform_input as pin  # noqa: E402

# NEVER let tests fire real keys, move the mouse or write screenshots.
pin.set_enabled(False)

import synthetic_hands as sh  # noqa: E402
from gesture import detect_gesture  # noqa: E402


def hand_dict(pts, label="Right", score=0.97):
    """Wrap landmarks in the dict shape the pipeline produces for one hand."""
    d = detect_gesture(pts)
    return {
        "points": pts,
        "color": (0, 200, 0),
        "score": score,
        "fingers": d["fingers"],
        "gesture": d["gesture"],
        "confidence": d["confidence"],
        "label": label,
    }


class FrameDriver:
    """Drives main._process_frame with a fixed clock and fresh controllers."""

    def __init__(self, cfg):
        import main
        from control import ActionController, MouseController, ScrollController
        from ui import DashboardState
        self.main = main
        self.cfg = cfg
        self.actions = ActionController(cfg)
        self.mouse = MouseController.from_config(cfg)
        self.scroller = ScrollController.from_config(cfg)
        self.dash = DashboardState()
        self.t = 1000.0
        self.W, self.H = 640, 480
        self.SW, self.SH = pin.screen_size()

    def step(self, mode, left=None, right=None, dt=0.033):
        self.t += dt
        cursor, slide = self.main._process_frame(
            mode, left, right, self.actions, self.mouse, self.scroller, self.t,
            self.W, self.H, self.SW, self.SH, self.actions.min_conf, False, self.dash)
        return cursor

    def hold(self, mode, gesture_pts, frames, hand="right", **kw):
        """Feed the same gesture for ``frames`` frames; return last cursor."""
        h = hand_dict(sh.make_hand(**gesture_pts) if isinstance(gesture_pts, dict)
                      else gesture_pts)
        cursor = None
        for _ in range(frames):
            cursor = self.step(mode, left=h if hand == "left" else None,
                               right=h if hand == "right" else None)
        return cursor
