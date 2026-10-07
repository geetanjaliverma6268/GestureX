"""Control layer tests: cooldown/debounce, control lock, action mapping."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import pin  # ensures OS input is disabled for the whole suite
from utils import load_config
from control import ActionController, ControlLock
from gesture import process_gesture_event
from utils.constants import (
    FIST,
    FOUR_FINGER,
    I_LOVE_YOU,
    MOUSE,
    NORMAL,
    OK_SIGN,
    OPEN_PALM,
    PEACE,
    PINCH,
    POINTING,
    PRESENTATION,
    THREE_FINGER,
    THUMBS_DOWN,
    THUMBS_UP,
)


class TestControlLock(unittest.TestCase):
    def test_starts_locked(self):
        self.assertTrue(ControlLock().locked)

    def test_hold_toggles_unlock(self):
        lock = ControlLock(hold_frames=6, min_conf=0.6)
        for _ in range(5):
            self.assertFalse(lock.update(True, 0.9))
        self.assertTrue(lock.locked)          # not yet
        self.assertTrue(lock.update(True, 0.9))  # 6th frame toggles
        self.assertFalse(lock.locked)

    def test_edge_triggered_needs_release(self):
        lock = ControlLock(hold_frames=4, min_conf=0.6)
        for _ in range(10):
            lock.update(True, 0.9)
        self.assertFalse(lock.locked)         # toggled once
        for _ in range(10):
            lock.update(True, 0.9)
        self.assertFalse(lock.locked)         # holding does not re-toggle
        lock.update(False, 0.0)               # release
        for _ in range(4):
            lock.update(True, 0.9)
        self.assertTrue(lock.locked)          # toggled again after release

    def test_low_confidence_does_not_unlock(self):
        lock = ControlLock(hold_frames=4, min_conf=0.6)
        for _ in range(10):
            lock.update(True, 0.2)
        self.assertTrue(lock.locked)


class TestActionControllerSafety(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.ac = ActionController(self.cfg)

    def test_no_action_while_locked(self):
        self.assertTrue(self.ac.lock.locked)
        self.assertEqual(self.ac.execute("media", "play", "Play"), "")
        self.assertFalse(self.ac.request(("media", "play"), 0.99, now=1.0))

    def test_no_action_when_disabled(self):
        self.ac.lock.locked = False
        self.ac.enabled = False
        self.assertFalse(self.ac.can_act())
        self.assertEqual(self.ac.execute("media", "play", "Play"), "")

    def test_stability_gate_requires_hold(self):
        self.ac.lock.locked = False
        fired = [self.ac.request(("media", "play"), 0.9, now=10 + i * 0.01)
                 for i in range(self.ac.hold_frames + 2)]
        self.assertIn(True, fired)
        self.assertEqual(fired.index(True), self.ac.hold_frames - 1)

    def test_confidence_gate(self):
        self.ac.lock.locked = False
        fired = any(self.ac.request(("volume", "up"), 0.1, now=20 + i * 0.01)
                    for i in range(self.ac.hold_frames + 3))
        self.assertFalse(fired)

    def test_cooldown_blocks_repeat(self):
        self.ac.lock.locked = False
        key = ("media", "play")
        # Fire once.
        for i in range(self.ac.hold_frames + 1):
            self.ac.request(key, 0.9, now=30 + i * 0.01)
        self.assertTrue(self.ac.is_on_cooldown(key, 30.2))
        self.assertFalse(self.ac.is_on_cooldown(key, 30.2 + self.ac.default_cooldown + 0.1))

    def test_set_cooldown_override(self):
        self.ac.set_cooldown("screenshot", 5.0)
        self.assertEqual(self.ac._cooldown_for(("screenshot", "capture")), 5.0)

    def test_reset_clears_cooldowns(self):
        self.ac.lock.locked = False
        key = ("media", "stop")
        for i in range(self.ac.hold_frames + 1):
            self.ac.request(key, 0.9, now=40 + i * 0.01)
        self.ac.reset()
        self.assertFalse(self.ac.is_on_cooldown(key, 40.2))


class TestActionMapping(unittest.TestCase):
    def test_normal_mode_mapping(self):
        self.assertEqual(process_gesture_event(THUMBS_UP, NORMAL), ("Play", "media", "play"))
        self.assertEqual(process_gesture_event(THUMBS_DOWN, NORMAL), ("Stop", "media", "stop"))
        self.assertEqual(process_gesture_event(FIST, NORMAL), ("Previous Track", "media", "prev"))
        self.assertEqual(process_gesture_event(POINTING, NORMAL), ("Volume Up", "volume", "up"))
        self.assertEqual(process_gesture_event(PINCH, NORMAL), ("Volume Down", "volume", "down"))
        self.assertEqual(process_gesture_event(OK_SIGN, NORMAL), ("Brightness Up", "brightness", "up"))
        self.assertEqual(process_gesture_event(I_LOVE_YOU, NORMAL), ("Brightness Down", "brightness", "down"))
        self.assertEqual(process_gesture_event(OPEN_PALM, NORMAL), ("Control Lock/Unlock", "lock", "toggle"))
        self.assertEqual(process_gesture_event(THREE_FINGER, NORMAL), ("Screenshot", "screenshot", "capture"))
        self.assertEqual(process_gesture_event(FOUR_FINGER, NORMAL), ("Show Desktop", "desktop", "toggle"))

    def test_victory_has_no_static_normal_action(self):
        # Two-finger scroll is a dynamic gesture, resolved from motion.
        self.assertIsNone(process_gesture_event(PEACE, NORMAL))

    def test_presentation_mode_mapping(self):
        self.assertEqual(process_gesture_event(PEACE, PRESENTATION), ("Next Slide", "presentation", "next"))
        self.assertEqual(process_gesture_event(FIST, PRESENTATION), ("Previous Slide", "presentation", "prev"))
        self.assertEqual(process_gesture_event(OPEN_PALM, PRESENTATION),
                         ("Start/Stop Presentation", "presentation", "toggle"))

    def test_mouse_mode_only_lock_and_global(self):
        self.assertEqual(process_gesture_event(OPEN_PALM, MOUSE), ("Control Lock/Unlock", "lock", "toggle"))
        self.assertEqual(process_gesture_event(THREE_FINGER, MOUSE), ("Screenshot", "screenshot", "capture"))
        self.assertIsNone(process_gesture_event(THUMBS_UP, MOUSE))

    def test_dispatch_returns_string_when_unlocked(self):
        ac = ActionController(load_config())
        ac.lock.locked = False
        out = ac.execute("media", "play", "Play", now=1.0)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("Play ["))


if __name__ == "__main__":
    unittest.main()
