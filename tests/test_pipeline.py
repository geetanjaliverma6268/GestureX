"""End-to-end pipeline tests: hand counts, two-hand routing, noise, rapid
changes, ML-ready features. All run headlessly with OS input disabled."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from helpers import FrameDriver, hand_dict, sh
from gesture import FEATURE_DIM, calculate_landmark_features, detect_gesture
from utils import load_config
from utils.constants import (
    FIST,
    MOUSE,
    NORMAL,
    OPEN_PALM,
    PEACE,
    PRESENTATION,
    THUMBS_UP,
    UNKNOWN,
)


class TestHandCounts(unittest.TestCase):
    def setUp(self):
        self.d = FrameDriver(load_config())
        self.d.actions.lock.locked = False
        self.d.actions.enabled = True

    def test_no_hand_is_safe(self):
        for _ in range(10):
            cursor = self.d.step(NORMAL, None, None)
        self.assertIsNone(cursor)
        self.assertEqual(self.d.actions.last_action, "")

    def test_one_hand_fires_media(self):
        tu = hand_dict(sh.thumbs_up(), "Right")
        for _ in range(self.d.actions.hold_frames + 3):
            self.d.step(NORMAL, None, tu)
        self.assertTrue(self.d.actions.last_action.startswith("Play"))

    def test_two_hands_independent_channels(self):
        """Right hand -> media, left hand -> mouse; no cross-triggering."""
        right = hand_dict(sh.thumbs_up(), "Right")
        left = hand_dict(sh.pointing_up(center=(200, 240)), "Left")
        cursor = None
        for _ in range(self.d.actions.hold_frames + 3):
            cursor = self.d.step(MOUSE, left, right)
        # Right hand still drives media in MOUSE mode.
        self.assertTrue(self.d.actions.last_action.startswith("Play"))
        # Left hand drives the cursor.
        self.assertIsNotNone(cursor)


class TestSafetyInPipeline(unittest.TestCase):
    def setUp(self):
        self.d = FrameDriver(load_config())

    def test_locked_pipeline_never_acts(self):
        self.assertTrue(self.d.actions.lock.locked)
        tu = hand_dict(sh.thumbs_up(), "Right")
        for _ in range(30):
            self.d.step(NORMAL, None, tu)
        self.assertEqual(self.d.actions.last_action, "")

    def test_mouse_does_not_move_while_locked(self):
        self.assertTrue(self.d.actions.lock.locked)
        left = hand_dict(sh.pointing_up(center=(320, 240)), "Left")
        cursor = self.d.step(MOUSE, left, None)
        self.assertIsNone(cursor)

    def test_open_palm_hold_unlocks_pipeline(self):
        palm = hand_dict(sh.open_palm(), "Right")
        for _ in range(self.d.actions.lock.hold_frames + 3):
            self.d.step(NORMAL, None, palm)
        self.assertFalse(self.d.actions.lock.locked)


class TestDynamicGesturesInPipeline(unittest.TestCase):
    def setUp(self):
        self.d = FrameDriver(load_config())
        self.d.actions.lock.locked = False
        self.d.actions.enabled = True

    def test_two_finger_scroll_up(self):
        self.d.step(NORMAL, None, hand_dict(sh.victory(center=(320, 360)), "Right"))
        for _ in range(4):
            self.d.step(NORMAL, None, hand_dict(sh.victory(center=(320, 200)), "Right"))
        self.assertIn("Scroll", self.d.actions.last_action)

    def test_static_two_finger_does_not_scroll(self):
        v = hand_dict(sh.victory(center=(320, 300)), "Right")
        for _ in range(20):
            self.d.step(NORMAL, None, v)
        self.assertEqual(self.d.actions.last_action, "")

    def test_three_finger_screenshot(self):
        tf = hand_dict(sh.three_finger(), "Right")
        for _ in range(self.d.actions.hold_frames + 3):
            self.d.step(NORMAL, None, tf)
        self.assertIn("Screenshot", self.d.actions.last_action)

    def test_four_finger_show_desktop(self):
        ff = hand_dict(sh.four_finger(), "Right")
        for _ in range(self.d.actions.hold_frames + 3):
            self.d.step(NORMAL, None, ff)
        self.assertIn("Show Desktop", self.d.actions.last_action)


class TestNoiseAndRapidChanges(unittest.TestCase):
    def setUp(self):
        self.d = FrameDriver(load_config())
        self.d.actions.lock.locked = False
        self.d.actions.enabled = True

    def _count_fires(self):
        """Number of times an action actually executed (last_action_time moves)."""
        return getattr(self, "_fires", 0)

    def test_noisy_landmarks_debounced_by_cooldown(self):
        """Heavily jittered frames must not produce an action burst.

        20 frames at ~33ms span ~0.66s, well under the 1.2s action cooldown, so
        at most one discrete command may fire no matter how the noise classifies.
        """
        prev_time = self.d.actions.last_action_time
        fires = 0
        for i in range(20):
            pts = sh.thumbs_up(noise=60.0, seed=i)  # heavy jitter
            self.d.step(NORMAL, None, hand_dict(pts, "Right"))
            if self.d.actions.last_action_time != prev_time:
                fires += 1
                prev_time = self.d.actions.last_action_time
        self.assertLessEqual(fires, 1)

    def test_rapid_gesture_changes_never_settle(self):
        """Cycling a different gesture every frame must never satisfy the
        multi-frame stability gate, so no static action fires."""
        gestures = [sh.thumbs_up(), sh.closed_fist(), sh.pointing_up(), sh.victory()]
        for i in range(40):
            self.d.step(NORMAL, None, hand_dict(gestures[i % len(gestures)], "Right"))
        self.assertEqual(self.d.actions.last_action, "")

    def test_unknown_gesture_no_action(self):
        """A hand explicitly reported as UNKNOWN has no mapping and never acts."""
        h = hand_dict(sh.closed_fist(), "Right")
        h["gesture"] = UNKNOWN
        h["confidence"] = 0.99
        for _ in range(20):
            self.d.step(NORMAL, None, h)
        self.assertEqual(self.d.actions.last_action, "")


class TestMLReadyFeatures(unittest.TestCase):
    def test_feature_dim(self):
        f = calculate_landmark_features(sh.open_palm())
        self.assertEqual(f.shape, (FEATURE_DIM,))

    def test_translation_invariance(self):
        a = calculate_landmark_features(sh.open_palm(center=(200, 200)))
        b = calculate_landmark_features(sh.open_palm(center=(400, 350)))
        self.assertTrue(np.allclose(a, b, atol=1e-4))

    def test_scale_invariance(self):
        a = calculate_landmark_features(sh.victory(scale=120))
        b = calculate_landmark_features(sh.victory(scale=300))
        self.assertTrue(np.allclose(a, b, atol=1e-3))

    def test_distinct_gestures_differ(self):
        a = calculate_landmark_features(sh.open_palm())
        b = calculate_landmark_features(sh.closed_fist())
        self.assertFalse(np.allclose(a, b, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
