"""Gesture classification, finger-state and confidence tests."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import sh
from gesture import (
    calculate_finger_states,
    calculate_gesture_confidence,
    count_fingers,
    detect_gesture,
    recognize_gesture,
)
from utils.constants import (
    FIST,
    FOUR_FINGER,
    I_LOVE_YOU,
    OK_SIGN,
    OPEN_PALM,
    PEACE,
    PINCH,
    POINTING,
    THREE_FINGER,
    THUMBS_DOWN,
    THUMBS_UP,
    UNKNOWN,
)


class TestStaticGestures(unittest.TestCase):
    CASES = [
        ("thumb_up", sh.thumbs_up(), THUMBS_UP),
        ("thumb_down", sh.thumbs_down(), THUMBS_DOWN),
        ("fist", sh.closed_fist(), FIST),
        ("pointing", sh.pointing_up(), POINTING),
        ("victory", sh.victory(), PEACE),
        ("three_finger", sh.three_finger(), THREE_FINGER),
        ("four_finger", sh.four_finger(), FOUR_FINGER),
        ("open_palm", sh.open_palm(), OPEN_PALM),
        ("ok_sign", sh.ok_sign(), OK_SIGN),
        ("pinch", sh.pinch_gesture(), PINCH),
        ("i_love_you", sh.i_love_you(), I_LOVE_YOU),
    ]

    def test_each_gesture_classifies(self):
        for name, pts, expected in self.CASES:
            with self.subTest(name):
                got = detect_gesture(pts)["gesture"]
                self.assertEqual(got, expected)

    def test_confidence_above_threshold(self):
        for name, pts, expected in self.CASES:
            with self.subTest(name):
                d = detect_gesture(pts)
                self.assertGreaterEqual(d["confidence"], 0.60,
                                        f"{name} confidence too low")

    def test_recognize_matches_detect(self):
        for name, pts, expected in self.CASES:
            with self.subTest(name):
                n, _ = recognize_gesture(pts)
                self.assertEqual(n, expected)

    def test_scale_invariance(self):
        """The same gesture at a different distance still classifies."""
        near = sh.open_palm(scale=120)
        far = sh.open_palm(scale=320)
        self.assertEqual(detect_gesture(near)["gesture"], OPEN_PALM)
        self.assertEqual(detect_gesture(far)["gesture"], OPEN_PALM)


class TestFingerStates(unittest.TestCase):
    def test_count_fingers(self):
        self.assertEqual(detect_gesture(sh.closed_fist())["fingers"], 0)
        self.assertEqual(detect_gesture(sh.pointing_up())["fingers"], 1)
        self.assertEqual(detect_gesture(sh.victory())["fingers"], 2)
        self.assertEqual(detect_gesture(sh.three_finger())["fingers"], 3)
        self.assertEqual(detect_gesture(sh.four_finger())["fingers"], 4)
        self.assertEqual(detect_gesture(sh.open_palm())["fingers"], 5)

    def test_finger_state_flags(self):
        info = calculate_finger_states(sh.victory())
        self.assertTrue(info["states"]["index"])
        self.assertTrue(info["states"]["middle"])
        self.assertFalse(info["states"]["ring"])
        self.assertFalse(info["states"]["pinky"])
        self.assertEqual(count_fingers(info), 2)


class TestConfidence(unittest.TestCase):
    def test_unknown_is_zero(self):
        # A hand with an ambiguous posture scores 0 for UNKNOWN.
        pts = sh.closed_fist()
        self.assertEqual(calculate_gesture_confidence(UNKNOWN, pts,
                                                      calculate_finger_states(pts)), 0.0)

    def test_clean_gesture_high_confidence(self):
        pts = sh.thumbs_up()
        c = calculate_gesture_confidence(THUMBS_UP, pts, calculate_finger_states(pts))
        self.assertGreater(c, 0.8)

    def test_confidence_bounded(self):
        for _, pts, name in TestStaticGestures.CASES:
            c = calculate_gesture_confidence(name, pts, calculate_finger_states(pts))
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)


if __name__ == "__main__":
    unittest.main()
