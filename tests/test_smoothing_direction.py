"""Temporal smoothing and direction-detection tests."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture import (
    DirectionDetector,
    GestureSmoother,
    detect_direction,
    smooth_gesture,
)
from utils.constants import FIST, THUMBS_UP, UNKNOWN


class TestGestureSmoothing(unittest.TestCase):
    def test_majority_vote_stabilises(self):
        s = GestureSmoother(window=9, min_votes=5)
        for _ in range(6):
            name, _ = smooth_gesture(s, "Right", THUMBS_UP, 0.9)
        self.assertEqual(name, THUMBS_UP)

    def test_single_noise_frame_does_not_flip(self):
        s = GestureSmoother(window=9, min_votes=5)
        seq = [THUMBS_UP, THUMBS_UP, UNKNOWN, FIST, THUMBS_UP,
               THUMBS_UP, THUMBS_UP, THUMBS_UP, THUMBS_UP]
        name = None
        for g in seq:
            name, _ = smooth_gesture(s, "Right", g, 0.8)
        self.assertEqual(name, THUMBS_UP)

    def test_unstable_sequence_stays_fallback(self):
        s = GestureSmoother(window=5, min_votes=4)
        # Alternating gestures never reach the vote threshold.
        name = None
        for i, g in enumerate([THUMBS_UP, FIST, THUMBS_UP, FIST, THUMBS_UP]):
            name, _ = smooth_gesture(s, "Right", g, 0.9)
        self.assertNotEqual(name, FIST)

    def test_per_hand_history_isolated(self):
        s = GestureSmoother(window=5, min_votes=3)
        for _ in range(3):
            smooth_gesture(s, "Right", THUMBS_UP, 0.9)
            smooth_gesture(s, "Left", FIST, 0.9)
        self.assertEqual(s.stable("Right", UNKNOWN, 0.0)[0], THUMBS_UP)
        self.assertEqual(s.stable("Left", UNKNOWN, 0.0)[0], FIST)


class TestDirectionDetection(unittest.TestCase):
    def test_pure_directions(self):
        self.assertEqual(detect_direction(0.0, -0.3, 0.1), "up")
        self.assertEqual(detect_direction(0.0, 0.3, 0.1), "down")
        self.assertEqual(detect_direction(0.3, 0.0, 0.1), "right")
        self.assertEqual(detect_direction(-0.3, 0.0, 0.1), "left")

    def test_below_threshold_ignored(self):
        self.assertIsNone(detect_direction(0.0, -0.02, 0.1))
        self.assertIsNone(detect_direction(0.03, 0.0, 0.1))

    def test_detector_resets_reference(self):
        d = DirectionDetector(threshold=0.10)
        self.assertIsNone(d.update(0.5, 0.5))      # sets reference
        self.assertEqual(d.update(0.5, 0.3), "up")  # moved up 0.2
        self.assertIsNone(d.update(0.5, 0.3))       # no further movement
        self.assertEqual(d.update(0.5, 0.5), "down")

    def test_jitter_does_not_fire(self):
        d = DirectionDetector(threshold=0.10)
        d.update(0.5, 0.5)
        # Sub-threshold jitter around the reference.
        outs = [d.update(0.5, 0.5 + ((-1) ** i) * 0.02) for i in range(10)]
        self.assertTrue(all(o is None for o in outs))


if __name__ == "__main__":
    unittest.main()
