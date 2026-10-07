"""Integration test for main.run() with a mocked camera and landmarker.

Exercises the full per-frame wiring end-to-end (detection wrapper -> landmark
smoothing -> gesture engine -> control routing -> overlay -> shutdown) without a
physical webcam or any real OS input. This is the test that proves run() itself
is wired correctly, complementing the unit tests around each component.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import helpers  # noqa: F401  (disables real OS input on import)
import main
from helpers import sh


class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeHandedness:
    def __init__(self, category_name, score):
        self.category_name = category_name
        self.score = score


def _fake_result(points, width, height, label="Right"):
    """Build a MediaPipe-shaped result from pixel landmarks."""
    lms = [_FakeLandmark(float(p[0] / width), float(p[1] / height)) for p in points]
    result = mock.Mock()
    result.hand_landmarks = [lms]
    result.handedness = [[_FakeHandedness(label, 0.97)]]
    return result


class _FakeCap:
    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def isOpened(self):
        return True

    def release(self):
        pass


class TestRunIntegration(unittest.TestCase):
    def _run_with(self, gesture_seq, label="Right", frames=30):
        W, H = 640, 480
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        cap = _FakeCap([blank] * frames)

        # Cycle the requested gesture landmarks across frames.
        seq = [_fake_result(g, W, H, label) for g in gesture_seq]
        call = {"i": 0}

        def fake_detect(landmarker, frame_bgr, timestamp_ms):
            r = seq[call["i"] % len(seq)]
            call["i"] += 1
            return r

        fake_landmarker = mock.Mock()
        with mock.patch.object(main, "_open_camera", return_value=cap), \
             mock.patch.object(main, "build_from_config", return_value=fake_landmarker), \
             mock.patch.object(main, "detect_hands", side_effect=fake_detect), \
             mock.patch.object(main.cv2, "imshow", lambda *a, **k: None), \
             mock.patch.object(main.cv2, "waitKey", lambda *a, **k: 255), \
             mock.patch.object(main.cv2, "destroyAllWindows", lambda *a, **k: None):
            rc = main.run()
        return rc

    def test_run_completes_with_thumbs_up(self):
        rc = self._run_with([sh.thumbs_up()])
        self.assertEqual(rc, 0)

    def test_run_completes_with_two_hands_and_modes(self):
        # Mixed gestures across frames; exercises smoothing + routing + overlay.
        seq = [sh.open_palm(), sh.thumbs_up(), sh.victory(), sh.closed_fist(),
               sh.pointing_up(), sh.three_finger(), sh.four_finger()]
        rc = self._run_with(seq, frames=60)
        self.assertEqual(rc, 0)

    def test_run_handles_no_hand_frames(self):
        W, H = 640, 480
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        cap = _FakeCap([blank] * 15)
        empty = mock.Mock()
        empty.hand_landmarks = []
        empty.handedness = []
        with mock.patch.object(main, "_open_camera", return_value=cap), \
             mock.patch.object(main, "build_from_config", return_value=mock.Mock()), \
             mock.patch.object(main, "detect_hands", return_value=empty), \
             mock.patch.object(main.cv2, "imshow", lambda *a, **k: None), \
             mock.patch.object(main.cv2, "waitKey", lambda *a, **k: 255), \
             mock.patch.object(main.cv2, "destroyAllWindows", lambda *a, **k: None):
            rc = main.run()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
