"""Landmark indices, geometry helpers, connection topology and smoothing.

Ported verbatim from the proven single-file implementation so the MediaPipe
21-landmark tracking behaviour is preserved exactly.
"""

import numpy as np

# --- MediaPipe hand landmark indices --------------------------------------- #
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")

TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
PIP_IDS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
MCP_IDS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

DEFAULT_SMOOTHING_ALPHA = 0.55


# --- Small math helpers (shared across gesture + control) ------------------- #

def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) -
                                np.asarray(b, dtype=float)))


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# --- Temporal landmark smoothing ------------------------------------------- #

class LandmarkSmoother:
    """Per-hand EMA over landmark pixel coordinates to reduce jitter."""

    def __init__(self, alpha=DEFAULT_SMOOTHING_ALPHA):
        self._alpha = alpha
        self._history = {}

    def smooth(self, label, points):
        prev = self._history.get(label)
        if prev is None or prev.shape != np.asarray(points).shape:
            smoothed = np.asarray(points).copy()
        else:
            smoothed = self._alpha * points + (1 - self._alpha) * prev
        self._history[label] = smoothed
        return smoothed

    def prune(self, active_labels):
        for label in list(self._history.keys()):
            if label not in active_labels:
                del self._history[label]

    def reset(self):
        self._history.clear()


def smooth_landmarks(smoother: LandmarkSmoother, label: str, points):
    return smoother.smooth(label, points)


def get_landmarks(result, hand_index, width, height):
    """Extract one hand's 21 landmarks as pixel coords + handedness label/score."""
    landmarks = result.hand_landmarks[hand_index]
    points = np.zeros((len(landmarks), 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        points[i] = (lm.x * width, lm.y * height)
    handedness = result.handedness[hand_index][0]
    return points, handedness.category_name, float(handedness.score)


def track_landmarks(result, hand_index, width, height, smoother):
    """Get one hand's landmarks and apply temporal smoothing."""
    raw, label, score = get_landmarks(result, hand_index, width, height)
    return smooth_landmarks(smoother, label, raw), label, score
