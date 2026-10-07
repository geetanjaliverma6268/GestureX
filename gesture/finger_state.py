"""Finger extension states from landmark geometry.

A finger counts as extended when its tip is farther from the wrist than its PIP
joint by a scale-invariant ratio. Using wrist-relative distances keeps the test
robust to how close the hand is to the camera.
"""

from hand_tracking.landmarks import (
    FINGER_KEYS,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    distance,
)
from .thresholds import DEFAULT_THRESHOLDS, GestureThresholds


def _safe_ratio(tip, pip, wrist):
    denom = distance(pip, wrist)
    return 0.0 if denom < 1e-6 else distance(tip, wrist) / denom


def calculate_finger_states(points, thresholds: GestureThresholds = None):
    """Return {'states': {finger: bool}, 'ratios': {finger: float}}."""
    th = thresholds or DEFAULT_THRESHOLDS
    wrist = points[WRIST]
    ratios = {
        "thumb": _safe_ratio(points[THUMB_TIP], points[THUMB_IP], wrist),
        "index": _safe_ratio(points[INDEX_TIP], points[INDEX_PIP], wrist),
        "middle": _safe_ratio(points[MIDDLE_TIP], points[MIDDLE_PIP], wrist),
        "ring": _safe_ratio(points[RING_TIP], points[RING_PIP], wrist),
        "pinky": _safe_ratio(points[PINKY_TIP], points[PINKY_PIP], wrist),
    }
    states = {
        "thumb": ratios["thumb"] > th.thumb_extend_ratio,
        "index": ratios["index"] > th.finger_extend_ratio,
        "middle": ratios["middle"] > th.finger_extend_ratio,
        "ring": ratios["ring"] > th.finger_extend_ratio,
        "pinky": ratios["pinky"] > th.finger_extend_ratio,
    }
    return {"states": states, "ratios": ratios}


def count_fingers(finger_info) -> int:
    """Number of extended fingers (thumb included)."""
    return sum(1 for k in FINGER_KEYS if finger_info["states"][k])


def count_extended_excluding_thumb(finger_info) -> int:
    return sum(1 for k in ("index", "middle", "ring", "pinky")
               if finger_info["states"][k])
