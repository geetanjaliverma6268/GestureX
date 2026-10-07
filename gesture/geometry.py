"""Shared geometry used by both classification and confidence scoring."""

import numpy as np

from hand_tracking.landmarks import (
    FINGER_KEYS,
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
    clamp,
    distance,
)
from .thresholds import DEFAULT_THRESHOLDS, GestureThresholds


def certainty(ratio, threshold, spread):
    """How far a ratio sits from its decision threshold, in [0,1]."""
    return clamp(abs(ratio - threshold) / max(spread, 1e-6))


def derived(points, finger_info, thresholds: GestureThresholds = None):
    """Compute the geometry bundle reused by the classifier and confidence.

    Returns (states, certainties, hand_size, pinch, vertical, fingers_ext).
    """
    th = thresholds or DEFAULT_THRESHOLDS
    pts = np.asarray(points, dtype=float)
    st = finger_info["states"]
    ratios = finger_info["ratios"]

    hand_size = max(distance(pts[MIDDLE_MCP], pts[WRIST]), 1e-6)
    pinch = distance(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size
    vertical = (pts[WRIST][1] - pts[THUMB_TIP][1]) / hand_size

    cert = {
        k: certainty(ratios[k],
                     th.thumb_extend_ratio if k == "thumb" else th.finger_extend_ratio,
                     th.ratio_spread)
        for k in FINGER_KEYS
    }
    fingers_ext = st["index"] + st["middle"] + st["ring"] + st["pinky"]
    return st, cert, hand_size, pinch, vertical, fingers_ext
