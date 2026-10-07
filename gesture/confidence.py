"""Geometric confidence scoring for a classified gesture.

Confidence is derived from how cleanly the landmark geometry satisfies the
gesture's defining constraints (finger-extension certainty, pinch tightness,
thumb direction) - never a random value.
"""

from hand_tracking.landmarks import FINGER_KEYS, clamp
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
from .geometry import derived
from .thresholds import DEFAULT_THRESHOLDS, GestureThresholds


def calculate_gesture_confidence(name, points, finger_info,
                                 thresholds: GestureThresholds = None):
    """Confidence in [0,1] for ``name`` given the landmark geometry."""
    th = thresholds or DEFAULT_THRESHOLDS
    if name == UNKNOWN:
        return 0.0

    _, cert, _, pinch, vertical, _ = derived(points, finger_info, th)

    def avg(keys):
        return sum(cert[k] for k in keys) / len(keys)

    if name in (OK_SIGN, PINCH):
        pinch_conf = clamp(1.0 - pinch / max(th.ok_pinch_scale, 1e-6))
        base = avg(("middle", "ring", "pinky")) if name == OK_SIGN else avg(("thumb",))
        return clamp(0.5 * base + 0.5 * pinch_conf, 0.4, 1.0)

    if name in (THUMBS_UP, THUMBS_DOWN):
        dir_conf = clamp(abs(vertical) / max(th.thumb_dir_scale, 1e-6))
        return clamp(0.5 * avg(("index", "middle", "ring", "pinky")) +
                     0.5 * dir_conf, 0.4, 1.0)

    if name in (FIST, OPEN_PALM, I_LOVE_YOU, FOUR_FINGER):
        return clamp(avg(FINGER_KEYS), 0.4, 1.0)

    if name in (PEACE, POINTING, THREE_FINGER):
        return clamp(avg(("index", "middle", "ring", "pinky")), 0.4, 1.0)

    return 0.0


# Backwards-compatible alias matching the original monolith function name.
calculate_confidence = calculate_gesture_confidence
