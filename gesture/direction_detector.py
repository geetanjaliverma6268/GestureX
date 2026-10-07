"""Dynamic (movement) gesture direction detection.

Static gestures (thumbs up, OK, ...) are classified by posture; *dynamic*
gestures are classified by motion. This module turns a movement vector between
two landmark positions into a discrete direction, with a threshold to ignore
jitter and a dominant-axis requirement so diagonal noise does not fire.

Designed to be extensible: swipe-left/right, circle and wave detectors can be
added later without changing the static-gesture path.
"""

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"


def detect_direction(delta_x, delta_y, threshold, axis_ratio=1.0):
    """Return 'up'/'down'/'left'/'right' or None from a movement vector.

    ``delta = current - previous`` in image coordinates (y grows downward).
    ``axis_ratio`` requires the dominant axis to exceed the other by this factor
    (1.0 = strictly larger) before a direction is reported.
    """
    ax, ay = abs(delta_x), abs(delta_y)
    if ay >= ax * axis_ratio and ay >= threshold:
        return UP if delta_y < 0 else DOWN
    if ax >= ay * axis_ratio and ax >= threshold:
        return RIGHT if delta_x > 0 else LEFT
    return None


class DirectionDetector:
    """Tracks a reference point and emits discrete directions on sufficient travel.

    Used for the two-finger scroll: while exactly two fingers are raised, the
    reference wrist position is compared against the current one. A movement
    beyond ``threshold`` (normalised units) yields one direction event, then the
    reference is reset so the same travel cannot re-fire.
    """

    def __init__(self, threshold=0.10, axis_ratio=1.0):
        self.threshold = threshold
        self.axis_ratio = axis_ratio
        self.ref = None

    def reset(self):
        self.ref = None

    def update(self, x, y):
        """Feed the current normalised position; return a direction or None."""
        if self.ref is None:
            self.ref = (x, y)
            return None
        dx = x - self.ref[0]
        dy = y - self.ref[1]
        direction = detect_direction(dx, dy, self.threshold, self.axis_ratio)
        if direction is not None:
            self.ref = (x, y)
        return direction
