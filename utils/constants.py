"""Shared, dependency-free constants.

Kept in ``utils`` so that the gesture, control and ui packages can all import
the canonical gesture / mode names without creating circular imports.
"""

# --- Gesture names ---------------------------------------------------------- #
FIST = "Closed Fist"
OPEN_PALM = "Open Palm"
FOUR_FINGER = "Four Finger"
THREE_FINGER = "Three Finger"
THUMBS_UP = "Thumbs Up"
THUMBS_DOWN = "Thumbs Down"
PEACE = "Victory"
POINTING = "Pointing Up"
OK_SIGN = "OK Sign"
PINCH = "Pinch"
I_LOVE_YOU = "I Love You"
UNKNOWN = "Unknown"

ALL_GESTURES = (
    THUMBS_UP,
    THUMBS_DOWN,
    FIST,
    POINTING,
    PEACE,
    OK_SIGN,
    I_LOVE_YOU,
    OPEN_PALM,
    PINCH,
    THREE_FINGER,
    FOUR_FINGER,
    UNKNOWN,
)

# --- Operating modes -------------------------------------------------------- #
NORMAL = "NORMAL"
MOUSE = "MOUSE"
PRESENTATION = "PRESENTATION"
MODES = (NORMAL, MOUSE, PRESENTATION)

# --- Hand labels ------------------------------------------------------------ #
LEFT = "Left"
RIGHT = "Right"
