"""Gesture -> action-intent resolution.

``process_gesture_event`` is the single place that decides what a *stable*
gesture means in the current mode. It returns an intent tuple
``(label, kind, arg)`` that the :class:`control.action_controller.ActionController`
knows how to execute, or ``None`` when the gesture has no static mapping (e.g.
the two-finger scroll, which is a dynamic gesture resolved from motion).

Mapping (NORMAL mode, per spec):
    Thumbs Up     -> Play            Thumbs Down  -> Stop
    Closed Fist   -> Previous Track  Pointing Up  -> Volume Up
    Pinch         -> Volume Down     OK Sign      -> Brightness Up
    I Love You    -> Brightness Down Open Palm    -> Lock/Unlock (held)
    Three Finger  -> Screenshot      Four Finger  -> Show Desktop

PRESENTATION mode:
    Victory -> Next Slide   Closed Fist -> Previous Slide
    Open Palm -> Start/Stop slideshow
"""

from utils.constants import (
    FIST,
    FOUR_FINGER,
    I_LOVE_YOU,
    MOUSE,
    NORMAL,
    OK_SIGN,
    OPEN_PALM,
    PEACE,
    PINCH,
    POINTING,
    PRESENTATION,
    THREE_FINGER,
    THUMBS_DOWN,
    THUMBS_UP,
)

# Available in NORMAL and MOUSE modes (not presentation, to avoid slide clashes).
GLOBAL_ACTIONS = {
    THREE_FINGER: ("Screenshot", "screenshot", "capture"),
    FOUR_FINGER: ("Show Desktop", "desktop", "toggle"),
}

NORMAL_ACTIONS = {
    THUMBS_UP: ("Play", "media", "play"),
    THUMBS_DOWN: ("Stop", "media", "stop"),
    FIST: ("Previous Track", "media", "prev"),
    POINTING: ("Volume Up", "volume", "up"),
    PINCH: ("Volume Down", "volume", "down"),
    OK_SIGN: ("Brightness Up", "brightness", "up"),
    I_LOVE_YOU: ("Brightness Down", "brightness", "down"),
    OPEN_PALM: ("Control Lock/Unlock", "lock", "toggle"),
}

MOUSE_ACTIONS = {
    OPEN_PALM: ("Control Lock/Unlock", "lock", "toggle"),
}

PRESENTATION_ACTIONS = {
    PEACE: ("Next Slide", "presentation", "next"),
    FIST: ("Previous Slide", "presentation", "prev"),
    OPEN_PALM: ("Start/Stop Presentation", "presentation", "toggle"),
}


def process_gesture_event(gesture, mode):
    """Resolve a stable gesture in ``mode`` to an intent tuple, or None."""
    if mode == PRESENTATION:
        return PRESENTATION_ACTIONS.get(gesture)
    if mode == NORMAL:
        return GLOBAL_ACTIONS.get(gesture) or NORMAL_ACTIONS.get(gesture)
    if mode == MOUSE:
        return GLOBAL_ACTIONS.get(gesture) or MOUSE_ACTIONS.get(gesture)
    return None
