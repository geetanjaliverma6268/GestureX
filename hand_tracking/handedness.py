"""Handedness helpers.

MediaPipe's Tasks API returns a per-hand handedness category (``Left``/``Right``)
with a confidence score. Because the preview frame is mirrored for a natural
selfie view, the reported label corresponds to the user's actual hand. This
module keeps label normalisation in one place.
"""

from utils.constants import LEFT, RIGHT

_VALID = {LEFT, RIGHT}


def normalize_label(category_name: str) -> str:
    """Return canonical 'Left'/'Right'; unknown labels fall back to 'Right'."""
    name = str(category_name).strip().capitalize()
    return name if name in _VALID else RIGHT


def hand_role(label: str, mode: str) -> str:
    """Human-readable role for a hand given the active mode (used by the UI)."""
    from utils.constants import MOUSE, NORMAL, PRESENTATION
    if label == RIGHT:
        return "Media"
    if mode == MOUSE:
        return "Mouse"
    if mode == PRESENTATION:
        return "Slides"
    if mode == NORMAL:
        return "Media*"
    return "-"


def split_hands(hands_by_label: dict):
    """Convenience accessor returning (left, right) hand dicts (may be None)."""
    return hands_by_label.get(LEFT), hands_by_label.get(RIGHT)
