"""Hand tracking layer: MediaPipe detection, landmarks and handedness."""

from mediapipe.tasks.python.vision import HandLandmarksConnections

from .detector import (
    build_from_config,
    build_hand_landmarker,
    detect_hands,
    hand_count,
)
from .landmarks import (
    FINGER_KEYS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    MCP_IDS,
    PIP_IDS,
    TIP_IDS,
    WRIST,
    LandmarkSmoother,
    clamp,
    distance,
    get_landmarks,
    smooth_landmarks,
    track_landmarks,
)
from .handedness import hand_role, normalize_label, split_hands

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

__all__ = [
    "build_hand_landmarker",
    "build_from_config",
    "detect_hands",
    "hand_count",
    "HAND_CONNECTIONS",
    "LandmarkSmoother",
    "smooth_landmarks",
    "get_landmarks",
    "track_landmarks",
    "distance",
    "clamp",
    "normalize_label",
    "hand_role",
    "split_hands",
    "WRIST",
    "INDEX_TIP",
    "INDEX_MCP",
    "MIDDLE_MCP",
    "MIDDLE_TIP",
    "FINGER_KEYS",
    "TIP_IDS",
    "PIP_IDS",
    "MCP_IDS",
]
