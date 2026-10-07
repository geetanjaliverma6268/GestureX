"""Gesture recognition layer: features, finger states, classification,
confidence, smoothing, direction detection and event resolution."""

from .confidence import calculate_confidence, calculate_gesture_confidence
from .direction_detector import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DirectionDetector,
    detect_direction,
)
from .events import (
    GLOBAL_ACTIONS,
    NORMAL_ACTIONS,
    PRESENTATION_ACTIONS,
    process_gesture_event,
)
from .finger_state import (
    calculate_finger_states,
    count_extended_excluding_thumb,
    count_fingers,
)
from .gesture_classifier import (
    BaseGestureClassifier,
    GestureEngine,
    GeometryClassifier,
    MLGestureClassifier,
    detect_gesture,
    recognize_gesture,
)
from .gesture_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    calculate_landmark_features,
    feature_vector,
)
from .gesture_smoothing import GestureSmoother, smooth_gesture
from .thresholds import DEFAULT_THRESHOLDS, GestureThresholds

__all__ = [
    "calculate_finger_states",
    "count_fingers",
    "count_extended_excluding_thumb",
    "calculate_landmark_features",
    "feature_vector",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "BaseGestureClassifier",
    "GeometryClassifier",
    "MLGestureClassifier",
    "GestureEngine",
    "recognize_gesture",
    "detect_gesture",
    "calculate_gesture_confidence",
    "calculate_confidence",
    "GestureSmoother",
    "smooth_gesture",
    "DirectionDetector",
    "detect_direction",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "process_gesture_event",
    "GLOBAL_ACTIONS",
    "NORMAL_ACTIONS",
    "PRESENTATION_ACTIONS",
    "GestureThresholds",
    "DEFAULT_THRESHOLDS",
]
