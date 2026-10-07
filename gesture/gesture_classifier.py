"""Gesture classification.

The system ships with a geometry-based classifier but is architected so a trained
ML model can replace or complement it without touching the rest of the pipeline:

    GeometryClassifier  ┐
                        ├─> BaseGestureClassifier -> GestureEngine -> Action
    MLGestureClassifier ┘

Both classifiers expose ``classify(points, finger_info) -> (name, confidence)``.
``GestureEngine`` picks a classifier at runtime and applies the confidence
threshold, returning UNKNOWN for anything below it.
"""

import os

import numpy as np

from hand_tracking.landmarks import FINGER_KEYS
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
from .confidence import calculate_gesture_confidence
from .finger_state import calculate_finger_states, count_fingers
from .geometry import derived
from .thresholds import DEFAULT_THRESHOLDS, GestureThresholds


class BaseGestureClassifier:
    """Interface shared by geometry and ML classifiers."""

    name = "base"

    def classify(self, points, finger_info):
        raise NotImplementedError


class GeometryClassifier(BaseGestureClassifier):
    """Rule-based classifier using scale-invariant landmark relationships."""

    name = "geometry"

    def __init__(self, thresholds: GestureThresholds = None):
        self.th = thresholds or DEFAULT_THRESHOLDS

    def _classify(self, points, finger_info) -> str:
        st, _, _, pinch, vertical, fingers_ext = derived(points, finger_info, self.th)
        thumb, index, middle, ring, pinky = (st[k] for k in FINGER_KEYS)
        th = self.th

        # Pinch-family (thumb + index tips close together).
        if pinch < th.ok_pinch_max and middle and ring and pinky:
            return OK_SIGN
        if thumb and pinch < th.ok_pinch_max and not (middle and ring and pinky):
            return PINCH
        # Thumb direction with all fingers curled.
        if thumb and fingers_ext == 0 and abs(vertical) > th.thumb_dir_min:
            return THUMBS_UP if vertical > 0 else THUMBS_DOWN
        if fingers_ext == 0 and not thumb:
            return FIST
        # All four fingers extended.
        if fingers_ext == 4 and thumb:
            return OPEN_PALM
        if fingers_ext == 4 and not thumb:
            return FOUR_FINGER
        # Thumb + index + pinky.
        if thumb and index and pinky and not middle and not ring:
            return I_LOVE_YOU
        # Two-finger / victory.
        if index and middle and not ring and not pinky:
            return PEACE
        # Three-finger (index + middle + ring, thumb tucked).
        if index and middle and ring and not pinky and not thumb:
            return THREE_FINGER
        # Single pointing finger.
        if index and not middle and not ring and not pinky:
            return POINTING
        return UNKNOWN

    def classify(self, points, finger_info):
        name = self._classify(points, finger_info)
        conf = calculate_gesture_confidence(name, points, finger_info, self.th)
        return name, conf


class MLGestureClassifier(BaseGestureClassifier):
    """Drop-in ML classifier.

    Loads a pickled scikit-learn estimator and maps the feature vector from
    ``calculate_landmark_features`` to a gesture label. No model is trained or
    shipped; this exists so a future dataset/pipeline can be added without
    refactoring the engine. If the model file is absent, ``available`` is False
    and ``GestureEngine`` transparently falls back to geometry.
    """

    name = "ml"

    def __init__(self, model_path, labels=None, thresholds: GestureThresholds = None):
        from .gesture_features import calculate_landmark_features
        self._features = calculate_landmark_features
        self.th = thresholds or DEFAULT_THRESHOLDS
        self.labels = labels
        self.model = None
        self.available = False
        if model_path and os.path.exists(model_path):
            try:
                import joblib
                self.model = joblib.load(model_path)
                self.available = True
            except Exception:
                self.model = None
                self.available = False

    def classify(self, points, finger_info):
        if not self.available:
            raise RuntimeError("ML classifier model not loaded")
        x = self._features(points, finger_info).reshape(1, -1)
        label = self.model.predict(x)[0]
        conf = 1.0
        if hasattr(self.model, "predict_proba"):
            conf = float(np.max(self.model.predict_proba(x)[0]))
        return str(label), conf


class GestureEngine:
    """Chooses a classifier and applies the confidence threshold."""

    def __init__(self, classifier: BaseGestureClassifier = None,
                 thresholds: GestureThresholds = None):
        self.th = thresholds or DEFAULT_THRESHOLDS
        self.classifier = classifier or GeometryClassifier(self.th)

    def detect(self, points, finger_info=None):
        """Landmarks -> {gesture, confidence, fingers, finger_info}."""
        if finger_info is None:
            finger_info = calculate_finger_states(points, self.th)
        name, conf = self.classifier.classify(points, finger_info)
        if conf < self.th.confidence_threshold:
            name = UNKNOWN
        return {
            "gesture": name,
            "confidence": conf,
            "fingers": count_fingers(finger_info),
            "finger_info": finger_info,
        }


_DEFAULT_ENGINE = None


def _default_engine():
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = GestureEngine()
    return _DEFAULT_ENGINE


def recognize_gesture(points, finger_info=None, classifier=None):
    """Return (name, confidence). Uses the geometry classifier by default."""
    if finger_info is None:
        finger_info = calculate_finger_states(points)
    clf = classifier or GeometryClassifier()
    return clf.classify(points, finger_info)


def detect_gesture(points, engine: GestureEngine = None):
    """Backwards-compatible helper matching the original monolith signature."""
    eng = engine or _default_engine()
    return eng.detect(points)
