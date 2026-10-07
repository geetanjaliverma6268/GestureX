"""ML-ready feature extraction from the 21 landmarks.

The geometry classifier does not need this, but it is the bridge to a trained
model: ``calculate_landmark_features`` turns raw landmarks into a fixed-length,
scale/translation-invariant float vector that any classifier (Random Forest,
SVM, MLP, LSTM window, ...) can consume.

Feature layout (fixed order, length 21*2 + 5 + 1 = 48):
    [0:42]  wrist-relative, hand-size-normalised (x, y) for all 21 landmarks
    [42:47] per-finger extension ratios (thumb, index, middle, ring, pinky)
    [47]    thumb-index pinch distance (normalised)
"""

import numpy as np

from hand_tracking.landmarks import (
    FINGER_KEYS,
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
    distance,
)
from .finger_state import calculate_finger_states

FEATURE_NAMES = (
    [f"lm{i}_x" for i in range(21)]
    + [f"lm{i}_y" for i in range(21)]
    + [f"ratio_{k}" for k in FINGER_KEYS]
    + ["pinch"]
)
FEATURE_DIM = len(FEATURE_NAMES)  # 48


def calculate_landmark_features(points, finger_info=None):
    """Return a normalised feature vector (np.float32, length FEATURE_DIM)."""
    pts = np.asarray(points, dtype=float)
    wrist = pts[WRIST]
    hand_size = max(distance(pts[MIDDLE_MCP], wrist), 1e-6)

    rel = (pts - wrist) / hand_size            # translation + scale invariant
    feat = np.concatenate([rel[:, 0], rel[:, 1]]).astype(np.float32)

    if finger_info is None:
        finger_info = calculate_finger_states(pts)
    ratios = np.array([finger_info["ratios"][k] for k in FINGER_KEYS],
                      dtype=np.float32)
    pinch = np.array([distance(pts[THUMB_TIP], pts[INDEX_TIP]) / hand_size],
                     dtype=np.float32)
    return np.concatenate([feat, ratios, pinch]).astype(np.float32)


def feature_vector(points):
    """Convenience alias returning (vector, finger_info)."""
    fi = calculate_finger_states(points)
    return calculate_landmark_features(points, fi), fi
