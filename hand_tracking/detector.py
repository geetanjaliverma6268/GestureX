"""MediaPipe Hand Landmarker (Tasks API) detection wrapper.

Preserves the proven VIDEO-mode detection from the original implementation, now
driven by config values (num_hands, confidence thresholds, model path).
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from utils.logger import get_logger

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "hand_landmarker.task")

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def build_hand_landmarker(model_path=DEFAULT_MODEL_PATH, num_hands=2,
                          detection_confidence=0.5, presence_confidence=0.5,
                          tracking_confidence=0.5):
    """Create a VIDEO-mode HandLandmarker. Raises FileNotFoundError if the model
    bundle is missing, with the exact download command."""
    log = get_logger("detector")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\nDownload it with:\n"
            f"  curl -L -o hand_landmarker.task {MODEL_URL}"
        )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    log.info("MediaPipe HandLandmarker initialized (num_hands=%d)", num_hands)
    return HandLandmarker.create_from_options(options)


def build_from_config(cfg):
    """Build a landmarker using values from a loaded Config object."""
    return build_hand_landmarker(
        model_path=cfg.path(cfg.get("model_path", "hand_landmarker.task")),
        num_hands=cfg.get("num_hands", 2),
        detection_confidence=cfg.get("detection_confidence", 0.5),
        presence_confidence=cfg.get("presence_confidence", 0.5),
        tracking_confidence=cfg.get("tracking_confidence", 0.5),
    )


def detect_hands(landmarker, frame_bgr, timestamp_ms):
    """Run inference on one BGR frame. ``timestamp_ms`` must be monotonically
    increasing across calls (VIDEO mode requirement)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, int(timestamp_ms))


def hand_count(result) -> int:
    """Number of hands in a detection result (0 when nothing is visible)."""
    try:
        return len(result.hand_landmarks)
    except AttributeError:
        return 0
