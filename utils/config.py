"""Configuration loading.

Everything tunable lives in ``config.json`` so no threshold is hard-coded in the
pipeline. ``Config`` deep-merges the file over built-in defaults, so a missing
key never crashes the app and a partial config file still works.
"""

import copy
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "config.json")

# Fallback defaults (mirror config.json). Used when the file is missing/partial.
DEFAULTS = {
    "camera_index": 0,
    "model_path": "hand_landmarker.task",
    "num_hands": 2,
    "detection_confidence": 0.5,
    "presence_confidence": 0.5,
    "tracking_confidence": 0.5,
    "screen_resolution": [1536, 864],
    "gesture": {
        "confidence_threshold": 0.60,
        "smoothing_window": 9,
        "min_votes": 5,
        "hold_frames": 8,
        "landmark_smoothing_alpha": 0.55,
        "finger_extend_ratio": 1.10,
        "thumb_extend_ratio": 1.05,
        "ratio_spread": 0.45,
        "ok_pinch_max": 0.28,
        "ok_pinch_scale": 0.45,
        "thumb_dir_min": 0.10,
        "thumb_dir_scale": 0.60,
    },
    "cooldown": {
        "action_seconds": 1.2,
        "scroll_seconds": 0.30,
        "screenshot_seconds": 2.0,
        "desktop_seconds": 2.0,
        "lock_hold_frames": 12,
    },
    "movement": {
        "scroll_threshold": 0.10,
        "scroll_steps": 3,
        "mouse_sensitivity": 1.0,
        "cursor_region": [0.15, 0.12, 0.85, 0.88],
        "cursor_alpha": 0.40,
        "cursor_deadzone": 2,
        "click_pinch_max": 0.30,
        "click_debounce_frames": 2,
        "click_max_duration": 0.5,
        "click_move_max": 30,
        "double_click_window": 0.5,
        "double_click_radius": 50,
    },
    "system": {
        "brightness_step": 10,
        "volume_presses": 2,
        "screenshot_dir": "screenshots",
    },
    "ui": {
        "action_display_seconds": 2.0,
        "window_name": "AI Touchless Computer Control",
    },
    "dashboard": {"enabled": False, "host": "127.0.0.1", "port": 8000},
    "logging": {"level": "INFO", "file": "app.log"},
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dict-backed config with dotted-path access: ``cfg.get("gesture.hold_frames")``."""

    def __init__(self, data=None, root=_PROJECT_ROOT):
        self._data = _deep_merge(DEFAULTS, data or {})
        self.root = root

    def get(self, path, default=None):
        node = self._data
        for part in str(path).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, relative):
        """Resolve a possibly-relative path against the project root."""
        if os.path.isabs(relative):
            return relative
        return os.path.join(self.root, relative)

    def as_dict(self):
        return copy.deepcopy(self._data)

    def __repr__(self):
        return f"Config(keys={list(self._data.keys())})"


def load_config(path=DEFAULT_PATH):
    """Load ``config.json`` (deep-merged over defaults). Never raises on bad JSON
    for optional fields; a completely unreadable file falls back to defaults.
    """
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    except (json.JSONDecodeError, OSError):
        data = {}
    root = os.path.dirname(os.path.abspath(path))
    return Config(data, root=root)
