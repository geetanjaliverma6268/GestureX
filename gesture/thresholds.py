"""Gesture geometry thresholds, loadable from config.

Isolated in its own module so ``finger_state``, ``classifier`` and ``confidence``
can all share one source of truth without circular imports.
"""

from dataclasses import dataclass


@dataclass
class GestureThresholds:
    finger_extend_ratio: float = 1.10
    thumb_extend_ratio: float = 1.05
    ratio_spread: float = 0.45
    ok_pinch_max: float = 0.28
    ok_pinch_scale: float = 0.45
    thumb_dir_min: float = 0.10
    thumb_dir_scale: float = 0.60
    confidence_threshold: float = 0.60

    @classmethod
    def from_config(cls, cfg):
        g = cfg.get("gesture", {}) or {}
        return cls(
            finger_extend_ratio=g.get("finger_extend_ratio", 1.10),
            thumb_extend_ratio=g.get("thumb_extend_ratio", 1.05),
            ratio_spread=g.get("ratio_spread", 0.45),
            ok_pinch_max=g.get("ok_pinch_max", 0.28),
            ok_pinch_scale=g.get("ok_pinch_scale", 0.45),
            thumb_dir_min=g.get("thumb_dir_min", 0.10),
            thumb_dir_scale=g.get("thumb_dir_scale", 0.60),
            confidence_threshold=g.get("confidence_threshold", 0.60),
        )


DEFAULT_THRESHOLDS = GestureThresholds()
