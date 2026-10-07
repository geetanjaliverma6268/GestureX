"""Performance helpers: smoothed FPS counter."""

import time


class FpsCounter:
    """Exponentially-smoothed frames-per-second meter (ported from the monolith)."""

    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else \
                self._alpha * inst + (1 - self._alpha) * self._fps
        return self._fps

    @property
    def fps(self):
        return self._fps
