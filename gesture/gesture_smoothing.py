"""Temporal gesture smoothing via majority vote over a rolling window.

A single noisy frame cannot flip the reported gesture: the label only becomes
'stable' once it has enough votes in the recent history.
"""

from collections import Counter, deque

DEFAULT_WINDOW = 9
DEFAULT_MIN_VOTES = 5


class GestureSmoother:
    def __init__(self, window=DEFAULT_WINDOW, min_votes=DEFAULT_MIN_VOTES):
        self._window = window
        self._min_votes = min_votes
        self._history = {}

    def push(self, label, name, conf):
        self._history.setdefault(label, deque(maxlen=self._window)).append((name, conf))

    def stable(self, label, fallback_name, fallback_conf):
        hist = self._history.get(label)
        if not hist:
            return fallback_name, fallback_conf
        counts = Counter(n for n, _ in hist)
        top_name, top_votes = counts.most_common(1)[0]
        name = top_name if top_votes >= self._min_votes else fallback_name
        confs = [c for n, c in hist if n == name] or [fallback_conf]
        return name, float(sum(confs) / len(confs))

    def votes(self, label, name):
        hist = self._history.get(label)
        if not hist:
            return 0
        return sum(1 for n, _ in hist if n == name)

    def prune(self, active_labels):
        for label in list(self._history.keys()):
            if label not in active_labels:
                del self._history[label]

    def reset(self):
        self._history.clear()


def smooth_gesture(smoother: GestureSmoother, label, name, conf):
    """Push a fresh observation and return the smoothed (name, confidence)."""
    smoother.push(label, name, conf)
    return smoother.stable(label, name, conf)
