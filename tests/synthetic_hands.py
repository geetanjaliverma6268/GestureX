"""Synthetic hand-landmark generator for headless tests.

Builds a 21x2 landmark array in MediaPipe order from a per-finger
extended/curled specification, using scale-invariant geometry that mirrors a real
hand (wrist at the origin, fingers extending upward). This lets the gesture,
smoothing, direction and control logic be tested without a webcam.
"""

import numpy as np

# Landmark indices (MediaPipe hand topology).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

_MCP = {
    "thumb": np.array([-0.50, -0.20]),
    "index": np.array([-0.30, -0.90]),
    "middle": np.array([-0.05, -0.95]),
    "ring": np.array([0.20, -0.90]),
    "pinky": np.array([0.45, -0.80]),
}
_DIR = {
    "index": np.array([-0.10, -0.995]),
    "middle": np.array([0.00, -1.000]),
    "ring": np.array([0.12, -0.990]),
    "pinky": np.array([0.28, -0.960]),
}
_THUMB_DIR = {
    "up": np.array([-0.20, -0.980]),
    "down": np.array([-0.20, 0.980]),
    "side": np.array([-0.98, -0.20]),
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False,
              thumb_dir="up", pinch=False, scale=200.0, center=(320.0, 300.0),
              noise=0.0, seed=None):
    """Return a (21,2) float32 landmark array in pixel coordinates."""
    rng = np.random.default_rng(seed) if noise else None
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[WRIST] = (0.0, 0.0)

    def place(mcp_id, pip_id, dip_id, tip_id, key, direction, extended):
        mcp = _MCP[key]
        d = _unit(direction)
        pts[mcp_id] = mcp
        pts[pip_id] = mcp + d * 0.28
        pts[dip_id] = mcp + d * 0.60
        pts[tip_id] = mcp + d * 1.15 if extended else mcp * 0.6

    # Thumb (its "pip" for the ratio test is THUMB_IP).
    place(THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP, "thumb",
          _THUMB_DIR[thumb_dir], thumb)
    pts[THUMB_CMC] = _MCP["thumb"] * 0.5

    place(INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, "index", _DIR["index"], index)
    place(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, "middle", _DIR["middle"], middle)
    place(RING_MCP, RING_PIP, RING_DIP, RING_TIP, "ring", _DIR["ring"], ring)
    place(PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, "pinky", _DIR["pinky"], pinky)

    if pinch:
        # Bring the index tip onto the thumb tip to form a small pinch/OK ring.
        pts[INDEX_TIP] = pts[THUMB_TIP] + np.array([0.04, 0.03])

    # To pixel coordinates.
    pts = pts * scale
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]

    if rng is not None:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)

    return pts.astype(np.float32)


# --- Named gesture builders ------------------------------------------------- #

def thumbs_up(**kw):
    return make_hand(thumb=True, thumb_dir="up", **kw)


def thumbs_down(**kw):
    return make_hand(thumb=True, thumb_dir="down", **kw)


def closed_fist(**kw):
    return make_hand(**kw)


def pointing_up(**kw):
    return make_hand(index=True, **kw)


def victory(**kw):
    return make_hand(index=True, middle=True, **kw)


def three_finger(**kw):
    return make_hand(index=True, middle=True, ring=True, **kw)


def four_finger(**kw):
    return make_hand(index=True, middle=True, ring=True, pinky=True, **kw)


def open_palm(**kw):
    return make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True, **kw)


def ok_sign(**kw):
    return make_hand(thumb=True, middle=True, ring=True, pinky=True, pinch=True, **kw)


def pinch_gesture(**kw):
    return make_hand(thumb=True, pinch=True, **kw)


def i_love_you(**kw):
    return make_hand(thumb=True, index=True, pinky=True, **kw)
