"""UI layer: OpenCV overlay and optional web dashboard."""

from .dashboard import DashboardState, create_app, run_dashboard
from .overlay import HAND_COLORS, draw_hands, draw_ui

__all__ = [
    "draw_ui",
    "draw_hands",
    "HAND_COLORS",
    "DashboardState",
    "create_app",
    "run_dashboard",
]
