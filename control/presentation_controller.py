"""Presentation control via generic keyboard events (no app dependency)."""

from . import platform_input as pin
from utils.logger import get_logger

log = get_logger("presentation")


def presentation_controller(command: str) -> str:
    """next | prev | start | stop. Uses arrow keys / F5 / Esc so it works with
    any slideshow software rather than a specific vendor."""
    command = str(command).lower()
    vk_map = {"next": pin.VK_RIGHT, "prev": pin.VK_LEFT,
              "start": pin.VK_F5, "stop": pin.VK_ESCAPE}
    vk = vk_map.get(command)
    if vk is None:
        return "Presentation: unknown command"
    ok = pin.press_vk(vk)
    log.info("Presentation: %s (%s)", command, "sent" if ok else "unavailable")
    return "Presentation: sent" if ok else "Presentation: unavailable (non-Windows)"
