"""Media transport control (play / stop / next / previous).

Windows exposes a single combined Play/Pause media key (VK_MEDIA_PLAY_PAUSE) and
no separate Play or Pause virtual key, so "play" sends that toggle and "stop"
uses VK_MEDIA_STOP. There is intentionally no "resume" command.
"""

from . import platform_input as pin
from utils.logger import get_logger

log = get_logger("media")

_VK_MAP = {
    "play": pin.VK_MEDIA_PLAY_PAUSE,
    "pause": pin.VK_MEDIA_PLAY_PAUSE,
    "stop": pin.VK_MEDIA_STOP,
    "next": pin.VK_MEDIA_NEXT_TRACK,
    "prev": pin.VK_MEDIA_PREV_TRACK,
}


def media_controller(command: str) -> str:
    vk = _VK_MAP.get(str(command).lower())
    if vk is None:
        return "Media: unknown command"
    sent = pin.press_vk(vk)
    log.info("Media command: %s (%s)", command, "sent" if sent else "unavailable")
    return "Media: sent" if sent else "Media: unavailable (non-Windows)"
