from __future__ import annotations

import platform
from dataclasses import dataclass

from cv_practice.assistive.types import CommandEvent

try:
    from pycaw.pycaw import AudioUtilities
except Exception:  # pragma: no cover - optional runtime dependency
    AudioUtilities = None


VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_KEYUP = 0x0002


def _send_media_play_pause() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


@dataclass(slots=True)
class ActionExecutor:
    endpoint_volume: object | None = None

    def __post_init__(self) -> None:
        if self.endpoint_volume is not None:
            return
        if AudioUtilities is None:
            return
        try:
            self.endpoint_volume = AudioUtilities.GetSpeakers().EndpointVolume
        except Exception:
            self.endpoint_volume = None

    def execute(self, event: CommandEvent) -> bool:
        command = event.command
        if command in {"lock", "unlock", "confirm"}:
            return True
        if command == "media_play_pause":
            return _send_media_play_pause()
        if command == "volume_toggle_mute":
            if self.endpoint_volume is None:
                return False
            try:
                muted = int(self.endpoint_volume.GetMute())
                self.endpoint_volume.SetMute(0 if muted else 1, None)
                return True
            except Exception:
                return False
        if command == "volume_set":
            if self.endpoint_volume is None or event.value is None:
                return False
            try:
                value = max(0.0, min(1.0, float(event.value)))
                self.endpoint_volume.SetMasterVolumeLevelScalar(value, None)
                return True
            except Exception:
                return False
        return False


def execute(event: CommandEvent) -> bool:
    executor = ActionExecutor()
    return executor.execute(event)

