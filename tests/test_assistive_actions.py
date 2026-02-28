from __future__ import annotations

from cv_practice.assistive.actions import ActionExecutor
from cv_practice.assistive.types import CommandEvent


class _FakeEndpoint:
    def __init__(self) -> None:
        self.vol = 0.0
        self.mute = 0

    def SetMasterVolumeLevelScalar(self, value: float, _ctx) -> None:
        self.vol = value

    def GetMute(self) -> int:
        return self.mute

    def SetMute(self, value: int, _ctx) -> None:
        self.mute = value


def test_action_volume_set_routes_to_endpoint() -> None:
    endpoint = _FakeEndpoint()
    exe = ActionExecutor(endpoint_volume=endpoint)
    ok = exe.execute(CommandEvent(command="volume_set", timestamp_ms=1, value=0.42))
    assert ok is True
    assert abs(endpoint.vol - 0.42) < 1e-9


def test_action_toggle_mute_routes_to_endpoint() -> None:
    endpoint = _FakeEndpoint()
    exe = ActionExecutor(endpoint_volume=endpoint)
    ok = exe.execute(CommandEvent(command="volume_toggle_mute", timestamp_ms=1))
    assert ok is True
    assert endpoint.mute == 1

