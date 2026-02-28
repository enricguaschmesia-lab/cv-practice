from __future__ import annotations

from pathlib import Path

from cv_practice.assistive.calibration import load_profile, save_profile
from cv_practice.assistive.types import UserProfile


def test_profile_roundtrip() -> None:
    path = Path("outputs/assistive/test_profile_roundtrip.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = UserProfile(name="alice", pinch_max=0.61, open_min=1.45)
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.name == "alice"
    assert abs(loaded.pinch_max - 0.61) < 1e-9
    assert abs(loaded.open_min - 1.45) < 1e-9
