from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AssistiveConfig:
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    model_path: str = "models/hand_landmarker.task"
    profile_path: str = "configs/profiles/default.json"
    output_dir: str = "outputs/assistive"
    show_debug: bool = False


def load_assistive_config(path: str | Path | None = None) -> AssistiveConfig:
    config = AssistiveConfig()
    if path is None:
        root = Path(__file__).resolve().parents[3]
        path = root / "configs" / "assistive.default.yaml"
    p = Path(path)
    if not p.exists():
        return config

    try:
        import yaml  # type: ignore
    except Exception:
        return config

    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    for key, value in payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

