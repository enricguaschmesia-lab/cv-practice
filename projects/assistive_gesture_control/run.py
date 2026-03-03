from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from agcp.app import run_assistive_app

    parser = argparse.ArgumentParser(description="Run Assistive Gesture Control Platform (AGCP).")
    parser.add_argument("--config", default=None, help="Path to yaml config file.")
    args = parser.parse_args()
    run_assistive_app(config_path=args.config)


if __name__ == "__main__":
    main()
