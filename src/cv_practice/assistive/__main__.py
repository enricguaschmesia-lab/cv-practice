from __future__ import annotations

import argparse

from cv_practice.assistive.app import run_assistive_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assistive Gesture Control Platform (AGCP).")
    parser.add_argument("--config", default=None, help="Path to yaml config file.")
    args = parser.parse_args()
    run_assistive_app(config_path=args.config)


if __name__ == "__main__":
    main()

