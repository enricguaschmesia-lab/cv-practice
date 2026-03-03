from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from agcp.evaluation import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
