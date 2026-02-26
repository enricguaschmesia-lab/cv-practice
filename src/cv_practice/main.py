from __future__ import annotations

from pathlib import Path

from cv_practice.io.image_io import read_bgr
from cv_practice.pipelines.baseline import BaselineConfig, run_baseline
from cv_practice.viz.plot import save_gray


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sample = root / "data" / "samples" / "sample.jpg"
    out_dir = root / "outputs" / "figures"

    bgr = read_bgr(sample)
    results = run_baseline(bgr, BaselineConfig(blur_ksize=5, canny_low=80, canny_high=160))

    save_gray(results["edges"], out_dir / "edges.png")
    print(f"Saved: {out_dir / 'edges.png'}")


if __name__ == "__main__":
    main()