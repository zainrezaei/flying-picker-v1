#!/usr/bin/env python3
"""Simple homography calibration script.

Usage:
  python simple_homography_calibration.py

What it does:
- Opens the camera stream
- Lets you click 4 reference points in the shown order
- Computes homography (pixel -> world mm)
- Saves result to config/homography.json
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.coordinate_transform import (  # noqa: E402
    calibrate_homography_interactive,
    save_homography,
)
from src.vision.frame_source import FrameSource  # noqa: E402


WORLD_POINTS_MM = np.array(
    [
        [136.0, 933.0],
        [366.0, 933.0],
        [366.0, 1106.0],
        [136.0, 1106.0],
    ],
    dtype=np.float32,
)

OUTPUT_PATH = "config/homography.json"


def main() -> None:
    source = FrameSource(loop=False)
    try:
        result = calibrate_homography_interactive(source, WORLD_POINTS_MM)
    finally:
        source.release()

    save_homography(result, OUTPUT_PATH)
    print(f"\nSaved homography to: {OUTPUT_PATH}")
    print(f"Average reprojection error: {result.reprojection_error:.2f} mm")


if __name__ == "__main__":
    main()
