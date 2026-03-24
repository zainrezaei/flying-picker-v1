"""
shape_classifier.py — Hu Moment shape matching against reference images.

Loads reference part images (Part_*.jpg / Part_*.png) from a directory,
extracts their largest contour, and classifies detected contours by
comparing Hu Moment signatures via cv2.matchShapes().

Usage
-----
    classifier = ShapeClassifier("path/to/reference/dir", threshold=0.20)
    part_id, score = classifier.classify(contour)
"""

import os
import glob
import logging
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of shape classification."""
    part_id: str          # e.g. "Part_1"
    score: float          # matchShapes score (lower = better match)


class ShapeClassifier:
    """Classify contours by matching against reference part images.

    Parameters
    ----------
    reference_dir : str
        Directory containing Part_*.jpg / Part_*.png reference images.
    threshold : float
        Maximum matchShapes score to accept a classification.
        Scores above this are rejected (returns None).
    method : int
        cv2.matchShapes comparison method (default 1 = CONTOURS_MATCH_I2).
    blur_kernel : int
        Gaussian blur kernel applied to reference images before thresholding.
    thresh_value : int
        Binary threshold applied to reference images.
    """

    def __init__(
        self,
        reference_dir: str,
        threshold: float = 0.20,
        method: int = 1,
        blur_kernel: int = 5,
        thresh_value: int = 160,
    ):
        self.threshold = threshold
        self.method = method
        self._references: list[tuple[str, np.ndarray]] = []

        self._load_references(reference_dir, blur_kernel, thresh_value)

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_references(
        self, reference_dir: str, blur_kernel: int, thresh_value: int
    ) -> None:
        """Discover and load all Part_* images from the reference directory."""
        patterns = ["Part_*.jpg", "Part_*.jpeg", "Part_*.png"]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(
                glob.glob(os.path.join(reference_dir, pattern))
            )

        # Sort so Part_1 < Part_2 < Part_3 etc.
        image_paths.sort()

        if not image_paths:
            logger.warning(
                f"[ShapeClassifier] No Part_* images found in '{reference_dir}'"
            )
            return

        for path in image_paths:
            contour = self._extract_contour(path, blur_kernel, thresh_value)
            if contour is not None:
                # Derive part name from filename: "Part_1.jpg" → "Part_1"
                part_id = os.path.splitext(os.path.basename(path))[0]
                self._references.append((part_id, contour))
                logger.info(
                    f"[ShapeClassifier] Loaded reference '{part_id}' "
                    f"from {os.path.basename(path)} "
                    f"(contour area={cv.contourArea(contour):.0f} px²)"
                )

        print(
            f"[ShapeClassifier] {len(self._references)} reference part(s) loaded "
            f"from '{reference_dir}'"
        )

    @staticmethod
    def _extract_contour(
        image_path: str, blur_kernel: int, thresh_value: int
    ) -> Optional[np.ndarray]:
        """Read an image, threshold it, and return the largest contour."""
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(
                f"[ShapeClassifier] Could not read image: {image_path}"
            )
            return None

        # Optional blur to smooth edges
        if blur_kernel > 1:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            img = cv.GaussianBlur(img, (k, k), 0)

        _, thresh = cv.threshold(img, thresh_value, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning(
                f"[ShapeClassifier] No contours found in: {image_path}"
            )
            return None

        return max(contours, key=cv.contourArea)

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def classify(
        self, contour: np.ndarray
    ) -> Optional[ClassificationResult]:
        """Compare a contour against all references and return the best match.

        Parameters
        ----------
        contour : np.ndarray
            The detected object's contour points.

        Returns
        -------
        ClassificationResult or None
            The best-matching part and its score, or None if no match
            is within the threshold (or no references are loaded).
        """
        if not self._references:
            return None

        best_id: Optional[str] = None
        best_score = float("inf")

        for part_id, ref_contour in self._references:
            score = cv.matchShapes(ref_contour, contour, self.method, 0.0)
            if score < best_score:
                best_score = score
                best_id = part_id

        if best_score > self.threshold:
            return None

        return ClassificationResult(part_id=best_id, score=best_score)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_references(self) -> int:
        """Number of loaded reference parts."""
        return len(self._references)

    @property
    def part_names(self) -> list[str]:
        """List of loaded part names."""
        return [name for name, _ in self._references]

    def __repr__(self) -> str:
        return (
            f"ShapeClassifier(parts={self.part_names}, "
            f"threshold={self.threshold}, method={self.method})"
        )
