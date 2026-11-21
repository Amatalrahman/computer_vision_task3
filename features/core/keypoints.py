"""Shared keypoint container — the 'contract' between all four members.

Detectors (M1, M2) produce it, SIFT (M3) adds orientations, matching (M4) reads it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoints:
    xy: np.ndarray                          # (N, 2) float32, columns = (x, y)
    scores: np.ndarray                      # (N,) detector response
    orientations: np.ndarray | None = None  # (N,) radians, filled by SIFT

    def __len__(self) -> int:
        return len(self.xy)

    @classmethod
    def empty(cls) -> "Keypoints":
        return cls(np.zeros((0, 2), np.float32), np.zeros(0, np.float32))

    def with_orientations(self, orientations: np.ndarray) -> "Keypoints":
        return Keypoints(self.xy, self.scores, orientations.astype(np.float32))
