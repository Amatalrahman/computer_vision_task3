"""Shared keypoint container — the 'contract' between all four members.

Detectors (Harris, λ-, DoG) produce it, SIFT adds orientations, matching reads it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoints:
    xy: np.ndarray                          # (N, 2) float32, columns = (x, y)
    scores: np.ndarray                      # (N,) detector response
    orientations: np.ndarray | None = None  # (N,) radians, filled by SIFT
    scales: np.ndarray | None = None        # (N,) characteristic scale σ, filled by DoG only

    def __len__(self) -> int:
        return len(self.xy)

    @classmethod
    def empty(cls) -> "Keypoints":
        return cls(np.zeros((0, 2), np.float32), np.zeros(0, np.float32))

    def with_orientations(self, orientations: np.ndarray) -> "Keypoints":
        return Keypoints(self.xy, self.scores, orientations.astype(np.float32), self.scales)