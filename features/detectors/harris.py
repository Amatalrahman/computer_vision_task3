"""MEMBER 1 — Harris corner detector.

    R = det(M) - k * trace(M)^2
      = (Sxx*Syy - Sxy^2) - k * (Sxx + Syy)^2

  R >> 0  -> corner      (both eigenvalues large)
  R << 0  -> edge        (one eigenvalue large)
  |R| ~ 0 -> flat region (both small)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import StructureTensor, param
from .base import DetectorParams, FeatureDetector


@dataclass(frozen=True)
class HarrisParams(DetectorParams):
    k: float = param(0.04, "Harris k", 0.01, 0.20, 0.005,
                     "Empirical constant, typically 0.04 – 0.06")


class HarrisDetector(FeatureDetector):
    name = "Harris"
    params_cls = HarrisParams

    def compute_response(self, tensor: StructureTensor) -> np.ndarray:
        k = self.params.k
        return tensor.determinant() - k * tensor.trace() ** 2
