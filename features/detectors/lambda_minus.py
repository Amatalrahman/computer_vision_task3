"""MEMBER 2 — Lambda-minus (λ-) feature detector  (Shi–Tomasi "minimum eigenvalue").

The structure tensor M (shared, see core/gradients.py) is a symmetric 2x2 matrix
        | a  b |      a = Sxx,  b = Sxy,  c = Syy
    M = | b  c |
Its two eigenvalues have a closed form (no np.linalg loop over pixels):

    λ± = (a + c)/2  ±  sqrt( ((a - c)/2)^2 + b^2 )

A pixel is a good feature when the SMALLER eigenvalue λ- is large,
i.e. the intensity changes strongly in every direction.  Response = λ-.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import StructureTensor, param
from .base import DetectorParams, FeatureDetector


def eigenvalues(tensor: StructureTensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (λ+, λ-) for every pixel."""
    half_trace = 0.5 * (tensor.sxx + tensor.syy)
    root = np.sqrt((0.5 * (tensor.sxx - tensor.syy)) ** 2 + tensor.sxy ** 2)
    return half_trace + root, half_trace - root


@dataclass(frozen=True)
class LambdaParams(DetectorParams):
    # Same knobs as the base detector (threshold & NMS reused from Member 1).
    # Only the default threshold differs: λ- grows linearly with contrast,
    # Harris R grows quadratically, so λ- needs a higher relative threshold.
    threshold_ratio: float = param(0.05, "Threshold (× max)", 0.0001, 0.5, 0.005,
                                   "Keep λ- above this fraction of the maximum")


class LambdaMinusDetector(FeatureDetector):
    name = "Lambda-"
    params_cls = LambdaParams

    def compute_response(self, tensor: StructureTensor) -> np.ndarray:
        _, lam_minus = eigenvalues(tensor)
        return lam_minus
