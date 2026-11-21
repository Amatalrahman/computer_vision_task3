"""MEMBER 1 — Base class for every corner/feature detector (Template Method pattern).

`detect()` is identical for all detectors:
    gradients -> structure tensor -> RESPONSE -> threshold + NMS -> keypoints
Only the RESPONSE formula differs, so a subclass implements just
`compute_response(tensor)`.  Harris (M1) and Lambda-minus (M2) are each
~10 lines because everything else lives here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core import ImageContext, Keypoints, StructureTensor, Timer, param, select_keypoints


@dataclass(frozen=True)
class DetectorParams:
    pre_sigma: float = param(1.0, "Gradient smoothing σ", 0.0, 5.0, 0.1,
                             "Gaussian blur before Sobel (noise reduction)")
    window_sigma: float = param(1.5, "Window σ", 0.5, 6.0, 0.1,
                                "Gaussian window used to sum the structure tensor")
    threshold_ratio: float = param(0.01, "Threshold (× max)", 0.0001, 0.5, 0.001,
                                   "Keep responses above this fraction of the maximum")
    nms_size: int = param(7, "NMS window (px)", 3, 51, 2,
                          "Neighbourhood size for non-maximum suppression")
    max_keypoints: int = param(800, "Max keypoints", 10, 5000, 50,
                               "Keep only the strongest N points")
    border: int = param(8, "Ignore border (px)", 0, 64, 1,
                        "Discard points too close to the image edge")


@dataclass
class DetectionResult:
    detector: str
    params: DetectorParams
    keypoints: Keypoints
    response: np.ndarray
    gradients_ms: float          # shared step (Sobel)
    tensor_ms: float             # shared step (structure tensor)
    response_ms: float           # detector-specific formula
    nms_ms: float                # threshold + NMS

    @property
    def total_ms(self) -> float:
        """Full time to generate the points from the raw image."""
        return self.gradients_ms + self.tensor_ms + self.response_ms + self.nms_ms


class FeatureDetector(ABC):
    name: ClassVar[str]
    params_cls: ClassVar[type[DetectorParams]] = DetectorParams

    def __init__(self, params: DetectorParams | None = None) -> None:
        self.params = params or self.params_cls()

    def detect(self, ctx: ImageContext) -> DetectionResult:
        p = self.params
        grads = ctx.gradients(p.pre_sigma)                         # cached / shared
        tensor = ctx.structure_tensor(p.pre_sigma, p.window_sigma)  # cached / shared

        with Timer() as t_resp:
            response = self.compute_response(tensor)
        with Timer() as t_nms:
            kps = select_keypoints(response, p.threshold_ratio, p.nms_size,
                                   p.max_keypoints, p.border)

        return DetectionResult(self.name, p, kps, response,
                               grads.elapsed_ms, tensor.elapsed_ms, t_resp.ms, t_nms.ms)

    @abstractmethod
    def compute_response(self, tensor: StructureTensor) -> np.ndarray:
        """Return a per-pixel 'cornerness' map (larger = stronger feature)."""
