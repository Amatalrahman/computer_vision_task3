"""ImageContext — one loaded image + cache of its shared intermediate results.

This is what removes redundancy between members:
  * gradients are computed once per smoothing sigma,
  * the structure tensor once per (smoothing sigma, window sigma),
and Harris, Lambda-minus and SIFT all read from this cache.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .gradients import ImageGradients, StructureTensor
from .image import ImageData

if TYPE_CHECKING:  # only for type hints, avoids circular imports
    from ..descriptors.sift import DescriptorResult
    from ..detectors.base import DetectionResult


class ImageContext:
    def __init__(self, image: ImageData) -> None:
        self.image = image
        self._gradients: dict[float, ImageGradients] = {}
        self._tensors: dict[tuple[float, float], StructureTensor] = {}
        # latest results, keyed by detector name ("Harris", "Lambda-")
        self.detections: dict[str, "DetectionResult"] = {}
        self.descriptors: dict[str, "DescriptorResult"] = {}

    @property
    def name(self) -> str:
        return self.image.name

    def gradients(self, pre_sigma: float) -> ImageGradients:
        key = round(float(pre_sigma), 4)
        if key not in self._gradients:
            self._gradients[key] = ImageGradients.compute(self.image.gray, key)
        return self._gradients[key]

    def structure_tensor(self, pre_sigma: float, window_sigma: float) -> StructureTensor:
        key = (round(float(pre_sigma), 4), round(float(window_sigma), 4))
        if key not in self._tensors:
            self._tensors[key] = StructureTensor.compute(self.gradients(key[0]), key[1])
        return self._tensors[key]
