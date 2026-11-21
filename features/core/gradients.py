"""MEMBER 1 — Gradients (Sobel) and the structure tensor.

These are the SHARED building blocks of the whole project:
  * Harris (M1) and Lambda-minus (M2) both need the structure tensor M.
  * SIFT (M3) needs gradient magnitude + orientation.
They are computed once per image and cached by `ImageContext`.

Structure tensor (second-moment matrix) at every pixel:
        | Sxx  Sxy |        Sxx = G_sigma * (Ix*Ix)
    M = |          |  with  Sxy = G_sigma * (Ix*Iy)
        | Sxy  Syy |        Syy = G_sigma * (Iy*Iy)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .timer import Timer

# Sobel operators written explicitly (from scratch — no cv2.Sobel)
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()

_BORDER = cv2.BORDER_REFLECT_101


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """1-D Gaussian kernel, radius = 3*sigma, normalised to sum 1."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing (our kernel, OpenCV only does the convolution)."""
    if sigma <= 0:
        return img
    k = gaussian_kernel_1d(sigma)
    return cv2.sepFilter2D(img, cv2.CV_32F, k, k, borderType=_BORDER)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, cv2.CV_32F, kernel, borderType=_BORDER)


@dataclass
class ImageGradients:
    ix: np.ndarray
    iy: np.ndarray
    magnitude: np.ndarray
    orientation: np.ndarray    # radians in [-pi, pi]
    pre_sigma: float
    elapsed_ms: float

    @classmethod
    def compute(cls, gray: np.ndarray, pre_sigma: float) -> "ImageGradients":
        with Timer() as t:
            smooth = gaussian_blur(gray, pre_sigma)      # reduce noise first
            ix = convolve(smooth, SOBEL_X)               # dI/dx
            iy = convolve(smooth, SOBEL_Y)               # dI/dy
            magnitude = np.sqrt(ix * ix + iy * iy)
            orientation = np.arctan2(iy, ix)
        return cls(ix, iy, magnitude, orientation, pre_sigma, t.ms)


@dataclass
class StructureTensor:
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray
    window_sigma: float
    elapsed_ms: float

    @classmethod
    def compute(cls, grads: ImageGradients, window_sigma: float) -> "StructureTensor":
        with Timer() as t:
            sxx = gaussian_blur(grads.ix * grads.ix, window_sigma)
            sxy = gaussian_blur(grads.ix * grads.iy, window_sigma)
            syy = gaussian_blur(grads.iy * grads.iy, window_sigma)
        return cls(sxx, sxy, syy, window_sigma, t.ms)

    def determinant(self) -> np.ndarray:
        return self.sxx * self.syy - self.sxy * self.sxy

    def trace(self) -> np.ndarray:
        return self.sxx + self.syy
