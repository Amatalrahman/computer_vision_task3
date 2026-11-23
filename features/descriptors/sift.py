"""MEMBER 3 — SIFT descriptor (from scratch).

Input : keypoints from a detector (Harris or λ-) + the SHARED image gradients
        (magnitude / orientation are reused from core — not recomputed).
Output: one 128-D vector per keypoint.

Steps
  1. Orientation assignment
       36-bin histogram of gradient orientations around the keypoint,
       weighted by magnitude x Gaussian.  The peak = dominant orientation θ
       (refined with a parabola through the 3 bins around the peak).
  2. Descriptor
       * take a patch (default 16x16) around the keypoint, ROTATED by -θ
         (=> rotation invariance)
       * split it into 4 x 4 cells (each 4x4 px)
       * per cell: 8-bin orientation histogram (angles relative to θ),
         weighted by magnitude x Gaussian(σ = half patch)
       * 4 x 4 cells x 8 bins = 128 values
  3. Normalisation
       L2-normalise -> clip values at 0.2 -> L2-normalise again
       (=> robustness to illumination changes)

Efficiency: all keypoints are processed together with NumPy (no Python loop
over pixels); histograms are built with a single np.bincount.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import ImageContext, Keypoints, Timer, param
from ..detectors.base import DetectionResult

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SIFTParams:
    patch_size: int = param(16, "Patch size (px)", 8, 64, 4,
                            "Side of the square region described (split into 4×4 cells)")
    rotation_invariant: bool = param(True, "Rotation invariant", tip="Rotate patch to the dominant orientation")
    clip_value: float = param(0.2, "Clip value", 0.05, 1.0, 0.05,
                              "Descriptor values are clipped here, then re-normalised")


@dataclass
class DescriptorResult:
    detector: str
    params: SIFTParams
    keypoints: Keypoints          # same points, now with orientations
    descriptors: np.ndarray       # (N, 128) float32
    orientation_ms: float
    descriptor_ms: float

    @property
    def total_ms(self) -> float:
        return self.orientation_ms + self.descriptor_ms


class SIFTDescriptor:
    NUM_CELLS = 4                 # 4 x 4 grid
    NUM_BINS = 8                  # 8 orientations per cell
    ORI_BINS = 36                 # orientation-assignment histogram
    DIM = NUM_CELLS * NUM_CELLS * NUM_BINS   # 128

    def __init__(self, params: SIFTParams | None = None) -> None:
        self.params = params or SIFTParams()

    # ================================================================ public
    def compute(self, ctx: ImageContext, detection: DetectionResult) -> DescriptorResult:
        kps = detection.keypoints
        n = len(kps)
        pre_sigma = float(getattr(detection.params, "pre_sigma", 1.0))
        grads = ctx.gradients(pre_sigma)                           # shared, cached
        if n == 0:
            return DescriptorResult(detection.detector, self.params, kps,
                                    np.zeros((0, self.DIM), np.float32), 0.0, 0.0)

        if kps.scales is None:
            scale_groups = [(np.arange(n), self.params.patch_size / 2.0)]
        else:
            scale_groups = []
            # Keep the effective support bounded for unusually large DoG scales.
            max_scale_factor = 3.0
            for scale in np.unique(kps.scales):
                indices = np.flatnonzero(kps.scales == scale)
                factor = min(float(scale) / max(pre_sigma, 1e-6), max_scale_factor)
                scale_groups.append((indices, self.params.patch_size / 2.0 * factor))

        theta = np.zeros(n, np.float32)
        desc = np.zeros((n, self.DIM), np.float32)

        with Timer() as t_ori:
            for indices, half in scale_groups:
                desc_radius = int(np.ceil(half * np.sqrt(2))) + 1
                ori_radius = int(np.ceil(half))
                pad = max(desc_radius, ori_radius)
                mag = np.pad(grads.magnitude, pad)
                ori = np.pad(grads.orientation, pad)
                centers = np.round(kps.xy[indices]).astype(np.int64) + pad
                if self.params.rotation_invariant:
                    theta[indices] = self._dominant_orientations(mag, ori, centers, ori_radius)

        with Timer() as t_desc:
            for indices, half in scale_groups:
                desc_radius = int(np.ceil(half * np.sqrt(2))) + 1
                pad = desc_radius
                mag = np.pad(grads.magnitude, pad)
                ori = np.pad(grads.orientation, pad)
                centers = np.round(kps.xy[indices]).astype(np.int64) + pad
                desc[indices] = self._descriptors(
                    mag, ori, centers, theta[indices], desc_radius, half)

        return DescriptorResult(detection.detector, self.params, kps.with_orientations(theta),
                                desc, t_ori.ms, t_desc.ms)

    # ============================================================== helpers
    @staticmethod
    def _patches(arr: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
        """Cut an (N, 2r+1, 2r+1) stack of square patches around every center."""
        offs = np.arange(-radius, radius + 1)
        rows = centers[:, 1, None, None] + offs[None, :, None]
        cols = centers[:, 0, None, None] + offs[None, None, :]
        return arr[rows, cols]

    @staticmethod
    def _grid(radius: int) -> tuple[np.ndarray, np.ndarray]:
        offs = np.arange(-radius, radius + 1, dtype=np.float32)
        dy, dx = np.meshgrid(offs, offs, indexing="ij")
        return dx, dy

    # ---------------------------------------------- step 1: orientation
    def _dominant_orientations(self, mag, ori, centers, radius) -> np.ndarray:
        n, nb = len(centers), self.ORI_BINS
        dx, dy = self._grid(radius)
        sigma = radius / 3.0
        gauss = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

        weights = self._patches(mag, centers, radius) * gauss
        angles = self._patches(ori, centers, radius) % TWO_PI          # [0, 2π)
        bins = (angles * nb / TWO_PI).astype(np.int64) % nb

        index = np.arange(n)[:, None, None] * nb + bins                 # flat (kp, bin)
        hist = np.bincount(index.ravel(), weights.ravel(), minlength=n * nb).reshape(n, nb)
        hist = (np.roll(hist, 1, 1) + hist + np.roll(hist, -1, 1)) / 3.0  # circular smoothing

        peak = hist.argmax(axis=1)
        rows = np.arange(n)
        left, mid, right = hist[rows, (peak - 1) % nb], hist[rows, peak], hist[rows, (peak + 1) % nb]
        denom = left - 2 * mid + right
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (left - right) / denom, 0.0)  # parabola fit
        return (((peak + 0.5 + shift) * TWO_PI / nb) % TWO_PI).astype(np.float32)

    # ---------------------------------------------- step 2 + 3: descriptor
    def _descriptors(self, mag, ori, centers, theta, radius, half) -> np.ndarray:
        n, nc, nb = len(centers), self.NUM_CELLS, self.NUM_BINS
        cell = (2 * half) / nc
        dx, dy = self._grid(radius)

        # Rotate the sampling grid into the keypoint's own frame (by -θ)
        cos_t = np.cos(theta)[:, None, None]
        sin_t = np.sin(theta)[:, None, None]
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy

        # Which of the 4x4 cells does every pixel fall into?
        col = np.floor((xr + half) / cell).astype(np.int64)
        row = np.floor((yr + half) / cell).astype(np.int64)
        inside = (col >= 0) & (col < nc) & (row >= 0) & (row < nc)

        # Orientation relative to θ -> one of 8 bins
        rel = (self._patches(ori, centers, radius) - theta[:, None, None]) % TWO_PI
        obin = (rel * nb / TWO_PI).astype(np.int64) % nb

        # Weight = gradient magnitude x Gaussian(σ = half patch width)
        weight = self._patches(mag, centers, radius) * np.exp(-(xr ** 2 + yr ** 2) / (2 * half ** 2))

        kp = np.broadcast_to(np.arange(n)[:, None, None], col.shape)
        index = ((kp * nc + row) * nc + col) * nb + obin                # flat (kp, row, col, bin)
        desc = np.bincount(index[inside], weight[inside], minlength=n * self.DIM)
        desc = desc.reshape(n, self.DIM)
        return self._normalise(desc, self.params.clip_value).astype(np.float32)

    @staticmethod
    def _normalise(desc: np.ndarray, clip: float) -> np.ndarray:
        desc = desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
        desc = np.minimum(desc, clip)
        return desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
