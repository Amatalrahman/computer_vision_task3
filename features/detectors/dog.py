"""Difference-of-Gaussians (DoG) keypoint detector — the classic SIFT keypoint
source (Lowe, 2004), usable here instead of Harris / λ- as the input to
features/descriptors/sift.py.

Unlike Harris and λ- (single scale: one structure tensor, one window size),
DoG searches a whole GAUSSIAN SCALE-SPACE PYRAMID and keeps points that are
extrema in space AND scale, so every keypoint gets a characteristic SCALE
(σ) alongside its (x, y). SIFT uses that scale to size each keypoint's own
descriptor patch instead of one fixed patch for every point (see the
`kps.scales` branch in descriptors/sift.py).

Pipeline, following Lowe's paper:
  1. OCTAVES: the image is halved at the start of every octave (`octaves`
     of them). Each octave is a stack of `scales_per_octave + 3` images,
     all blurred from the octave's base image with core.gaussian_blur, σ
     growing by k = 2**(1/scales_per_octave) at each step.
  2. DIFFERENCE OF GAUSSIANS: subtracting adjacent blur levels gives
     `scales_per_octave + 2` DoG images per octave — a cheap approximation
     of the scale-normalised Laplacian of Gaussian, which responds
     strongly to blob-like structures.
  3. EXTREMA: a candidate keypoint is any pixel that is a local min/max of
     its 3x3x3 neighbourhood (8 neighbours in its own DoG layer, 9 in the
     layer above, 9 in the layer below).
  4. REJECTION: low-contrast points (|D| below `contrast_threshold`) and
     edge-like points are discarded. The edge test reuses the Harris idea
     on the DoG layer's own 2x2 Hessian: an edge has one large and one
     tiny curvature, a good blob keypoint has two comparably large ones.
  5. Coordinates and scale are mapped back to the original image, and the
     strongest `max_keypoints` points (by |D|) are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core import ImageContext, Keypoints, Timer, gaussian_blur, param
from .base import DetectionResult


@dataclass(frozen=True)
class DoGParams:
    pre_sigma: float = param(1.6, "Base σ", 0.8, 3.0, 0.1,
                             "σ of the first scale-space level; also used by SIFT for its gradients")
    octaves: int = param(4, "Octaves", 1, 8, 1, "Number of times the image is halved")
    scales_per_octave: int = param(3, "Scales / octave", 2, 5, 1,
                                   "s in Lowe's paper -> s+3 blur levels, s+2 DoG images per octave")
    contrast_threshold: float = param(0.02, "Contrast threshold", 0.0, 0.2, 0.002,
                                      "Reject DoG extrema with |D| below this")
    edge_ratio: float = param(10.0, "Edge ratio (r)", 3.0, 30.0, 1.0,
                              "Reject edge-like extrema: trace²·r ≥ (r+1)²·det of the DoG Hessian")
    max_keypoints: int = param(800, "Max keypoints", 10, 5000, 50, "Keep only the strongest N points")
    border: int = param(6, "Ignore border (px)", 0, 32, 1,
                        "Discard points too close to an octave's edge")


class DoGDetector:
    """Duck-types FeatureDetector (name / params_cls / detect) without inheriting
    it: Harris and λ- share ONE structure tensor per image (detectors/base.py's
    template method), DoG needs a whole scale-space pyramid instead, so it
    builds its own DetectionResult directly."""

    name = "DoG"
    params_cls = DoGParams

    def __init__(self, params: DoGParams | None = None) -> None:
        self.params = params or DoGParams()

    def detect(self, ctx: ImageContext) -> DetectionResult:
        p = self.params
        k = 2.0 ** (1.0 / p.scales_per_octave)
        levels_per_octave = p.scales_per_octave + 3        # blur levels -> +2 DoG images per octave

        octave_img = ctx.image.gray
        all_xy, all_scores, all_sigmas = [], [], []
        first_response = None
        pyramid_ms = extrema_ms = 0.0

        for o in range(p.octaves):
            if min(octave_img.shape) < 2 * p.border + 8:    # too small to search further
                break

            with Timer() as pyramid_timer:
                blur_levels = [gaussian_blur(octave_img, p.pre_sigma * (k ** i))
                               for i in range(levels_per_octave)]
                dog_levels = [blur_levels[i + 1] - blur_levels[i]
                              for i in range(levels_per_octave - 1)]
            pyramid_ms += pyramid_timer.ms

            if first_response is None:
                first_response = cv2.resize(
                    dog_levels[1], (ctx.image.gray.shape[1], ctx.image.gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR)

            height, width = octave_img.shape
            kernel = np.ones((3, 3), np.uint8)
            for level in range(1, len(dog_levels) - 1):
                previous, current, following = dog_levels[level - 1:level + 2]
                local_max = np.maximum.reduce(
                    [cv2.dilate(layer, kernel) for layer in (previous, current, following)])
                local_min = np.minimum.reduce(
                    [cv2.erode(layer, kernel) for layer in (previous, current, following)])

                with Timer() as extrema_timer:
                    candidates = np.argwhere(
                        ((current == local_max) | (current == local_min))
                        & (np.abs(current) >= p.contrast_threshold))
                    if candidates.size:
                        cy, cx = candidates.T
                        valid = ((cy >= max(1, p.border)) & (cy < min(height - 1, height - p.border))
                                 & (cx >= max(1, p.border)) & (cx < min(width - 1, width - p.border)))
                        cy, cx = cy[valid], cx[valid]
                        values = current[cy, cx]
                        is_max = current[cy, cx] == local_max[cy, cx]
                        is_min = current[cy, cx] == local_min[cy, cx]
                        keep = is_max | is_min
                        rank = (level * height + cy) * width + cx

                        # Resolve flat extrema deterministically so one plateau
                        # cannot produce a cluster of duplicate keypoints.
                        for dz in (-1, 0, 1):
                            neighbor_layer = dog_levels[level + dz]
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    if dz == dy == dx == 0:
                                        continue
                                    neighbor_y, neighbor_x = cy + dy, cx + dx
                                    neighbor = neighbor_layer[neighbor_y, neighbor_x]
                                    neighbor_rank = ((level + dz) * height + neighbor_y) * width + neighbor_x
                                    keep &= ((is_max & ((values > neighbor)
                                                       | ((values == neighbor) & (rank < neighbor_rank))))
                                             | (is_min & ((values < neighbor)
                                                          | ((values == neighbor) & (rank < neighbor_rank)))))

                        cy, cx, values = cy[keep], cx[keep], values[keep]
                        if len(cx):
                            dxx = current[cy, cx + 1] + current[cy, cx - 1] - 2 * values
                            dyy = current[cy + 1, cx] + current[cy - 1, cx] - 2 * values
                            dxy = (current[cy + 1, cx + 1] - current[cy + 1, cx - 1]
                                   - current[cy - 1, cx + 1] + current[cy - 1, cx - 1]) * 0.25
                            det = dxx * dyy - dxy * dxy
                            trace = dxx + dyy
                            edge_limit = (p.edge_ratio + 1) ** 2 / p.edge_ratio
                            non_edge = (det > 1e-12) & (trace * trace / np.maximum(det, 1e-12) < edge_limit)
                            cy, cx, values = cy[non_edge], cx[non_edge], values[non_edge]

                        if len(cx):
                            scale = p.pre_sigma * (k ** (level + 0.5)) * (2 ** o)
                            all_xy.append(np.column_stack((cx, cy)).astype(np.float32) * (2 ** o))
                            all_scores.append(values.astype(np.float32))
                            all_sigmas.append(np.full(len(cx), scale, np.float32))
                extrema_ms += extrema_timer.ms

            octave_img = cv2.resize(octave_img, None, fx=0.5, fy=0.5,
                                    interpolation=cv2.INTER_AREA)

        if all_xy:
            xy = np.concatenate(all_xy)
            scores = np.concatenate(all_scores)
            sigmas = np.concatenate(all_sigmas)
            strongest = np.argsort(np.abs(scores))[::-1][:p.max_keypoints]
            keypoints = Keypoints(xy[strongest], scores[strongest],
                                  scales=sigmas[strongest])
        else:
            keypoints = Keypoints.empty()

        response = first_response if first_response is not None else np.zeros_like(ctx.image.gray)
        return DetectionResult(self.name, p, keypoints, response,
                               pyramid_ms, 0.0, extrema_ms, 0.0)
            