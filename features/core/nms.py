"""MEMBER 1 — Thresholding + Non-Maximum Suppression (NMS).

Written once, reused by every detector (Harris and Lambda-minus).

Steps:
  1. Threshold : keep pixels whose response > ratio * max(response).
  2. NMS       : keep a pixel only if it is the maximum inside its
                 (size x size) neighbourhood.  Implemented with a grey
                 dilation: dilate(R) holds the local max at every pixel,
                 so  R == dilate(R)  marks the local maxima.
  3. Top-N     : keep the strongest `max_keypoints` points.
"""
from __future__ import annotations

import cv2
import numpy as np

from .keypoints import Keypoints


def threshold_mask(response: np.ndarray, ratio: float) -> np.ndarray:
    peak = float(response.max())
    if peak <= 0:
        return np.zeros(response.shape, dtype=bool)
    return response > ratio * peak


def local_maxima_mask(response: np.ndarray, size: int) -> np.ndarray:
    size = max(3, int(size) | 1)                      # odd window >= 3
    kernel = np.ones((size, size), np.uint8)
    return response >= cv2.dilate(response, kernel)


def border_mask(shape: tuple[int, int], border: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if border * 2 < min(shape):
        mask[border:shape[0] - border, border:shape[1] - border] = True
    return mask


def select_keypoints(response: np.ndarray, threshold_ratio: float, nms_size: int,
                     max_keypoints: int, border: int = 0) -> Keypoints:
    mask = (threshold_mask(response, threshold_ratio)
            & local_maxima_mask(response, nms_size)
            & border_mask(response.shape, border))
    ys, xs = np.nonzero(mask)
    scores = response[ys, xs]
    order = np.argsort(-scores)[:max_keypoints]       # strongest first
    xy = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
    return Keypoints(xy=xy, scores=scores[order].astype(np.float32))
