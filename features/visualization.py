"""Shared drawing helpers (used by all tabs — written once).

All functions take/return RGB uint8 images and never modify their input.
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .core import Keypoints
from .matching import MatchResult

GREEN = (46, 204, 113)
ORANGE = (255, 159, 67)
CYAN = (0, 210, 255)


def _marker_radius(shape) -> int:
    return max(3, int(round(min(shape[:2]) / 160)))


def _palette(n: int, seed: int = 7) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    hues = rng.random(n)
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.85, 1.0)) for h in hues]


def draw_keypoints(rgb: np.ndarray, kps: Keypoints, color=GREEN,
                   show_orientation: bool = False) -> np.ndarray:
    out = rgb.copy()
    r = _marker_radius(out.shape)
    thickness = max(1, r // 3)
    for i, (x, y) in enumerate(np.round(kps.xy).astype(int)):
        cv2.circle(out, (x, y), r, color, thickness, cv2.LINE_AA)
        if show_orientation and kps.orientations is not None:
            t = kps.orientations[i]
            tip = (int(round(x + 2.5 * r * np.cos(t))), int(round(y + 2.5 * r * np.sin(t))))
            cv2.line(out, (x, y), tip, ORANGE, thickness, cv2.LINE_AA)
    return out


def response_heatmap(response: np.ndarray) -> np.ndarray:
    """Detector response as a colour map (log-scaled so weak corners stay visible)."""
    r = np.maximum(response, 0)
    peak = float(r.max())
    if peak <= 0:
        return np.zeros((*response.shape, 3), np.uint8)
    v = np.log1p(r / (peak * 1e-4)) / np.log1p(1e4)
    bgr = cv2.applyColorMap((255 * v).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return bgr[:, :, ::-1].copy()


def draw_descriptor(desc: np.ndarray, size: int = 240, cells: int = 4, bins: int = 8) -> np.ndarray:
    """Classic SIFT picture: a 4x4 grid, each cell shows its 8 orientation arrows."""
    out = np.full((size, size, 3), 18, np.uint8)
    step = size / cells
    hist = desc.reshape(cells, cells, bins)
    peak = max(float(hist.max()), 1e-12)
    for r in range(cells):
        for c in range(cells):
            x0, y0 = int(c * step), int(r * step)
            cv2.rectangle(out, (x0, y0), (int(x0 + step), int(y0 + step)), (60, 66, 80), 1)
            cx, cy = x0 + step / 2, y0 + step / 2
            for b in range(bins):
                ang = b * 2 * np.pi / bins
                length = 0.48 * step * hist[r, c, b] / peak
                tip = (int(round(cx + length * np.cos(ang))), int(round(cy + length * np.sin(ang))))
                cv2.line(out, (int(cx), int(cy)), tip, CYAN, 2, cv2.LINE_AA)
    return out


def draw_matches(rgb_a: np.ndarray, kps_a: Keypoints, rgb_b: np.ndarray, kps_b: Keypoints,
                 result: MatchResult, max_lines: int = 150, gap: int = 12) -> np.ndarray:
    """Place A and B side by side and connect matched points with coloured lines."""
    ha, wa = rgb_a.shape[:2]
    hb, wb = rgb_b.shape[:2]
    canvas = np.full((max(ha, hb), wa + gap + wb, 3), 24, np.uint8)
    canvas[:ha, :wa] = rgb_a
    canvas[:hb, wa + gap:] = rgb_b

    shown = result.best(max_lines)
    r = _marker_radius(canvas.shape)
    thickness = max(1, r // 3)
    for color, i, j in zip(_palette(len(shown)), shown.query_idx, shown.train_idx):
        pa = tuple(np.round(kps_a.xy[i]).astype(int))
        xb, yb = np.round(kps_b.xy[j]).astype(int)
        pb = (int(xb) + wa + gap, int(yb))
        cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pa, r, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pb, r, color, thickness, cv2.LINE_AA)
    return canvas
