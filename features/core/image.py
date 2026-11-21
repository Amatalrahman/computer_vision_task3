"""MEMBER 1 — Image loading and preprocessing.

Every image (color or grayscale) is loaded ONCE into an `ImageData` object:
  * `rgb`  -> uint8 HxWx3, used only for display / drawing.
  * `gray` -> float32 HxW in [0, 1], the single input of every algorithm.

So no other member ever has to load or convert images again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ITU-R BT.601 luminance weights (R, G, B)
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Color -> grayscale (from scratch): Y = 0.299 R + 0.587 G + 0.114 B."""
    return (rgb.astype(np.float32) @ LUMA_WEIGHTS) / 255.0


@dataclass
class ImageData:
    path: Path
    rgb: np.ndarray        # HxWx3 uint8 (display)
    gray: np.ndarray       # HxW float32 in [0, 1] (algorithms)
    is_color: bool         # was the original file a color image?
    scale: float = 1.0     # resize factor applied at load time (1.0 = original)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path, max_side: int | None = None) -> "ImageData":
        """Read any common image format (also works with non-ASCII paths on Windows)."""
        path = Path(path)
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot read image: {path}")
        return cls.from_array(raw, path, max_side)

    @classmethod
    def from_array(cls, raw: np.ndarray, path: Path, max_side: int | None = None) -> "ImageData":
        raw = cls._to_uint8(raw)

        # 1) Bring everything to RGB for display
        if raw.ndim == 2:
            rgb = np.repeat(raw[:, :, None], 3, axis=2)
            is_color = False
        else:
            if raw.shape[2] == 4:                        # drop alpha channel
                raw = raw[:, :, :3]
            rgb = raw[:, :, ::-1].copy()                 # OpenCV is BGR -> RGB
            # A 3-channel file whose channels are identical is really grayscale
            is_color = not (np.array_equal(rgb[..., 0], rgb[..., 1])
                            and np.array_equal(rgb[..., 1], rgb[..., 2]))

        # 2) Optional downscale of very large images (keeps the app responsive)
        scale = 1.0
        if max_side and max(rgb.shape[:2]) > max_side:
            scale = max_side / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 3) Grayscale float image = the algorithm input
        gray = rgb_to_gray(rgb) if is_color else rgb[:, :, 0].astype(np.float32) / 255.0
        return cls(path=path, rgb=np.ascontiguousarray(rgb), gray=np.ascontiguousarray(gray),
                   is_color=is_color, scale=scale)

    @staticmethod
    def _to_uint8(raw: np.ndarray) -> np.ndarray:
        if raw.dtype == np.uint8:
            return raw
        raw = raw.astype(np.float32)
        raw -= raw.min()
        return (255 * raw / max(float(raw.max()), 1e-6)).astype(np.uint8)

    # --------------------------------------------------------------- properties
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def height(self) -> int:
        return self.gray.shape[0]

    @property
    def width(self) -> int:
        return self.gray.shape[1]

    @property
    def kind(self) -> str:
        return "Color" if self.is_color else "Grayscale"
