"""Tiny helper to declare an algorithm parameter ONCE.

Each parameter carries its own label / range / tooltip, so the GUI builds the
input forms automatically from the dataclass — no member writes GUI code.

    k: float = param(0.04, "Harris k", 0.01, 0.20, 0.005, "Sensitivity constant")
"""
from __future__ import annotations

from dataclasses import field


def param(default, label: str, minimum=None, maximum=None, step=None, tip: str = ""):
    return field(default=default, metadata={
        "label": label, "min": minimum, "max": maximum, "step": step, "tip": tip,
    })
