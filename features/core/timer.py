"""Shared timing utility (used by every member — written once).

Usage:
    with Timer() as t:
        do_work()
    print(t.ms)
"""
from __future__ import annotations

import time


class Timer:
    """Context manager that measures wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
