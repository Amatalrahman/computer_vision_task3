"""Runs heavy work on a background thread so the window never freezes."""
from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class Task(QRunnable):
    """`fn` receives one argument: a `progress(message)` callback."""

    def __init__(self, fn: Callable[[Callable[[str], None]], object]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self) -> None:
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:  # noqa: BLE001 - shown to the user in a dialog
            traceback.print_exc()
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.done.emit(result)
