"""Reusable GUI widgets (written once, used by every tab)."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDoubleSpinBox, QFormLayout, QFrame,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------- image view
class ImageView(QGraphicsView):
    """Zoom with the mouse wheel, pan by dragging, double-click to fit."""

    def __init__(self, placeholder: str = "No image") -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.placeholder = placeholder
        self.array: np.ndarray | None = None     # what is shown (for "Save view")
        self._auto_fit = True
        self.clear()

    def clear(self, text: str | None = None) -> None:
        self._scene.clear()
        self.array = None
        item = self._scene.addText(text or self.placeholder)
        item.setDefaultTextColor(Qt.GlobalColor.gray)
        self._scene.setSceneRect(item.boundingRect())
        self.resetTransform()

    def show_array(self, rgb: np.ndarray) -> None:
        first = self.array is None or self.array.shape != rgb.shape
        self.array = rgb
        self._scene.clear()
        self._scene.addPixmap(QPixmap.fromImage(to_qimage(rgb)))
        self._scene.setSceneRect(QRectF(0, 0, rgb.shape[1], rgb.shape[0]))
        if first or self._auto_fit:
            self.fit()

    def fit(self) -> None:
        self._auto_fit = True
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self.array is None:
            return
        self._auto_fit = False
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit and self.array is not None:
            self.fit()


# ---------------------------------------------------------------- stat card
class StatCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        lay.addWidget(t)
        lay.addWidget(self.value)

    def set(self, value) -> None:
        if isinstance(value, float):
            value = f"{value:.2f} ms"
        self.value.setText(str(value))


class StatRow(QWidget):
    """A horizontal row of StatCards addressed by title."""

    def __init__(self, titles: list[str]) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cards = {t: StatCard(t) for t in titles}
        for c in self.cards.values():
            lay.addWidget(c)

    def set(self, **values) -> None:
        for k, v in values.items():
            self.cards[k].set(v)

    def reset(self) -> None:
        for c in self.cards.values():
            c.set("—")


# ---------------------------------------------------------------- segmented control
class Segmented(QWidget):
    changed = Signal(str)

    def __init__(self, options: list[str]) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        for i, name in enumerate(options):
            b = QPushButton(name)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.group.addButton(b)
            lay.addWidget(b)
        self.group.buttonClicked.connect(lambda b: self.changed.emit(b.text()))

    def value(self) -> str:
        return self.group.checkedButton().text()


# ---------------------------------------------------------------- auto parameter form
class ParamForm(QWidget):
    """Builds input widgets automatically from a frozen dataclass of parameters.

    Each field's metadata (label / min / max / step / tip) comes from
    `features.core.param(...)`, so algorithm owners never write GUI code.
    """

    def __init__(self, params_cls: type) -> None:
        super().__init__()
        self.params_cls = params_cls
        self.widgets: dict[str, QWidget] = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        for f in fields(params_cls):
            meta, default = f.metadata, f.default
            if isinstance(default, bool):
                w = QCheckBox()
            elif isinstance(default, int):
                w = QSpinBox()
                w.setRange(int(meta.get("min") or 0), int(meta.get("max") or 10 ** 6))
                w.setSingleStep(int(meta.get("step") or 1))
            else:
                w = QDoubleSpinBox()
                step = float(meta.get("step") or 0.1)
                w.setDecimals(max(2, len(f"{step:.6f}".rstrip("0").split(".")[1])))
                w.setRange(float(meta.get("min") or 0.0), float(meta.get("max") or 1e6))
                w.setSingleStep(step)
            w.setToolTip(meta.get("tip", ""))
            label = QLabel(meta.get("label", f.name))
            label.setToolTip(meta.get("tip", ""))
            form.addRow(label, w)
            self.widgets[f.name] = w
        self.reset()

    def reset(self) -> None:
        for f in fields(self.params_cls):
            self._set(self.widgets[f.name], f.default)

    @staticmethod
    def _set(w, value) -> None:
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)

    def values(self):
        kwargs = {}
        for name, w in self.widgets.items():
            kwargs[name] = w.isChecked() if isinstance(w, QCheckBox) else w.value()
        return self.params_cls(**kwargs)


def primary_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("primary")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b
