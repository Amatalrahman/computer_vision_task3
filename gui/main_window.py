"""Main window: image list (left), one tab per step (center), status bar (bottom).

It owns the single `FeatureProject` and runs every heavy job on a background
thread through `run_task`, so the UI stays responsive.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from features import FeatureProject
from features.detectors import DETECTORS
from features.visualization import CYAN, GREEN, ORANGE

from .tabs import DetectorTab, MatchingTab, ReportTab, SiftTab
from .widgets import to_qimage

APP_NAME = "Feature Lab"
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm *.ppm *.webp)"

class MainWindow(QMainWindow):
    busy_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.project = FeatureProject(max_side=1600)
        self.pool = QThreadPool.globalInstance()
        self._task = None
        self.setWindowTitle(f"{APP_NAME} — Harris · λ− · DoG · SIFT · SSD / NCC")
        self.resize(1480, 900)
        self.setAcceptDrops(True)

        self._build_tabs()
        self._build_image_dock()
        self._build_toolbar()
        self._build_statusbar()
        self.refresh_all()

    # ================================================================== UI
    def _build_tabs(self) -> None:
        self.detector_tabs = {
            "Harris": DetectorTab(self, "Harris", "Harris Corner Detector", GREEN),
            "Lambda-": DetectorTab(self, "Lambda-", "λ− Feature Detector", CYAN),
            "DoG": DetectorTab(self, "DoG", "DoG Scale-space Detector", ORANGE),
        }
        self.sift_tab = SiftTab(self)
        self.match_tab = MatchingTab(self)
        self.sift_tab.source.currentTextChanged.connect(self.match_tab.source.setCurrentText)
        self.match_tab.source.currentTextChanged.connect(self.sift_tab.source.setCurrentText)
        self.report_tab = ReportTab(self)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.detector_tabs["Harris"], "1 · Harris")
        self.tabs.addTab(self.detector_tabs["Lambda-"], "2 · λ−")
        self.tabs.addTab(self.detector_tabs["DoG"], "3 · DoG")
        self.tabs.addTab(self.sift_tab, "4 · SIFT")
        self.tabs.addTab(self.match_tab, "5 · Matching")
        self.tabs.addTab(self.report_tab, "Report")
        self.tabs.currentChanged.connect(lambda _: self.current_tab().refresh())
        central = QWidget()
        central.setObjectName("central")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_image_dock(self) -> None:
        body = QWidget()
        body.setObjectName("dockBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(72, 54))
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.image_list.currentRowChanged.connect(lambda _: self.refresh_all())
        hint = QLabel("Drop image files here, or use the buttons below.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        row = QHBoxLayout()
        for text, slot in (("Add…", self.add_images_dialog), ("Remove", self.remove_selected),
                           ("Clear", self.clear_images)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addWidget(hint)
        lay.addWidget(self.image_list, 1)
        lay.addLayout(row)

        dock = QDockWidget("Images", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(body)
        dock.setMinimumWidth(250)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        def action(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        action("Add images", self.add_images_dialog, "Ctrl+O")
        action("Load samples", self.load_samples, tip="Load the demo image set in /samples")
        tb.addSeparator()
        self.run_action = action("▶  Run full pipeline", self.run_full_pipeline, "F5",
                                 "Harris + λ− + SIFT on every image, SSD + NCC on every pair")
        tb.addSeparator()
        action("Save view", self.save_view, "Ctrl+S", "Save the image currently shown")
        action("Export report", self.export_report, "Ctrl+E", "Save all timings as CSV")
        tb.addSeparator()
        action("About", self.about)
        self.busy_changed.connect(lambda busy: self.run_action.setEnabled(not busy))

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(180)
        self.progress.hide()
        self.count_label = QLabel()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.count_label)
        self.statusBar().showMessage("Ready")

    # ================================================================== state queries (used by tabs)
    def current_context(self):
        row = self.image_list.currentRow()
        ctxs = self.project.contexts
        return ctxs[row] if 0 <= row < len(ctxs) else None

    def require_image(self):
        ctx = self.current_context()
        if ctx is None:
            self.warn("Add an image first (toolbar ▸ Add images, or drag & drop).")
        return ctx

    def detector_params(self, name: str):
        return self.detector_tabs[name].form.values()

    def sift_params(self):
        return self.sift_tab.form.values()

    def current_tab(self):
        return self.tabs.currentWidget()

    # ================================================================== background jobs
    def run_task(self, message: str, fn, on_done=None) -> None:
        if self._task is not None:
            return
        from .worker import Task
        task = Task(fn)
        task.signals.progress.connect(lambda m: self.statusBar().showMessage(m + " …"))
        task.signals.done.connect(lambda result: self._finish(message, result, on_done))
        task.signals.failed.connect(self._failed)
        self._task = task
        self.progress.show()
        self.busy_changed.emit(True)
        self.statusBar().showMessage(message + " …")
        self.pool.start(task)

    def _end(self) -> None:
        self._task = None
        self.progress.hide()
        self.busy_changed.emit(False)

    def _finish(self, message, result, on_done) -> None:
        self._end()
        if on_done:
            on_done(result)
        self.statusBar().showMessage(f"Done: {message}", 6000)
        self.refresh_all()

    def _failed(self, error: str) -> None:
        self._end()
        self.statusBar().showMessage("Failed", 6000)
        QMessageBox.critical(self, APP_NAME, error)
        self.refresh_all()

    # ================================================================== images
    def add_images(self, paths) -> None:
        paths = [str(p) for p in paths]
        if not paths:
            return
        before = len(self.project.contexts)

        def done(errors):
            self._sync_image_list(select=before if len(self.project.contexts) > before else None)
            if errors:
                QMessageBox.warning(self, APP_NAME, "Some files could not be loaded:\n\n" + "\n".join(errors))
        self.run_task(f"Loading {len(paths)} image(s)", lambda _p: self.project.add_images(paths), done)

    def add_images_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", str(SAMPLES), IMAGE_FILTER)
        self.add_images(paths)

    def load_samples(self) -> None:
        files = sorted(SAMPLES.glob("*.png"))
        if not files:
            import runpy
            runpy.run_path(str(ROOT / "tools" / "make_samples.py"), run_name="__main__")
            files = sorted(SAMPLES.glob("*.png"))
        self.add_images(files)

    def remove_selected(self) -> None:
        row = self.image_list.currentRow()
        if self._task is None and 0 <= row < len(self.project.contexts):
            self.project.remove_image(row)
            self._sync_image_list(select=min(row, len(self.project.contexts) - 1))

    def clear_images(self) -> None:
        if self._task is None:
            self.project.clear()
            self._sync_image_list()

    def _sync_image_list(self, select: int | None = None) -> None:
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for ctx in self.project.contexts:
            img = ctx.image
            thumb = QPixmap.fromImage(to_qimage(img.rgb)).scaled(
                72, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            item = QListWidgetItem(QIcon(thumb), f"{img.name}\n{img.width}×{img.height} · {img.kind}")
            item.setToolTip(str(img.path))
            self.image_list.addItem(item)
        if self.project.contexts:
            self.image_list.setCurrentRow(select if select is not None and select >= 0 else 0)
        self.image_list.blockSignals(False)
        self.match_tab.refresh_images()
        self.refresh_all()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self.add_images([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])

    # ================================================================== actions
    def run_full_pipeline(self) -> None:
        if not self.project.contexts:
            self.warn("Add images first.")
            return
        det = {name: self.detector_params(name) for name in DETECTORS}
        sift, match = self.sift_params(), self.match_tab.params()
        self.run_task("Full pipeline", lambda progress: self.project.run_all(det, sift, match, progress),
                      lambda _r: self.tabs.setCurrentWidget(self.report_tab))

    def save_view(self) -> None:
        view = getattr(self.current_tab(), "view", None)
        if view is None or view.array is None:
            self.warn("Nothing to save in this tab.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save view", "result.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            ok, buf = cv2.imencode(Path(path).suffix or ".png", np.ascontiguousarray(view.array[:, :, ::-1]))
            if ok:
                buf.tofile(path)
                self.statusBar().showMessage(f"Saved {path}", 6000)

    def export_report(self) -> None:
        if not self.project.timing_rows():
            self.warn("No results yet — run a step or the full pipeline first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export report", "timings.csv", "CSV (*.csv)")
        if path:
            self.project.export_csv(path)
            self.statusBar().showMessage(f"Report saved to {path}", 6000)

    def about(self) -> None:
        QMessageBox.about(self, APP_NAME, (
            f"<b>{APP_NAME}</b><br>Feature detection, description and matching — implemented from scratch."
            "<br><br>Images larger than 1600 px are resized on load for speed."))

    def warn(self, text: str) -> None:
        QMessageBox.information(self, APP_NAME, text)

    def refresh_all(self) -> None:
        n = len(self.project.contexts)
        self.count_label.setText(f"{n} image{'s' if n != 1 else ''}")
        self.current_tab().refresh()
