"""One tab per project step. Every tab = sidebar (parameters + actions) + stats + image view.

`StepTab` holds the common layout; each subclass only adds its own controls
and decides what to draw.  Tabs never compute anything themselves — they ask
`FeatureProject` (features/pipeline.py) through the main window.
"""
from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from features import TimingRow
from features.core import Keypoints
from features.descriptors import SIFTParams
from features.detectors import DETECTORS
from features.matching import MATCHERS
from features.visualization import (GREEN, ORANGE, draw_descriptor, draw_keypoints,
                                    draw_matches, response_heatmap)

from .widgets import ImageView, ParamForm, Segmented, StatRow, primary_button, to_qimage


# =====================================================================================
class StepTab(QWidget):
    TITLE = ""
    OWNER = ""
    DESCRIPTION = ""
    STATS: list[str] = []

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._run_buttons: list[QPushButton] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- sidebar
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(330)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(16, 16, 16, 16)
        title = QLabel(self.TITLE)
        title.setObjectName("title")
        owner = QLabel(self.OWNER)
        owner.setObjectName("owner")
        desc = QLabel(self.DESCRIPTION)
        desc.setObjectName("muted")
        desc.setWordWrap(True)
        for w in (title, owner, desc):
            side_lay.addWidget(w)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("scrollBody")
        self.controls = QVBoxLayout(body)
        self.controls.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(body)
        side_lay.addWidget(scroll, 1)
        self.actions = QVBoxLayout()
        side_lay.addLayout(self.actions)
        root.addWidget(side)

        # ---- main area
        main = QVBoxLayout()
        main.setContentsMargins(16, 16, 16, 16)
        self.stats = StatRow(self.STATS)
        main.addWidget(self.stats)
        self.toolbar = QHBoxLayout()
        main.addLayout(self.toolbar)
        self.view = ImageView("Add images from the left panel to start")
        main.addWidget(self.view, 1)
        root.addLayout(main, 1)

        self.build()
        self.controls.addStretch(1)
        fit = QPushButton("Fit to window")
        fit.clicked.connect(self.view.fit)
        self.toolbar.addStretch(1)
        self.toolbar.addWidget(QLabel("Wheel = zoom · drag = pan"))
        self.toolbar.addWidget(fit)
        app.busy_changed.connect(lambda busy: [b.setEnabled(not busy) for b in self._run_buttons])

    # helpers for subclasses
    def add_group(self, title: str, widget: QWidget) -> None:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addWidget(widget)
        self.controls.addWidget(box)

    def add_action(self, text: str, slot, primary: bool = False) -> QPushButton:
        btn = primary_button(text) if primary else QPushButton(text)
        btn.clicked.connect(slot)
        self.actions.addWidget(btn)
        self._run_buttons.append(btn)
        return btn

    def build(self) -> None: ...
    def refresh(self) -> None: ...


# =====================================================================================
class DetectorTab(StepTab):
    """Shared by Harris (Member 1) and Lambda-minus (Member 2)."""
    STATS = ["Keypoints", "Gradients", "Structure tensor", "Response", "Threshold + NMS", "Total"]

    def __init__(self, app, detector: str, title: str, owner: str, description: str, color) -> None:
        self.detector, self.color = detector, color
        self.TITLE, self.OWNER, self.DESCRIPTION = title, owner, description
        super().__init__(app)

    def build(self) -> None:
        self.form = ParamForm(DETECTORS[self.detector].params_cls)
        self.add_group("Parameters", self.form)
        self.add_action("Detect on current image", self.run_current, primary=True)
        self.add_action("Detect on all images", self.run_all)
        reset = QPushButton("Reset defaults")
        reset.clicked.connect(self.form.reset)
        self.actions.addWidget(reset)
        self.mode = Segmented(["Keypoints", "Response map", "Original"])
        self.mode.changed.connect(lambda _: self.refresh())
        self.toolbar.addWidget(self.mode)

    def run_current(self) -> None:
        ctx = self.app.require_image()
        if ctx:
            params = self.form.values()
            self.app.run_task(f"{self.detector} on {ctx.name}",
                              lambda _p: self.app.project.detect(ctx, self.detector, params))

    def run_all(self) -> None:
        if not self.app.require_image():
            return
        params, project = self.form.values(), self.app.project

        def job(progress):
            for ctx in project.contexts:
                progress(f"{self.detector} on {ctx.name}")
                project.detect(ctx, self.detector, params)
        self.app.run_task(f"{self.detector} on all images", job)

    def refresh(self) -> None:
        ctx = self.app.current_context()
        if ctx is None:
            self.view.clear()
            self.stats.reset()
            return
        r = ctx.detections.get(self.detector)
        mode = self.mode.value()
        if r is None:
            self.stats.reset()
            self.view.show_array(ctx.image.rgb)
            return
        self.stats.set(**{"Keypoints": len(r.keypoints), "Gradients": r.gradients_ms,
                          "Structure tensor": r.tensor_ms, "Response": r.response_ms,
                          "Threshold + NMS": r.nms_ms, "Total": r.total_ms})
        if mode == "Response map":
            self.view.show_array(response_heatmap(r.response))
        elif mode == "Original":
            self.view.show_array(ctx.image.rgb)
        else:
            self.view.show_array(draw_keypoints(ctx.image.rgb, r.keypoints, self.color))


# =====================================================================================
class SiftTab(StepTab):
    TITLE = "SIFT Descriptors"
    OWNER = "Member 3"
    DESCRIPTION = ("128-D descriptor per keypoint: dominant orientation → rotated 16×16 patch → "
                   "4×4 cells × 8-bin histograms → normalise, clip 0.2, re-normalise. "
                   "Reuses the gradients already computed for detection.")
    STATS = ["Descriptors", "Dimension", "Orientation", "Descriptor", "Total"]

    def build(self) -> None:
        self.source = QComboBox()
        self.source.addItems(list(DETECTORS))
        self.source.currentTextChanged.connect(lambda _: self.refresh())
        hint = QLabel("Uses the parameters set in that detector's tab.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.source)
        lay.addWidget(hint)
        self.add_group("Keypoints from", box)

        self.form = ParamForm(SIFTParams)
        self.add_group("SIFT parameters", self.form)

        inspect = QWidget()
        il = QFormLayout(inspect)
        il.setContentsMargins(0, 0, 0, 0)
        self.kp_index = QSpinBox()
        self.kp_index.setRange(0, 0)
        self.kp_index.valueChanged.connect(lambda _: self.refresh())
        self.kp_info = QLabel("—")
        self.kp_info.setObjectName("muted")
        self.desc_image = QLabel()
        self.desc_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        il.addRow("Keypoint #", self.kp_index)
        il.addRow(self.kp_info)
        il.addRow(self.desc_image)
        self.add_group("Inspect one descriptor", inspect)

        self.add_action("Describe current image", self.run_current, primary=True)
        self.add_action("Describe all images", self.run_all)
        self.mode = Segmented(["Keypoints + orientation", "Original"])
        self.mode.changed.connect(lambda _: self.refresh())
        self.toolbar.addWidget(self.mode)

    def _job_args(self):
        det = self.source.currentText()
        return det, self.form.values(), self.app.detector_params(det)

    def run_current(self) -> None:
        ctx = self.app.require_image()
        if ctx:
            det, sift, dp = self._job_args()
            self.app.run_task(f"SIFT ({det}) on {ctx.name}",
                              lambda _p: self.app.project.describe(ctx, det, sift, dp))

    def run_all(self) -> None:
        if not self.app.require_image():
            return
        det, sift, dp = self._job_args()
        project = self.app.project

        def job(progress):
            for ctx in project.contexts:
                progress(f"SIFT ({det}) on {ctx.name}")
                project.describe(ctx, det, sift, dp)
        self.app.run_task("SIFT on all images", job)

    def refresh(self) -> None:
        ctx = self.app.current_context()
        self.desc_image.clear()
        self.kp_info.setText("—")
        if ctx is None:
            self.view.clear()
            self.stats.reset()
            return
        d = ctx.descriptors.get(self.source.currentText())
        if d is None or len(d.descriptors) == 0:
            self.stats.reset()
            self.view.show_array(ctx.image.rgb)
            return
        self.stats.set(Descriptors=len(d.descriptors), Dimension=d.descriptors.shape[1],
                       Orientation=d.orientation_ms, Descriptor=d.descriptor_ms, Total=d.total_ms)
        self.kp_index.blockSignals(True)
        self.kp_index.setMaximum(len(d.descriptors) - 1)
        self.kp_index.blockSignals(False)
        i = self.kp_index.value()
        x, y = d.keypoints.xy[i]
        theta = float(d.keypoints.orientations[i]) * 180 / 3.14159265
        self.kp_info.setText(f"(x={x:.0f}, y={y:.0f})   θ = {theta:.1f}°")
        self.desc_image.setPixmap(QPixmap.fromImage(to_qimage(draw_descriptor(d.descriptors[i]))))

        if self.mode.value() == "Original":
            self.view.show_array(ctx.image.rgb)
            return
        img = draw_keypoints(ctx.image.rgb, d.keypoints, GREEN, show_orientation=True)
        selected = Keypoints(d.keypoints.xy[i:i + 1], d.keypoints.scores[i:i + 1])
        img = draw_keypoints(img, selected, ORANGE)          # highlight the inspected point
        self.view.show_array(img)


# =====================================================================================
class MatchingTab(StepTab):
    TITLE = "Feature Matching"
    OWNER = "Member 4"
    DESCRIPTION = ("Compares every SIFT descriptor of image A with every one of image B in a single "
                   "matrix product. SSD = lowest distance + Lowe ratio test; NCC = highest correlation "
                   "+ minimum threshold. Optional cross-check keeps mutual best matches only.")
    STATS = ["Descriptors A", "Descriptors B", "SSD matches", "SSD time", "NCC matches", "NCC time"]

    def build(self) -> None:
        pick = QWidget()
        pl = QFormLayout(pick)
        pl.setContentsMargins(0, 0, 0, 0)
        self.combo_a, self.combo_b, self.source = QComboBox(), QComboBox(), QComboBox()
        self.source.addItems(list(DETECTORS))
        for c in (self.combo_a, self.combo_b, self.source):
            c.currentIndexChanged.connect(lambda _: self.refresh())
        pl.addRow("Image A", self.combo_a)
        pl.addRow("Image B", self.combo_b)
        pl.addRow("Keypoints", self.source)
        self.add_group("Image pair", pick)

        self.forms = {name: ParamForm(cls.params_cls) for name, cls in MATCHERS.items()}
        for name, form in self.forms.items():
            self.add_group(name, form)

        self.max_lines = QSpinBox()
        self.max_lines.setRange(5, 2000)
        self.max_lines.setValue(120)
        self.max_lines.setToolTip("Draw only the N most confident matches")
        self.max_lines.valueChanged.connect(lambda _: self.refresh())
        disp = QWidget()
        dl = QFormLayout(disp)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addRow("Lines to draw", self.max_lines)
        self.add_group("Display", disp)

        self.add_action("Match A ↔ B  (SSD + NCC)", self.run_pair, primary=True)
        self.add_action("Match all image pairs", self.run_all_pairs)
        self.mode = Segmented(list(MATCHERS))
        self.mode.changed.connect(lambda _: self.refresh())
        self.toolbar.addWidget(self.mode)

    def params(self) -> dict:
        return {name: form.values() for name, form in self.forms.items()}

    def refresh_images(self) -> None:
        names = [c.name for c in self.app.project.contexts]
        for combo, default in ((self.combo_a, 0), (self.combo_b, 1)):
            keep = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if names:
                combo.setCurrentIndex(keep if 0 <= keep < len(names) else min(default, len(names) - 1))
            combo.blockSignals(False)

    def _pair(self):
        ctxs = self.app.project.contexts
        ia, ib = self.combo_a.currentIndex(), self.combo_b.currentIndex()
        if not (0 <= ia < len(ctxs) and 0 <= ib < len(ctxs)):
            return None, None
        return ctxs[ia], ctxs[ib]

    def _match_job(self, pairs):
        det = self.source.currentText()
        dp, sift, mp, project = self.app.detector_params(det), self.app.sift_params(), self.params(), self.app.project

        def job(progress):
            for a, b in pairs:
                for m in MATCHERS:
                    progress(f"{m} ({det}): {a.name} ↔ {b.name}")
                    project.match(a, b, det, m, mp[m], sift, dp)
        return job

    def run_pair(self) -> None:
        a, b = self._pair()
        if a is None:
            self.app.warn("Add at least two images first.")
        elif a is b:
            self.app.warn("Choose two different images for A and B.")
        else:
            self.app.run_task(f"Matching {a.name} ↔ {b.name}", self._match_job([(a, b)]))

    def run_all_pairs(self) -> None:
        ctxs = self.app.project.contexts
        if len(ctxs) < 2:
            self.app.warn("Add at least two images first.")
            return
        pairs = [(ctxs[i], ctxs[j]) for i in range(len(ctxs)) for j in range(i + 1, len(ctxs))]
        self.app.run_task("Matching all pairs", self._match_job(pairs))

    def refresh(self) -> None:
        a, b = self._pair()
        if a is None:
            self.view.clear("Add at least two images to match")
            self.stats.reset()
            return
        det = self.source.currentText()
        results = {m: self.app.project.matches.get((a, b, det, m)) for m in MATCHERS}
        self.stats.reset()
        for m, r in results.items():
            if r is not None:
                self.stats.set(**{f"{m} matches": len(r), f"{m} time": r.elapsed_ms,
                                  "Descriptors A": r.n_query, "Descriptors B": r.n_train})
        shown = results[self.mode.value()]
        if shown is None:
            self.view.clear("Press “Match A ↔ B” to compute matches")
            return
        kps_a, kps_b = a.descriptors[det].keypoints, b.descriptors[det].keypoints
        self.view.show_array(draw_matches(a.image.rgb, kps_a, b.image.rgb, kps_b, shown,
                                          self.max_lines.value()))


# =====================================================================================
class ReportTab(QWidget):
    """All computation times in one table (+ averages) — ready for the report."""

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)

        head = QHBoxLayout()
        title = QLabel("Computation-time report")
        title.setObjectName("title")
        head.addWidget(title)
        head.addStretch(1)
        self.run_btn = primary_button("Run full pipeline on all images")
        self.run_btn.clicked.connect(app.run_full_pipeline)
        export = QPushButton("Export CSV…")
        export.clicked.connect(app.export_report)
        head.addWidget(self.run_btn)
        head.addWidget(export)
        lay.addLayout(head)
        note = QLabel("Detection time = shared gradients + shared structure tensor + detector response + "
                      "threshold/NMS. Shared steps are computed once and reused by Harris, λ- and SIFT.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.stats = StatRow(["Images", "Detections", "Descriptor sets", "Match runs"])
        lay.addWidget(self.stats)

        lay.addWidget(QLabel("Average time per step"))
        self.summary = self._table(["Step", "Method", "Runs", "Avg count", "Avg time (ms)"])
        lay.addWidget(self.summary, 1)
        lay.addWidget(QLabel("All runs"))
        self.table = self._table(list(TimingRow.HEADERS))
        lay.addWidget(self.table, 2)
        app.busy_changed.connect(lambda busy: self.run_btn.setEnabled(not busy))

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    @staticmethod
    def _fill(table: QTableWidget, rows: list[tuple]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if isinstance(value, (int, float)) or c == len(row) - 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)

    def refresh(self) -> None:
        project = self.app.project
        rows = project.timing_rows()
        self._fill(self.table, [r.as_tuple() for r in rows])

        groups = defaultdict(list)
        for r in rows:
            groups[(r.step, r.method)].append(r)
        summary = [(step, method, len(g), f"{sum(x.count for x in g) / len(g):.0f}",
                    f"{sum(x.total_ms for x in g) / len(g):.2f}")
                   for (step, method), g in groups.items()]
        self._fill(self.summary, summary)
        self.stats.set(**{"Images": len(project.contexts),
                          "Detections": sum(len(c.detections) for c in project.contexts),
                          "Descriptor sets": sum(len(c.descriptors) for c in project.contexts),
                          "Match runs": len(project.matches)})

    # the main window asks every tab for its view when saving
    view = None
