"""FeatureProject — the single entry point that glues the four members together.

    load images -> detect (Harris | λ-) -> describe (SIFT) -> match (SSD | NCC) -> report

* GUI-independent: can be used from the desktop app, a script or a notebook.
* Lazy & cached: a step runs its prerequisites only if they are missing or were
  produced with different parameters, so nothing is ever computed twice.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .core import ImageContext, ImageData
from .descriptors import DescriptorResult, SIFTDescriptor, SIFTParams
from .detectors import DETECTORS, DetectionResult, DetectorParams
from .matching import MATCHERS, MatchParams, MatchResult

MatchKey = tuple[ImageContext, ImageContext, str, str]   # (A, B, detector, matcher)
Progress = Callable[[str], None]


@dataclass
class TimingRow:
    subject: str          # image name or "A ↔ B"
    step: str             # Detection / Description / Matching
    method: str           # Harris, Lambda-, SIFT(Harris), SSD(Harris) ...
    count: int            # keypoints / descriptors / matches
    breakdown: str        # how the time is split
    total_ms: float

    HEADERS = ("Image / Pair", "Step", "Method", "Count", "Breakdown (ms)", "Total (ms)")

    def as_tuple(self) -> tuple:
        return (self.subject, self.step, self.method, self.count, self.breakdown,
                f"{self.total_ms:.2f}")


class FeatureProject:
    def __init__(self, max_side: int | None = 1600) -> None:
        self.max_side = max_side
        self.contexts: list[ImageContext] = []
        self.matches: dict[MatchKey, MatchResult] = {}

    # ============================================================ images
    def add_images(self, paths) -> list[str]:
        """Load images; returns a list of error messages (empty = all good)."""
        errors = []
        known = {c.image.path.resolve() for c in self.contexts}
        for p in map(Path, paths):
            if p.resolve() in known:
                continue
            try:
                self.contexts.append(ImageContext(ImageData.load(p, self.max_side)))
            except Exception as exc:  # noqa: BLE001 - report every bad file to the user
                errors.append(f"{p.name}: {exc}")
        return errors

    def remove_image(self, index: int) -> None:
        ctx = self.contexts.pop(index)
        self.matches = {k: v for k, v in self.matches.items() if ctx not in k[:2]}

    def clear(self) -> None:
        self.contexts.clear()
        self.matches.clear()

    # ============================================================ step 1: detection
    def detect(self, ctx: ImageContext, detector: str,
               params: DetectorParams | None = None) -> DetectionResult:
        result = DETECTORS[detector](params).detect(ctx)
        ctx.detections[detector] = result
        self._invalidate(ctx, detector)            # old descriptors / matches are stale
        return result

    def ensure_detection(self, ctx, detector, params=None) -> DetectionResult:
        params = params or DETECTORS[detector].params_cls()
        current = ctx.detections.get(detector)
        if current is not None and current.params == params:
            return current
        return self.detect(ctx, detector, params)

    # ============================================================ step 2: description
    def describe(self, ctx: ImageContext, detector: str, sift: SIFTParams | None = None,
                 det_params: DetectorParams | None = None) -> DescriptorResult:
        detection = self.ensure_detection(ctx, detector, det_params)
        result = SIFTDescriptor(sift).compute(ctx, detection)
        ctx.descriptors[detector] = result
        self.matches = {k: v for k, v in self.matches.items()
                        if not (ctx in k[:2] and k[2] == detector)}
        return result

    def ensure_descriptors(self, ctx, detector, sift=None, det_params=None) -> DescriptorResult:
        sift = sift or SIFTParams()
        before = ctx.detections.get(detector)
        detection = self.ensure_detection(ctx, detector, det_params)
        current = ctx.descriptors.get(detector)
        if current is not None and current.params == sift and detection is before:
            return current
        return self.describe(ctx, detector, sift, det_params)

    # ============================================================ step 3: matching
    def match(self, a: ImageContext, b: ImageContext, detector: str, matcher: str,
              params: MatchParams | None = None, sift: SIFTParams | None = None,
              det_params: DetectorParams | None = None) -> MatchResult:
        da = self.ensure_descriptors(a, detector, sift, det_params)
        db = self.ensure_descriptors(b, detector, sift, det_params)
        result = MATCHERS[matcher](params).match(da.descriptors, db.descriptors)
        self.matches[(a, b, detector, matcher)] = result
        return result

    # ============================================================ everything
    def run_all(self, det_params: dict[str, DetectorParams], sift: SIFTParams,
                match_params: dict[str, MatchParams], progress: Progress = lambda _: None) -> None:
        for ctx in self.contexts:
            for det in DETECTORS:
                progress(f"{det} on {ctx.name}")
                self.detect(ctx, det, det_params.get(det))
                progress(f"SIFT ({det}) on {ctx.name}")
                self.describe(ctx, det, sift, det_params.get(det))
        for a, b in itertools.combinations(self.contexts, 2):
            for det in DETECTORS:
                for m in MATCHERS:
                    progress(f"{m} ({det}): {a.name} ↔ {b.name}")
                    self.match(a, b, det, m, match_params.get(m), sift, det_params.get(det))

    # ============================================================ report
    def timing_rows(self) -> list[TimingRow]:
        rows: list[TimingRow] = []
        for ctx in self.contexts:
            for det, r in ctx.detections.items():
                rows.append(TimingRow(
                    ctx.name, "Detection", det, len(r.keypoints),
                    f"grad {r.gradients_ms:.2f} + tensor {r.tensor_ms:.2f} + "
                    f"response {r.response_ms:.2f} + NMS {r.nms_ms:.2f}", r.total_ms))
            for det, d in ctx.descriptors.items():
                rows.append(TimingRow(
                    ctx.name, "Description", f"SIFT ({det})", len(d.descriptors),
                    f"orientation {d.orientation_ms:.2f} + descriptor {d.descriptor_ms:.2f}",
                    d.total_ms))
        for (a, b, det, m), r in self.matches.items():
            rows.append(TimingRow(
                f"{a.name} ↔ {b.name}", "Matching", f"{m} ({det})", len(r),
                f"{r.n_query} × {r.n_train} descriptors", r.elapsed_ms))
        return rows

    def export_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(TimingRow.HEADERS)
            for row in self.timing_rows():
                w.writerow(row.as_tuple())

    # ============================================================ internal
    def _invalidate(self, ctx: ImageContext, detector: str) -> None:
        ctx.descriptors.pop(detector, None)
        self.matches = {k: v for k, v in self.matches.items()
                        if not (ctx in k[:2] and k[2] == detector)}
