"""Feature extraction & matching library (no GUI code here).

Package map (who owns what):
    core/         Member 1 — image loading, Sobel gradients, structure tensor, NMS, timer
    detectors/    Member 1 — base detector + Harris   |   Member 2 — Lambda-minus
    descriptors/  Member 3 — SIFT descriptor
    matching/     Member 4 — SSD and NCC matchers
    pipeline.py   Member 1 — glues the steps together + timing report
    visualization.py        shared drawing helpers
"""
from .pipeline import FeatureProject, TimingRow

__all__ = ["FeatureProject", "TimingRow"]
