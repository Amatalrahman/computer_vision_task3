from .base import DetectionResult, DetectorParams, FeatureDetector
from .dog import DoGDetector, DoGParams
from .harris import HarrisDetector, HarrisParams
from .lambda_minus import LambdaMinusDetector, LambdaParams

# Registry: the GUI and pipeline discover detectors from here.
# DoGDetector doesn't inherit FeatureDetector (see dog.py), hence the plain `type`.
DETECTORS: dict[str, type] = {
    HarrisDetector.name: HarrisDetector,
    LambdaMinusDetector.name: LambdaMinusDetector,
    DoGDetector.name: DoGDetector,
}

__all__ = ["DETECTORS", "DetectionResult", "DetectorParams", "FeatureDetector",
           "HarrisDetector", "HarrisParams", "LambdaMinusDetector", "LambdaParams",
           "DoGDetector", "DoGParams"]