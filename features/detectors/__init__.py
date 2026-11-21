from .base import DetectionResult, DetectorParams, FeatureDetector
from .harris import HarrisDetector, HarrisParams
from .lambda_minus import LambdaMinusDetector, LambdaParams

# Registry: the GUI and pipeline discover detectors from here.
DETECTORS: dict[str, type[FeatureDetector]] = {
    HarrisDetector.name: HarrisDetector,
    LambdaMinusDetector.name: LambdaMinusDetector,
}

__all__ = ["DETECTORS", "DetectionResult", "DetectorParams", "FeatureDetector",
           "HarrisDetector", "HarrisParams", "LambdaMinusDetector", "LambdaParams"]
