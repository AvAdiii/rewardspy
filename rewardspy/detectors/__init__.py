"""The reward hacking detection engine.

Six independent detectors, each reporting OK / WARNING / ALERT, coordinated by
``DetectionEngine``. See each module for the heuristic and its math.
"""

from .base import BaseDetector, DetectionResult, Status
from .ceiling import CeilingRateDetector
from .components import ComponentDominanceDetector
from .divergence import ProxyEvalDivergenceDetector
from .engine import DetectionEngine
from .length import LengthDriftDetector
from .slope import RewardSlopeChangeDetector
from .variance import VarianceCollapseDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "Status",
    "DetectionEngine",
    "VarianceCollapseDetector",
    "RewardSlopeChangeDetector",
    "ComponentDominanceDetector",
    "CeilingRateDetector",
    "LengthDriftDetector",
    "ProxyEvalDivergenceDetector",
]
