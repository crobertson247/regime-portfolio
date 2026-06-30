"""
Regime detection module (Phase 3).

Detects a market regime (calm, volatile, crisis) for each trading day from the
Phase 2 feature matrix, under strictly causal, point-in-time inference.

Three detector families share the RegimeDetector interface:
- HMMDetector: Gaussian hidden Markov model (probabilistic baseline).
- JumpModelDetector: temporal clustering with a jump penalty.
- ChangePointDetector: assumption-light change-point segmentation (ruptures).
"""

from regime.detection.base import (
    CALM,
    CRISIS,
    REGIME_NAMES,
    VOLATILE,
    RegimeDetector,
    regime_name,
)
from regime.detection.changepoint import ChangePointDetector
from regime.detection.config import DetectionConfig, load_detection_config
from regime.detection.features import prepare_detection_features
from regime.detection.hmm import HMMDetector
from regime.detection.jump import JumpModelDetector
from regime.detection.walkforward import label_walk_forward

# Detector registry, keyed by the --method name used in scripts.
DETECTORS = {
    "hmm": HMMDetector,
    "jump": JumpModelDetector,
    "changepoint": ChangePointDetector,
}

__all__ = [
    "RegimeDetector",
    "HMMDetector",
    "JumpModelDetector",
    "ChangePointDetector",
    "DETECTORS",
    "DetectionConfig",
    "load_detection_config",
    "prepare_detection_features",
    "label_walk_forward",
    "regime_name",
    "REGIME_NAMES",
    "CALM",
    "VOLATILE",
    "CRISIS",
]
