"""
Regime detection module (Phase 3).

Detects a market regime (calm, volatile, crisis) for each trading day from the
Phase 2 feature matrix, under strictly causal, point-in-time inference.

This phase implements the hidden Markov model detector and the causal
walk-forward labelling harness. The statistical jump / clustering and
change-point detectors are planned next and will share the RegimeDetector
interface.
"""

from regime.detection.base import (
    CALM,
    CRISIS,
    REGIME_NAMES,
    VOLATILE,
    RegimeDetector,
    regime_name,
)
from regime.detection.config import DetectionConfig, load_detection_config
from regime.detection.features import prepare_detection_features
from regime.detection.hmm import HMMDetector
from regime.detection.walkforward import label_walk_forward

__all__ = [
    "RegimeDetector",
    "HMMDetector",
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
