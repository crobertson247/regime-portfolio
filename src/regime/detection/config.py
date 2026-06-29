"""
Configuration for the regime detection module.

Defaults are set for the US cross-asset ETF basket and can be overridden from a
YAML file (config/detection.yaml). The detection feature set is deliberately
small: a market return, a market volatility, the average pairwise correlation,
and the VIX. The severity features are the subset used to order the fitted
states from calm to crisis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class DetectionConfig(BaseModel):
    """Parameters for HMM regime detection and walk-forward labelling."""

    n_states: int = Field(default=3, ge=2, le=5)
    features: list[str] = Field(
        default_factory=lambda: ["SPY_ret", "SPY_vol63d", "avg_corr63d", "VIXCLS"]
    )
    severity_features: list[str] = Field(
        default_factory=lambda: ["SPY_vol63d", "avg_corr63d", "VIXCLS"]
    )
    standardize_min_periods: int = Field(default=60, ge=10)

    # HMM fitting
    covariance_type: str = "diag"
    n_iter: int = Field(default=100, ge=10)
    tol: float = 1e-4
    n_restarts: int = Field(default=8, ge=1)
    random_state: int = 42
    min_covar: float = 1e-3

    # Walk-forward labelling
    min_train: int = Field(default=252, ge=30)
    refit_every: int = Field(default=63, ge=1)

    def severity_indices(self) -> list[int]:
        """Positions of the severity features within the detection feature list."""
        return [self.features.index(c) for c in self.severity_features if c in self.features]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectionConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Detection config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw.get("detection", raw))


def load_detection_config(path: Optional[str | Path] = None) -> DetectionConfig:
    """Load detection config from YAML, or return defaults if no path is given."""
    if path is None:
        default = Path(__file__).parent.parent.parent.parent / "config" / "detection.yaml"
        if default.exists():
            return DetectionConfig.from_yaml(default)
        return DetectionConfig()
    return DetectionConfig.from_yaml(path)
