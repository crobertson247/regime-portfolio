"""
Configuration for the regime detection module.

Defaults are set for the US cross-asset ETF basket and can be overridden from a
YAML file (config/detection.yaml). The detection feature set is deliberately
small: a market return, a market volatility, the average pairwise correlation,
and the VIX. The severity features are the subset used to order the fitted
states from calm to crisis. Each detector family has its own parameters; the
common feature set, standardisation and walk-forward settings are shared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import yaml
from pydantic import BaseModel, Field


class DetectionConfig(BaseModel):
    """Parameters for regime detection and walk-forward labelling."""

    method: str = "hmm"  # default detector: hmm, jump or changepoint
    n_states: int = Field(default=3, ge=2, le=5)
    features: list[str] = Field(
        default_factory=lambda: ["SPY_ret", "SPY_vol63d", "avg_corr63d", "VIXCLS"]
    )
    severity_features: list[str] = Field(
        default_factory=lambda: ["SPY_vol63d", "avg_corr63d", "VIXCLS"]
    )
    standardize_min_periods: int = Field(default=60, ge=10)
    random_state: int = 42

    # HMM
    covariance_type: str = "diag"
    n_iter: int = Field(default=100, ge=10)
    tol: float = 1e-4
    n_restarts: int = Field(default=8, ge=1)
    min_covar: float = 1e-3

    # Jump model
    jump_penalty: float = 50.0
    jump_n_init: int = Field(default=10, ge=1)

    # Change-point
    cp_model: str = "l2"
    cp_min_size: int = Field(default=10, ge=2)
    cp_pen_scale: float = 2.0
    cp_window: int = Field(default=126, ge=20)

    # Walk-forward labelling
    min_train: int = Field(default=252, ge=30)
    refit_every: int = Field(default=63, ge=1)

    def severity_indices(self) -> list[int]:
        """Positions of the severity features within the detection feature list."""
        return [self.features.index(c) for c in self.severity_features if c in self.features]

    def build_factory(self, method: Optional[str] = None) -> Callable[[], object]:
        """Return a callable that builds a fresh detector of the given method."""
        from regime.detection.changepoint import ChangePointDetector
        from regime.detection.hmm import HMMDetector
        from regime.detection.jump import JumpModelDetector

        method = (method or self.method).lower()
        sev = self.severity_indices()
        if method == "hmm":
            return lambda: HMMDetector(
                n_states=self.n_states, severity_indices=sev,
                covariance_type=self.covariance_type, n_iter=self.n_iter, tol=self.tol,
                n_restarts=self.n_restarts, random_state=self.random_state,
                min_covar=self.min_covar,
            )
        if method == "jump":
            return lambda: JumpModelDetector(
                n_states=self.n_states, severity_indices=sev,
                jump_penalty=self.jump_penalty, n_init=self.jump_n_init,
                random_state=self.random_state,
            )
        if method == "changepoint":
            return lambda: ChangePointDetector(
                n_states=self.n_states, severity_indices=sev, model=self.cp_model,
                min_size=self.cp_min_size, pen_scale=self.cp_pen_scale,
                window=self.cp_window, random_state=self.random_state,
            )
        raise ValueError(f"unknown detection method: {method}")

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
