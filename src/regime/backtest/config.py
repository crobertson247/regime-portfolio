"""
Configuration for the backtest.

Defaults match the evaluation design: a proportional transaction cost, cash (BIL)
as the risk-free rate in the Sharpe ratio, the three stress windows used
throughout, and a cost grid for the sensitivity check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    """Parameters for the walk-forward backtest and evaluation."""

    assets: list[str] = Field(default_factory=lambda: ["SPY", "TLT", "LQD", "GLD", "DBC", "BIL"])
    rf_asset: str = "BIL"  # risk-free proxy for the Sharpe ratio

    cost_bps: float = 10.0  # proportional cost per unit traded, basis points
    cost_grid_bps: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0, 20.0])

    cscv_blocks: int = 16  # CSCV blocks for the PBO estimate

    stress_windows: dict[str, tuple[str, str]] = Field(
        default_factory=lambda: {
            "2008 GFC": ("2008-09-01", "2009-03-31"),
            "2020 COVID": ("2020-02-15", "2020-04-30"),
            "2022 tightening": ("2022-01-01", "2022-10-31"),
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BacktestConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Backtest config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw.get("backtest", raw))


def load_backtest_config(path: Optional[str | Path] = None) -> BacktestConfig:
    """Load the backtest config from YAML, or return defaults if path is None."""
    return BacktestConfig() if path is None else BacktestConfig.from_yaml(path)
