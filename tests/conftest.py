"""
Pytest configuration and shared fixtures for regime-portfolio tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    Generate sample price data for testing.

    Returns a DataFrame with 500 trading days of synthetic prices
    for 6 assets, mimicking the ETF universe.
    """
    np.random.seed(42)

    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Generate correlated random walks
    assets = ["SPY", "TLT", "LQD", "GLD", "DBC", "BIL"]

    # Base returns with different volatilities
    vols = [0.15, 0.12, 0.08, 0.14, 0.18, 0.02]

    prices = {}
    for asset, vol in zip(assets, vols):
        daily_vol = vol / np.sqrt(252)
        returns = np.random.normal(0.0003, daily_vol, n_days)
        price = 100 * np.exp(np.cumsum(returns))
        prices[asset] = price

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "Date"

    return df


@pytest.fixture
def sample_macro() -> pd.DataFrame:
    """
    Generate sample macro data for testing.

    Returns a DataFrame with macro series aligned to trading days.
    """
    np.random.seed(42)

    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    macro = pd.DataFrame({
        "VIXCLS": np.random.uniform(12, 40, n_days),
        "T10Y2Y": np.random.uniform(-0.5, 2.5, n_days),
        "BAA10Y": np.random.uniform(1.5, 4.0, n_days),
    }, index=dates)

    macro.index.name = "Date"

    return macro


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sample return data from prices.
    """
    log_returns = np.log(sample_prices).diff()
    log_returns.columns = [f"{col}_ret" for col in log_returns.columns]
    return log_returns


@pytest.fixture
def trading_days() -> pd.DatetimeIndex:
    """
    Generate a sample trading calendar.
    """
    return pd.date_range("2020-01-01", periods=500, freq="B")
