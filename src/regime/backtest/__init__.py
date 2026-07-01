"""
Walk-forward backtesting and evaluation module (Phase 5).

Applies the Phase 4 weight series to realised returns under a causal timing
convention with transaction costs, then scores the result. Performance uses the
annualised Sharpe ratio and maximum drawdown; significance uses the deflated
Sharpe ratio and the probability of backtest overfitting, so an edge has to
survive both costs and the multiplicity of the configurations tried.
"""

from regime.backtest.config import BacktestConfig, load_backtest_config
from regime.backtest.engine import run_backtest
from regime.backtest.metrics import (
    annualised_return,
    annualised_vol,
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    summarise,
)
from regime.backtest.overfitting import pbo_cscv

__all__ = [
    "run_backtest",
    "BacktestConfig",
    "load_backtest_config",
    "annualised_return",
    "annualised_vol",
    "sharpe_ratio",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "summarise",
    "pbo_cscv",
]
