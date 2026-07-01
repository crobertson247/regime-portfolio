"""
Tests for the backtest module (Phase 5).

The engine tests check the causal timing (the return on day t uses the weights
recorded on day t-1) and that costs reduce the net return. The metric tests
check known values and the expected direction of the probabilistic and deflated
Sharpe ratios. The PBO tests check that no-skill strategies give a high PBO
(the in-sample winner reverses out of sample) while a genuinely dominant
strategy gives a low PBO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.backtest import (
    deflated_sharpe_ratio,
    max_drawdown,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    run_backtest,
    sharpe_ratio,
    summarise,
)
from regime.backtest.metrics import expected_max_sharpe


def _frame(dct):
    idx = pd.date_range("2020-01-01", periods=len(next(iter(dct.values()))), freq="B")
    return pd.DataFrame(dct, index=idx)


# --- engine -----------------------------------------------------------------

def test_engine_is_causal():
    """The return on day t uses the weights held from day t-1, not day t."""
    returns = _frame({"A": [0.0, 0.10, 0.10, 0.10, 0.10, 0.10],
                      "B": [0.0, 0.20, 0.20, 0.20, 0.20, 0.20]})
    weights = _frame({"A": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                      "B": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
    bt = run_backtest(weights, returns, cost_bps=0.0)
    assert bt["gross"].iloc[0] == 0.0
    assert bt["gross"].iloc[1] == pytest.approx(0.10)
    assert bt["gross"].iloc[4] == pytest.approx(0.20)


def test_costs_reduce_net_return():
    returns = _frame({"A": [0.0, 0.01, 0.01, 0.01], "B": [0.0, 0.01, 0.01, 0.01]})
    weights = _frame({"A": [1.0, 0.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0, 1.0]})
    free = run_backtest(weights, returns, cost_bps=0.0)["net"].sum()
    costly = run_backtest(weights, returns, cost_bps=50.0)["net"].sum()
    assert costly < free


def test_equity_matches_cumulative_net():
    returns = _frame({"A": [0.0, 0.05, -0.02, 0.03], "B": [0.0, 0.05, -0.02, 0.03]})
    weights = _frame({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]})
    bt = run_backtest(weights, returns, cost_bps=0.0)
    assert bt["equity"].iloc[-1] == pytest.approx(float((1.0 + bt["net"]).prod()))


# --- metrics ----------------------------------------------------------------

def test_max_drawdown_known():
    net = pd.Series([0.0, -0.20, 0.0, 0.10])
    assert max_drawdown(net) == pytest.approx(0.20, abs=1e-9)


def test_sharpe_zero_variance_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 50)))


def test_sharpe_direction():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.005, 2000))
    assert sharpe_ratio(good) > 0


def test_probabilistic_sharpe_direction():
    assert probabilistic_sharpe_ratio(0.10, 0.0, 500, 0.0, 3.0) > 0.5
    assert probabilistic_sharpe_ratio(0.0, 0.10, 500, 0.0, 3.0) < 0.5


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 5)


def test_deflated_sharpe_in_unit_interval():
    rng = np.random.default_rng(1)
    net = pd.Series(rng.normal(0.0004, 0.01, 1500),
                    index=pd.date_range("2010-01-01", periods=1500, freq="B"))
    d = deflated_sharpe_ratio(net, [0.02, 0.03, 0.01, 0.04, 0.05])
    assert 0.0 <= d <= 1.0


def test_summarise_keys():
    returns = _frame({"A": [0.0, 0.01, 0.01, 0.01], "B": [0.0, 0.01, 0.01, 0.01]})
    weights = _frame({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]})
    s = summarise(run_backtest(weights, returns), rf=0.0)
    assert set(s) == {"ann_return", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"}


# --- overfitting ------------------------------------------------------------

def test_pbo_no_skill_is_high():
    """No persistent edge means the in-sample winner reverses out of sample, so
    CSCV reports a high PBO."""
    rng = np.random.default_rng(3)
    m = rng.normal(0.0, 0.01, (2400, 8))
    pbo, _ = pbo_cscv(m, n_blocks=16)
    assert pbo > 0.6


def test_pbo_dominant_strategy_is_low():
    rng = np.random.default_rng(4)
    m = rng.normal(0.0, 0.01, (2400, 6))
    m[:, 0] += 0.002
    pbo, _ = pbo_cscv(m, n_blocks=16)
    assert pbo < 0.2
