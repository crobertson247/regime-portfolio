"""
Walk-forward backtest engine.

Turns a daily weight series into a realised, net-of-cost portfolio return series.
The timing is deliberately causal: the weights recorded on day t are decided from
data up to t and are only put to work on day t+1, so the return earned on any day
uses weights that were already known the day before. Rebalancing is charged a
proportional transaction cost on the traded notional.
"""

from __future__ import annotations

import pandas as pd


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Backtest a weight series against asset returns.

    Args:
        weights: daily target weights (dates x assets); warm-up rows may be NaN.
        returns: simple asset returns (dates x assets).
        cost_bps: proportional cost per unit of notional traded, in basis points.
            A rebalance that changes weights by a total of sum|dw| is charged
            cost_bps/10000 * sum|dw|, which applies the one-way cost to each leg.

    Returns:
        A DataFrame indexed by date with columns:
        gross (pre-cost return), cost, net (post-cost return), turnover and
        equity (the cumulative net value, starting at 1).
    """
    w = weights.dropna()
    common = w.index.intersection(returns.index)
    w = w.loc[common]
    r = returns.loc[common, w.columns]

    held = w.shift(1).fillna(0.0)          # weights in effect during day t
    gross = (held * r).sum(axis=1)

    prev_held = held.shift(1).fillna(0.0)  # weights held the day before
    turnover = (held - prev_held).abs().sum(axis=1)
    cost = (cost_bps / 1e4) * turnover
    net = gross - cost

    return pd.DataFrame(
        {"gross": gross, "cost": cost, "net": net, "turnover": turnover, "equity": (1.0 + net).cumprod()},
        index=common,
    )
