"""
Regime-switching allocator and the walk-forward weighting harness.

The switching allocator holds one sub-allocator per regime and, on each
rebalance date, dispatches to the one mapped to the current regime. The default
map follows the methodological direction: mean-variance in calm, risk parity in
volatile, and minimum-CVaR in crisis, so the objective itself changes with the
detected state rather than just the risk budget.

allocate_walk_forward turns a daily return panel and a regime label series into a
daily weight series. It rebalances on a fixed cadence and, optionally, whenever
the regime changes; between rebalances the weights are held. Every rebalance uses
only the trailing lookback window, so the weight on day t depends on returns up
to t alone.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from regime.allocation.base import Allocator


class RegimeSwitchingAllocator:
    """Dispatch to a per-regime sub-allocator.

    Args:
        allocator_by_regime: maps a severity rank (0 = calm ... n-1 = crisis) to
            the Allocator used in that regime.
        fallback: allocator used if a regime has no entry (defaults to the calm one).
    """

    def __init__(
        self,
        allocator_by_regime: dict[int, Allocator],
        fallback: Optional[Allocator] = None,
    ):
        self.allocator_by_regime = allocator_by_regime
        self.fallback = fallback or allocator_by_regime[min(allocator_by_regime)]

    def for_regime(self, regime: int) -> Allocator:
        return self.allocator_by_regime.get(int(regime), self.fallback)

    def weights(
        self,
        regime: int,
        returns: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.for_regime(regime).weights(returns, prev_weights)


def allocate_walk_forward(
    returns: pd.DataFrame,
    choose: Callable[[Optional[int]], Allocator],
    regimes: Optional[pd.Series] = None,
    lookback: int = 252,
    rebalance_every: int = 21,
    on_regime_change: bool = True,
) -> pd.DataFrame:
    """Daily long-only weights from a causal walk-forward loop.

    Args:
        returns: simple returns, dates x assets.
        choose: maps the current regime label to an Allocator. For a static
            baseline, ignore the argument and return the same allocator.
        regimes: per-date regime labels aligned to returns.index. If None, the
            label passed to choose is None and rebalancing is purely periodic.
        lookback: trailing window length used to estimate each rebalance.
        rebalance_every: trading days between scheduled rebalances.
        on_regime_change: also rebalance on the day the regime label changes.

    Returns:
        A weights DataFrame on returns.index; rows before the first rebalance and
        any row whose regime label is missing are NaN.
    """
    assets = list(returns.columns)
    n = len(assets)
    r = returns.to_numpy()
    dates = returns.index

    if regimes is not None:
        reg = regimes.reindex(dates).to_numpy(dtype=float)
    else:
        reg = np.full(len(dates), np.nan)

    weights = np.full((len(dates), n), np.nan)
    current = None
    last_rebalance = -10**9
    last_regime = None

    for t in range(len(dates)):
        regime_t = reg[t]
        have_regime = regimes is None or not np.isnan(regime_t)
        warm = t + 1 >= lookback

        if warm and have_regime:
            due = (t - last_rebalance) >= rebalance_every
            changed = on_regime_change and regimes is not None and regime_t != last_regime
            if current is None or due or changed:
                window = r[t + 1 - lookback : t + 1]
                label = None if regimes is None else int(regime_t)
                current = choose(label).weights(window, current)
                last_rebalance = t
                last_regime = regime_t

        if current is not None:
            weights[t] = current

    return pd.DataFrame(weights, index=dates, columns=assets)
