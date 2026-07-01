"""
Probability of backtest overfitting (PBO) by combinatorially symmetric
cross-validation (CSCV), following Bailey, Borwein, Lopez de Prado and Zhu.

The idea: split the return record into S equal blocks, and over every way of
choosing half the blocks as in-sample (the rest out-of-sample), pick the
strategy that looks best in-sample and see where it ranks out-of-sample. If the
in-sample winner is routinely below the out-of-sample median, the selection is
overfit. PBO is the fraction of splits where the in-sample best lands in the
bottom half out-of-sample.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 16) -> tuple[float, np.ndarray]:
    """Estimate PBO from a (T, N) matrix of strategy returns.

    Args:
        returns_matrix: T time steps by N strategies.
        n_blocks: number of CSCV blocks S (even). Splits give C(S, S/2) trials.

    Returns:
        (pbo, logits): the PBO estimate in [0, 1] and the array of logit ranks,
        one per split (negative means the in-sample best underperformed).
    """
    m = np.asarray(returns_matrix, dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least two strategies")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")

    rows = t // n_blocks
    if rows < 2:
        raise ValueError("too few observations for the requested number of blocks")
    m = m[: rows * n_blocks]
    blocks = m.reshape(n_blocks, rows, n)          # S contiguous blocks

    # Per-block sufficient statistics let each split be scored in O(N*S).
    b_sum = blocks.sum(axis=1)                     # (S, N)
    b_sq = (blocks**2).sum(axis=1)                 # (S, N)

    def sharpe(sel: list[int]) -> np.ndarray:
        k = rows * len(sel)
        s = b_sum[sel].sum(axis=0)
        sq = b_sq[sel].sum(axis=0)
        mean = s / k
        var = np.maximum(sq / k - mean**2, 1e-18)
        return mean / np.sqrt(var)

    all_blocks = range(n_blocks)
    logits = []
    for train in combinations(all_blocks, n_blocks // 2):
        test = [i for i in all_blocks if i not in train]
        is_sr = sharpe(list(train))
        oos_sr = sharpe(test)
        best = int(np.argmax(is_sr))
        # out-of-sample rank of the in-sample best (1 = worst, N = best)
        rank = int(np.argsort(np.argsort(oos_sr))[best]) + 1
        omega = rank / (n + 1)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0.0))
    return pbo, logits
