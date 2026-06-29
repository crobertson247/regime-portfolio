"""
Feature preparation for regime detection.

The detectors work on a small, standardised subset of the Phase 2 feature
matrix rather than on all 39 columns: a market return, a market volatility, the
average pairwise correlation, and an implied-volatility (stress) series. This
keeps the state-space low-dimensional, which matters for a three-state Gaussian
HMM whose covariance estimates would otherwise be unstable.

Standardisation is a causal expanding z-score. The mean and standard deviation
at day t are taken over data up to and including t, so the transform adds no
look-ahead: the standardised value at t is the same whether it is computed on
the full series or on the series truncated at t.
"""

from __future__ import annotations

import pandas as pd

from regime.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_detection_features(
    feature_df: pd.DataFrame,
    columns: list[str],
    min_periods: int = 60,
) -> pd.DataFrame:
    """Select the detection columns and standardise them causally.

    Args:
        feature_df: Phase 2 feature matrix, indexed by trading day.
        columns: feature columns to use for detection.
        min_periods: minimum observations before a z-score is computed. Rows
            before this are returned as NaN and are dropped by the caller.

    Returns:
        DataFrame of expanding z-scores for the selected columns, same index.
    """
    missing = [c for c in columns if c not in feature_df.columns]
    if missing:
        raise KeyError(f"detection features not in matrix: {missing}")

    X = feature_df[columns].astype(float)
    mean = X.expanding(min_periods=min_periods).mean()
    std = X.expanding(min_periods=min_periods).std()
    z = (X - mean) / std

    logger.info(
        "Prepared %d detection features (%s), %d usable rows from %s",
        len(columns),
        ", ".join(columns),
        len(z.dropna()),
        z.dropna().index.min().date() if len(z.dropna()) else "n/a",
    )
    return z
