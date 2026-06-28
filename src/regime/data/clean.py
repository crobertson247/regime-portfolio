"""
Cleaning and validation for the price and macro panels.

Covers price validation (no negative or zero prices, a monotonic index),
inception handling for assets with ragged start dates, and missingness
reporting.

Two kinds of missing data are treated differently. Dates before an asset's
first observation are pre-inception and stay NaN. Missing days inside an
asset's coverage period are gaps; for prices these are also left as NaN, so a
halt is never filled in.
"""

from __future__ import annotations

from dataclasses import dataclass
#from typing import Optional

import pandas as pd

from regime.config import PipelineConfig
from regime.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, any]


def validate_price_series(
    df: pd.DataFrame,
    ticker: str,
    price_column: str = "Close",
) -> ValidationResult:
    """
    Validate a price series for data quality issues.

    Checks:
    1. No negative prices
    2. No zero prices
    3. No duplicate dates
    4. Monotonically increasing index
    5. Index is DatetimeIndex

    Args:
        df: DataFrame with price data.
        ticker: Ticker symbol for error messages.
        price_column: Column containing prices to validate.

    Returns:
        ValidationResult with errors, warnings, and stats.
    """
    errors = []
    warnings = []
    stats = {}

    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{ticker}: Index is not DatetimeIndex")
        return ValidationResult(False, errors, warnings, stats)

    # Check duplicate dates
    if df.index.has_duplicates:
        dup_dates = df.index[df.index.duplicated()].unique()
        errors.append(f"{ticker}: {len(dup_dates)} duplicate dates")

    # Check monotonically increasing
    if not df.index.is_monotonic_increasing:
        errors.append(f"{ticker}: Index is not monotonically increasing")

    # Check price column exists
    if price_column not in df.columns:
        errors.append(f"{ticker}: Missing price column '{price_column}'")
        return ValidationResult(len(errors) == 0, errors, warnings, stats)

    prices = df[price_column]

    # Check for negative prices
    neg_count = (prices < 0).sum()
    if neg_count > 0:
        errors.append(f"{ticker}: {neg_count} negative prices")

    # Check for zero prices
    zero_count = (prices == 0).sum()
    if zero_count > 0:
        errors.append(f"{ticker}: {zero_count} zero prices")

    # Check for NaN prices (warning - could be pre-inception)
    nan_count = prices.isna().sum()
    if nan_count > 0:
        warnings.append(f"{ticker}: {nan_count} missing prices ({nan_count/len(prices)*100:.1f}%)")

    # Collect stats
    stats = {
        "ticker": ticker,
        "total_rows": len(df),
        "valid_prices": prices.notna().sum(),
        "missing_prices": nan_count,
        "first_date": df.index.min().date() if len(df) > 0 else None,
        "last_date": df.index.max().date() if len(df) > 0 else None,
        "first_valid_date": prices.first_valid_index().date() if prices.notna().any() else None,
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_macro_series(
    df: pd.DataFrame,
    series_id: str,
) -> ValidationResult:
    """
    Validate a macro series from FRED.

    Args:
        df: DataFrame with macro data.
        series_id: FRED series ID for error messages.

    Returns:
        ValidationResult with errors, warnings, and stats.
    """
    errors = []
    warnings = []
    stats = {}

    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{series_id}: Index is not DatetimeIndex")
        return ValidationResult(False, errors, warnings, stats)

    # Check for duplicate dates
    if df.index.has_duplicates:
        dup_dates = df.index[df.index.duplicated()].unique()
        errors.append(f"{series_id}: {len(dup_dates)} duplicate dates")

    # Check data column exists
    if series_id not in df.columns and len(df.columns) == 0:
        errors.append(f"{series_id}: No data columns")
        return ValidationResult(len(errors) == 0, errors, warnings, stats)

    data_col = series_id if series_id in df.columns else df.columns[0]
    data = df[data_col]

    # Check for NaN values
    nan_count = data.isna().sum()
    if nan_count > 0:
        warnings.append(f"{series_id}: {nan_count} missing values ({nan_count/len(data)*100:.1f}%)")

    # Collect stats
    stats = {
        "series_id": series_id,
        "total_rows": len(df),
        "valid_values": data.notna().sum(),
        "missing_values": nan_count,
        "first_date": df.index.min().date() if len(df) > 0 else None,
        "last_date": df.index.max().date() if len(df) > 0 else None,
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def analyze_missingness(
    df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Analyze missingness patterns in a panel DataFrame.

    Distinguishes between:
    - Pre-inception NaNs: Before an asset's first valid observation
    - Post-inception gaps: Missing days within an asset's coverage period

    Args:
        df: DataFrame with columns for each asset/series.
        trading_days: Full trading calendar for reference.

    Returns:
        DataFrame with missingness statistics per column.
    """
    records = []

    for col in df.columns:
        series = df[col]

        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()

        if first_valid is None:
            records.append({
                "column": col,
                "first_valid_date": None,
                "last_valid_date": None,
                "total_observations": len(series),
                "valid_observations": 0,
                "pre_inception_nans": len(series),
                "post_inception_gaps": 0,
                "coverage_pct": 0.0,
            })
            continue

        # Count pre-inception NaNs
        pre_inception = series.index < first_valid
        pre_inception_count = pre_inception.sum()

        # Count post-inception gaps (NaNs within coverage period)
        in_coverage = (series.index >= first_valid) & (series.index <= last_valid)
        post_inception_nans = series[in_coverage].isna().sum()

        # Coverage percentage (within coverage period)
        coverage_period_len = in_coverage.sum()
        valid_in_coverage = series[in_coverage].notna().sum()

        records.append({
            "column": col,
            "first_valid_date": first_valid.date(),
            "last_valid_date": last_valid.date(),
            "total_observations": len(series),
            "valid_observations": series.notna().sum(),
            "pre_inception_nans": pre_inception_count,
            "post_inception_gaps": post_inception_nans,
            "coverage_pct": valid_in_coverage / coverage_period_len * 100 if coverage_period_len > 0 else 0,
        })

    return pd.DataFrame(records)


def clean_price_panel(
    price_data: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Clean and align price data into a panel DataFrame.

    Does NOT forward-fill prices (gaps are preserved as NaN).
    Pre-inception NaNs are expected and preserved.

    Args:
        price_data: Dict mapping ticker -> DataFrame with price data.
        trading_days: Trading calendar to align to.
        price_column: Column to extract from each DataFrame.

    Returns:
        DataFrame with one column per ticker, indexed by trading_days.
    """
    panels = {}

    for ticker, df in price_data.items():
        # Extract price column
        if price_column not in df.columns:
            raise ValueError(f"{ticker}: Missing price column '{price_column}'")

        prices = df[[price_column]].copy()
        prices.columns = [ticker]

        # Reindex to trading calendar WITHOUT forward-fill
        aligned = prices.reindex(trading_days)
        panels[ticker] = aligned[ticker]

    # Combine into single DataFrame
    panel = pd.DataFrame(panels)
    panel.index.name = "Date"

    logger.info(f"Created price panel: {panel.shape[0]} days x {panel.shape[1]} assets")

    return panel


def clean_macro_panel(
    macro_data: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    max_staleness: int = 5,
) -> pd.DataFrame:
    """
    Clean and align macro data into a panel DataFrame.

    Forward-fills macro data up to max_staleness days (macro data is
    published with lag, so forward-fill is appropriate).

    Args:
        macro_data: Dict mapping series_id -> DataFrame with macro data.
        trading_days: Trading calendar to align to.
        max_staleness: Maximum days to forward-fill.

    Returns:
        DataFrame with one column per series, indexed by trading_days.
    """
    panels = {}

    for series_id, df in macro_data.items():
        # Get the data column
        if series_id in df.columns:
            data = df[[series_id]].copy()
        else:
            data = df.iloc[:, [0]].copy()
            data.columns = [series_id]

        # Reindex to trading calendar with forward-fill
        aligned = data.reindex(trading_days)
        aligned = aligned.ffill(limit=max_staleness)
        panels[series_id] = aligned[series_id]

    # Combine into single DataFrame
    panel = pd.DataFrame(panels)
    panel.index.name = "Date"

    logger.info(f"Created macro panel: {panel.shape[0]} days x {panel.shape[1]} series")

    return panel