"""
Trading-calendar handling and coverage.

Uses pandas-market-calendars for the NYSE trading calendar, which is the master
index the panels align to. Also works out the common coverage period across
assets and summarises per-asset availability.

The exchange and currency are config-driven, so the same code could later cover
JSE (ZAR) assets without changing the feature computation.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal

from regime.utils.logging import get_logger

logger = get_logger(__name__)


def get_trading_calendar(
    exchange: str = "NYSE",
    start_date: Optional[str | date] = None,
    end_date: Optional[str | date] = None,
) -> pd.DatetimeIndex:
    """
    Get trading days for an exchange calendar.

    Args:
        exchange: Exchange code (NYSE, JSE, etc.). See pandas_market_calendars.
        start_date: Start date for the calendar range.
        end_date: End date for the calendar range.

    Returns:
        DatetimeIndex of valid trading days.

    Raises:
        ValueError: If the exchange code is not recognized.
    """
    try:
        calendar = mcal.get_calendar(exchange)
    except Exception as e:
        raise ValueError(f"Unknown exchange calendar: {exchange}") from e

    # Convert dates to strings if needed
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    # Get schedule (includes early closes, holidays excluded)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)

    # Extract just the dates (market_open column has the date)
    trading_days = pd.DatetimeIndex(schedule.index)

    logger.debug(
        f"Got {len(trading_days)} trading days for {exchange} "
        f"from {trading_days.min().date()} to {trading_days.max().date()}"
    )

    return trading_days


def determine_common_coverage(
    price_data: dict[str, pd.DataFrame],
    macro_data: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Determine the common coverage period where all assets have data.

    The start date is the latest first valid observation across all assets.
    The end date is the earliest last valid observation across all assets.

    Args:
        price_data: Dict mapping ticker -> DataFrame with price data.
        macro_data: Optional dict mapping series_id -> DataFrame with macro data.

    Returns:
        Tuple of (common_start, common_end) timestamps.

    Raises:
        ValueError: If there is no overlapping coverage period.
    """
    start_dates = []
    end_dates = []

    # Get coverage from price data
    for ticker, df in price_data.items():
        if df.empty:
            raise ValueError(f"No data for ticker {ticker}")

        first_valid = df.first_valid_index()
        last_valid = df.last_valid_index()

        if first_valid is None or last_valid is None:
            raise ValueError(f"No valid data for ticker {ticker}")

        start_dates.append(first_valid)
        end_dates.append(last_valid)

        logger.info(
            f"  {ticker}: {first_valid.date()} to {last_valid.date()} "
            f"({(last_valid - first_valid).days} days)"
        )

    # Get coverage from macro data
    if macro_data:
        for series_id, df in macro_data.items():
            if not df.empty:
                first_valid = df.first_valid_index()
                last_valid = df.last_valid_index()

                if first_valid is not None and last_valid is not None:
                    # More lenient for marco data - just log it
                    logger.info(
                        f"  {series_id}: {first_valid.date()} to {last_valid.date()}"
                    )

    # All assets must have data
    common_start = max(start_dates)
    common_end = min(end_dates)

    if common_start >= common_end:
        raise ValueError(
            f"No overlapping coverage period. "
            f"Latest start: {common_start.date()}, Earliest end: {common_end.date()}"
        )

    logger.info(
        f"Common coverage period: {common_start.date()} to {common_end.date()} "
        f"({(common_end - common_start).days} days)"
    )

    return common_start, common_end


def get_asset_availability(
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Create a summary of asset availability.

    Args:
        price_data: Dict mapping ticker -> DataFrame with price data.

    Returns:
        DataFrame with columns: ticker, first_date, last_date, trading_days, pct_coverage
    """
    records = []

    for ticker, df in price_data.items():
        if df.empty:
            records.append({
                "ticker": ticker,
                "first_date": None,
                "last_date": None,
                "trading_days": 0,
                "pct_coverage": 0.0,
            })
        else:
            first_valid = df.first_valid_index()
            last_valid = df.last_valid_index()
            valid_count = df.notna().sum().iloc[0] if len(df.columns) > 0 else 0

            records.append({
                "ticker": ticker,
                "first_date": first_valid.date() if first_valid else None,
                "last_date": last_valid.date() if last_valid else None,
                "trading_days": int(valid_count),
                "total_days": len(df),
            })

    availability = pd.DataFrame(records)

    # Calculate coverage percentage
    if not availability.empty and availability["total_days"].max() > 0:
        max_days = availability["total_days"].max()
        availability["pct_coverage"] = (
            availability["trading_days"] / max_days * 100
        ).round(1)
    else:
        availability["pct_coverage"] = 0.0

    return availability