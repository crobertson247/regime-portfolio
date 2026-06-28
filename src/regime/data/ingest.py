"""
Data ingestion from yfinance and FRED.

Fetches ETF prices from yfinance (adjusted for splits and dividends) and macro
series from FRED via pandas-datareader, caching each raw download under
data/raw/ with a fetch-date stamp so re-runs can work offline.

FRED returns latest-vintage values rather than point-in-time data; see the
README for the note on ALFRED.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from regime.config import PipelineConfig
from regime.utils.io import (
    find_latest_cache,
    get_cache_path,
    is_cache_valid,
    read_parquet,
    write_parquet,
)
from regime.utils.logging import LogContext, get_logger

logger = get_logger(__name__)


def fetch_price_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch price data from yfinance.

    Args:
        ticker: Stock/ETF ticker symbol.
        start_date: Start date (YYYY-MM-DD). Defaults to earliest available.
        end_date: End date (YYYY-MM-DD). Defaults to today.
        auto_adjust: If True, use adjusted prices (default). This handles
                     splits and dividends automatically.

    Returns:
        DataFrame with OHLCV data, indexed by date.

    Raises:
        ValueError: If no data is returned for the ticker.

    Note:
        auto_adjust=True is the recommended setting. It returns prices that
        are adjusted for corporate actions, making returns comparable over time.
    """
    logger.debug(f"Fetching {ticker} from yfinance (auto_adjust={auto_adjust})")

    # Create ticker object
    stock = yf.Ticker(ticker)

    # Fetch historical data
    if start_date is None:
        df = stock.history(period="max", end=end_date, auto_adjust=auto_adjust)
    else:
        df = stock.history(start=start_date, end=end_date, auto_adjust=auto_adjust)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Ensure index is DatetimeIndex without timezone
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    logger.info(
        f"Fetched {ticker}: {len(df)} rows from {df.index.min().date()} "
        f"to {df.index.max().date()}"
    )

    return df


def fetch_fred_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch a macro series from FRED via pandas-datareader.

    Args:
        series_id: FRED series identifier (e.g., "VIXCLS", "T10Y2Y").
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame with single column named after series_id, indexed by date.

    Raises:
        ValueError: If the series ID is invalid or no data returned.
    """
    logger.debug(f"Fetching {series_id} from FRED")

    # pandas-datareader defaults to a 2010 start when none is given, which would
    # drop the 2008 crisis. Use an early floor so None fetches full history.
    if start_date is None:
        start_date = "1990-01-01"

    try:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(
            series_id,
            "fred",
            start=start_date,
            end=end_date,
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch FRED series {series_id}: {e}") from e

    if df.empty:
        raise ValueError(f"No data returned for FRED series: {series_id}")

    # Ensure index is DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Rename column to series ID for clarity
    if len(df.columns) == 1:
        df.columns = [series_id]

    logger.info(
        f"Fetched {series_id}: {len(df)} rows from {df.index.min().date()} "
        f"to {df.index.max().date()}"
    )

    return df


def ingest_prices(
    config: PipelineConfig,
    base_path: Path,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Ingest price data for all assets in the universe.

    Args:
        config: Pipeline configuration.
        base_path: Base path for the project (for cache directories).
        use_cache: If True, use cached data when available.
        force_refresh: If True, ignore cache and re-download.

    Returns:
        Dict mapping ticker -> DataFrame with price data.
    """
    raw_dir = base_path / config.output.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    prices = {}
    today = date.today()

    with LogContext(logger, "price data ingestion"):
        for ticker in config.universe.tickers:
            # Check cache
            cache_path = get_cache_path(raw_dir, ticker, today)
            latest_cache = find_latest_cache(raw_dir, ticker)

            if use_cache and not force_refresh and latest_cache and latest_cache.exists():
                if config.cache.use_stale_cache or is_cache_valid(latest_cache, max_age_days=1):
                    logger.info(f"Using cached data for {ticker}: {latest_cache.name}")
                    prices[ticker] = read_parquet(latest_cache)
                    continue

            # Fetch fresh data
            try:
                df = fetch_price_data(
                    ticker,
                    start_date=config.dates.start_date,
                    end_date=config.dates.end_date,
                    auto_adjust=config.sources.prices.auto_adjust,
                )

                # Cache the raw download
                write_parquet(df, cache_path)
                logger.info(f"Cached {ticker} data to {cache_path.name}")

                prices[ticker] = df

            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                raise

    return prices


def ingest_macro(
    config: PipelineConfig,
    base_path: Path,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Ingest macro data from FRED.

    Args:
        config: Pipeline configuration.
        base_path: Base path for the project (for cache directories).
        use_cache: If True, use cached data when available.
        force_refresh: If True, ignore cache and re-download.

    Returns:
        Dict mapping series_id -> DataFrame with macro data.
    """
    raw_dir = base_path / config.output.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    macro = {}
    today = date.today()

    with LogContext(logger, "macro data ingestion"):
        for series_id, series_config in config.sources.macro.series.items():
            # Check cache
            cache_path = get_cache_path(raw_dir, series_id, today)
            latest_cache = find_latest_cache(raw_dir, series_id)

            if use_cache and not force_refresh and latest_cache and latest_cache.exists():
                if config.cache.use_stale_cache or is_cache_valid(latest_cache, max_age_days=1):
                    logger.info(f"Using cached data for {series_id}: {latest_cache.name}")
                    macro[series_id] = read_parquet(latest_cache)
                    continue

            # Fetch fresh data
            try:
                df = fetch_fred_series(
                    series_id,
                    start_date=config.dates.start_date,
                    end_date=config.dates.end_date,
                )

                # Cache the raw download
                write_parquet(df, cache_path)
                logger.info(f"Cached {series_id} data to {cache_path.name}")

                macro[series_id] = df

            except Exception as e:
                if series_config.optional:
                    logger.warning(f"Optional series {series_id} unavailable: {e}")
                    continue
                else:
                    logger.error(f"Failed to fetch required series {series_id}: {e}")
                    raise

    return macro