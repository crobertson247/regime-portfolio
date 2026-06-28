"""
I/O utilities for the regime portfolio data pipeline.

Provides:
- Parquet read/write with compression
- Cache path generation with date stamps
- Cache lookup and validation
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from regime.utils.logging import get_logger

logger = get_logger(__name__)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Read a parquet file into a DataFrame.

    Args:
        path: Path to the parquet file.

    Returns:
        DataFrame with the parquet contents.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.debug(f"Reading parquet file: {path}")
    df = pd.read_parquet(path)
    logger.debug(f"Read {len(df)} rows from {path}")
    return df


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    compression: str = "snappy",
) -> Path:
    """
    Write a DataFrame to a parquet file.

    Args:
        df: DataFrame to write.
        path: Output path for the parquet file.
        compression: Compression algorithm (snappy, gzip, zstd, none).

    Returns:
        Path to the written file.
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Writing {len(df)} rows to parquet: {path}")
    df.to_parquet(path, compression=compression, index=True)
    logger.debug(f"Successfully wrote parquet file: {path}")

    return path


def get_cache_path(
    base_dir: str | Path,
    identifier: str,
    fetch_date: Optional[date] = None,
    extension: str = ".parquet",
) -> Path:
    """
    Generate a cache file path with date stamp.

    Cache files are named: {identifier}_{YYYYMMDD}{extension}

    Args:
        base_dir: Base directory for cache files.
        identifier: Unique identifier (e.g., ticker symbol, FRED series ID).
        fetch_date: Date stamp for the cache file. Defaults to today.
        extension: File extension (default: .parquet).

    Returns:
        Full path to the cache file.

    Example:
        >>> get_cache_path("data/raw", "SPY", date(2024, 1, 15))
        Path("data/raw/SPY_20240115.parquet")
    """
    if fetch_date is None:
        fetch_date = date.today()

    date_str = fetch_date.strftime("%Y%m%d")
    filename = f"{identifier}_{date_str}{extension}"

    return Path(base_dir) / filename


def find_latest_cache(
    base_dir: str | Path,
    identifier: str,
    extension: str = ".parquet",
) -> Optional[Path]:
    """
    Find the most recent cache file for an identifier.

    Searches for files matching {identifier}_*.{extension} and returns
    the one with the latest date stamp.

    Args:
        base_dir: Base directory to search.
        identifier: Identifier to match.
        extension: File extension to match.

    Returns:
        Path to the latest cache file, or None if no cache exists.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    # Find all matching files
    pattern = f"{identifier}_*{extension}"
    matches = list(base_dir.glob(pattern))

    if not matches:
        return None

    # Sort by date stamp in filename (descending)
    def extract_date(p: Path) -> str:
        # Extract YYYYMMDD from {identifier}_{YYYYMMDD}.parquet
        stem = p.stem
        parts = stem.rsplit("_", 1)
        return parts[-1] if len(parts) > 1 else ""

    matches.sort(key=extract_date, reverse=True)
    return matches[0]


def get_cache_date(cache_path: Path) -> Optional[date]:
    """
    Extract the date stamp from a cache file path.

    Args:
        cache_path: Path to a cache file.

    Returns:
        Date extracted from the filename, or None if parsing fails.
    """
    stem = cache_path.stem
    parts = stem.rsplit("_", 1)

    if len(parts) != 2:
        return None

    date_str = parts[1]
    try:
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        return None


def is_cache_valid(
    cache_path: Path,
    max_age_days: Optional[int] = None,
) -> bool:
    """
    Check if a cache file is valid (exists and not too old).

    Args:
        cache_path: Path to the cache file.
        max_age_days: Maximum age in days. None means any age is valid.

    Returns:
        True if the cache is valid, False otherwise.
    """
    if not cache_path.exists():
        return False

    if max_age_days is None:
        return True

    cache_date = get_cache_date(cache_path)
    if cache_date is None:
        return False

    age = (date.today() - cache_date).days
    return age <= max_age_days
