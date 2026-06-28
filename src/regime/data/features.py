"""
Causal feature computation for the pipeline.

Every feature at day t uses only data up to and including t. In practice that
rules out centred windows, full-sample statistics, and negative shifts: each
rolling statistic looks backward from t and nothing else.

Features computed:
1. Daily log returns per asset
2. Rolling realised volatility (21d, 63d), annualised
3. Rolling pairwise correlations (63d) across the basket
4. Average pairwise correlation across the basket (63d)
5. Rolling drawdown per asset
6. Macro series (VIXCLS, T10Y2Y, BAA10Y, NFCI)
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from regime.config import FeaturesConfig
from regime.utils.logging import get_logger

logger = get_logger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from prices.

    log_return[t] = log(price[t]) - log(price[t-1]), so the return at t depends
    only on t and t-1.

    Args:
        prices: DataFrame with price columns.

    Returns:
        DataFrame with log return columns (suffixed with _ret).
    """
    log_prices = np.log(prices)
    returns = log_prices.diff()

    # Rename columns
    returns.columns = [f"{col}_ret" for col in returns.columns]

    return returns


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple (arithmetic) returns from prices.

    simple_return[t] = price[t] / price[t-1] - 1

    Strictly causal (uses only t and t-1). Simple returns are the correct basis
    for portfolio P&L in the backtest: a portfolio's return is the weighted sum
    of its assets' simple returns, an identity that does not hold for log
    returns. Kept out of the modelling feature matrix and persisted separately.

    Args:
        prices: DataFrame with price columns.

    Returns:
        DataFrame with simple return columns (suffixed with _simpleret).
    """
    returns = prices.pct_change()
    returns.columns = [f"{col}_simpleret" for col in returns.columns]
    return returns


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    min_periods: int,
    annualize: bool = True,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling realised volatility (standard deviation of returns).

    The window is backward-looking, ending at t.

    Args:
        returns: DataFrame with return columns.
        window: Rolling window size in trading days.
        min_periods: Minimum observations required.
        annualize: If True, multiply by sqrt(trading_days_per_year).
        trading_days_per_year: Annualization factor.

    Returns:
        DataFrame with volatility columns (suffixed with _vol{window}d).
    """
    # Rolling std with backward-looking window
    vol = returns.rolling(window=window, min_periods=min_periods).std()

    if annualize:
        vol = vol * np.sqrt(trading_days_per_year)

    # Rename columns
    vol.columns = [col.replace("_ret", f"_vol{window}d") for col in vol.columns]

    return vol


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int,
    min_periods: int,
    exclude: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations between all asset pairs.

    The window is backward-looking, ending at t. For n assets this gives
    n*(n-1)/2 pairs.

    Args:
        returns: DataFrame with return columns (one per asset).
        window: Rolling window size in trading days.
        min_periods: Minimum observations required.

    Returns:
        DataFrame with correlation columns named {asset1}_{asset2}_corr{window}d.
    """
    # Get asset names (strip _ret suffix if present)
    exclude_set = set(exclude or [])
    assets = [col.replace("_ret", "") for col in returns.columns]
    assets = [a for a in assets if a not in exclude_set]
    asset_pairs = list(combinations(assets, 2))

    correlations = {}

    for asset1, asset2 in asset_pairs:
        # Get return columns
        ret1 = f"{asset1}_ret" if f"{asset1}_ret" in returns.columns else asset1
        ret2 = f"{asset2}_ret" if f"{asset2}_ret" in returns.columns else asset2

        if ret1 not in returns.columns or ret2 not in returns.columns:
            continue

        # Rolling correlation
        corr = returns[ret1].rolling(window=window, min_periods=min_periods).corr(
            returns[ret2]
        )

        col_name = f"{asset1}_{asset2}_corr{window}d"
        correlations[col_name] = corr

    return pd.DataFrame(correlations, index=returns.index)


def compute_average_correlation(
    correlations: pd.DataFrame,
    window: int,
) -> pd.Series:
    """
    Average the pairwise correlations across all pairs.

    This is the series the project leans on: if diversification erodes in
    crises, the average pairwise correlation should climb during them.

    Args:
        correlations: DataFrame with pairwise correlation columns.
        window: Window size (for column naming).

    Returns:
        Series with average correlation.
    """
    # Filter to correlation columns for this window
    corr_cols = [col for col in correlations.columns if f"_corr{window}d" in col]

    if not corr_cols:
        return pd.Series(dtype=float, name=f"avg_corr{window}d")

    avg_corr = correlations[corr_cols].mean(axis=1)
    avg_corr.name = f"avg_corr{window}d"

    return avg_corr


def compute_rolling_drawdown(
    prices: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling drawdown from the recent peak.

    Drawdown[t] = (price[t] - rolling_max[t]) / rolling_max[t], where the
    rolling max is taken over the trailing window only.

    Args:
        prices: DataFrame with price columns.
        window: Lookback window for peak (default 252 = ~1 year).

    Returns:
        DataFrame with drawdown columns (suffixed with _dd).
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max

    # Rename columns
    drawdown.columns = [f"{col}_dd" for col in drawdown.columns]

    return drawdown


def compute_all_features(
    price_panel: pd.DataFrame,
    macro_panel: pd.DataFrame,
    config: FeaturesConfig,
) -> pd.DataFrame:
    """
    Compute all features from the price and macro panels.

    Every feature here is backward-looking: feature[t] uses only data up to t.

    Args:
        price_panel: DataFrame with price columns (one per asset).
        macro_panel: DataFrame with macro columns (one per series).
        config: Feature configuration (windows, min_periods, etc.).

    Returns:
        DataFrame with all feature columns, indexed by date.
    """
    logger.info("Computing causal features...")

    features = {}

    # 1. Daily log returns
    logger.info("  Computing log returns...")
    returns = compute_log_returns(price_panel)
    for col in returns.columns:
        features[col] = returns[col]

    # 2. Rolling realised volatility (short window)
    logger.info(f"  Computing {config.windows.volatility_short}d volatility...")
    vol_short = compute_rolling_volatility(
        returns,
        window=config.windows.volatility_short,
        min_periods=config.min_periods.volatility,
        trading_days_per_year=config.trading_days_per_year,
    )
    for col in vol_short.columns:
        features[col] = vol_short[col]

    # 3. Rolling realised volatility (long window)
    logger.info(f"  Computing {config.windows.volatility_long}d volatility...")
    vol_long = compute_rolling_volatility(
        returns,
        window=config.windows.volatility_long,
        min_periods=config.min_periods.volatility,
        trading_days_per_year=config.trading_days_per_year,
    )
    for col in vol_long.columns:
        features[col] = vol_long[col]

    # 4. Rolling pairwise correlations
    logger.info(f"  Computing {config.windows.correlation}d pairwise correlations...")
    correlations = compute_rolling_correlation(
        returns,
        window=config.windows.correlation,
        min_periods=config.min_periods.correlation,
        exclude=config.correlation_exclude,
    )
    for col in correlations.columns:
        features[col] = correlations[col]

    # 5. Average pairwise correlation
    logger.info("  Computing average correlation...")
    avg_corr = compute_average_correlation(correlations, config.windows.correlation)
    features[avg_corr.name] = avg_corr

    # 6. Rolling drawdown
    logger.info(f"  Computing {config.windows.drawdown}d drawdowns...")
    drawdowns = compute_rolling_drawdown(price_panel, window=config.windows.drawdown)
    for col in drawdowns.columns:
        features[col] = drawdowns[col]

    # 7. Macro series (already aligned and forward-filled in macro_panel)
    logger.info("  Adding macro series...")
    for col in macro_panel.columns:
        features[col] = macro_panel[col]

    # Combine all features
    feature_df = pd.DataFrame(features)
    feature_df.index.name = "Date"

    # Log summary
    logger.info(
        f"Computed {len(feature_df.columns)} features over {len(feature_df)} trading days"
    )

    return feature_df


def generate_feature_metadata(
    feature_df: pd.DataFrame,
    config: FeaturesConfig,
    tickers: list[str],
    macro_series: list[str],
) -> list[dict]:
    """
    Generate metadata for all features (for data dictionary).

    Args:
        feature_df: DataFrame with all features.
        config: Feature configuration.
        tickers: List of asset tickers.
        macro_series: List of macro series IDs.

    Returns:
        List of dicts with feature metadata.
    """
    metadata = []

    for col in feature_df.columns:
        info = {
            "column": col,
            "dtype": str(feature_df[col].dtype),
        }

        # Determine feature type and details
        if col.endswith("_ret"):
            ticker = col.replace("_ret", "")
            info.update({
                "type": "return",
                "description": f"Daily log return for {ticker}",
                "unit": "log return (decimal)",
                "window": 1,
                "source": "yfinance",
            })

        elif "_vol21d" in col:
            ticker = col.replace("_vol21d", "")
            info.update({
                "type": "volatility",
                "description": f"21-day rolling annualised volatility for {ticker}",
                "unit": "annualised standard deviation",
                "window": config.windows.volatility_short,
                "source": "computed from returns",
            })

        elif "_vol63d" in col:
            ticker = col.replace("_vol63d", "")
            info.update({
                "type": "volatility",
                "description": f"63-day rolling annualised volatility for {ticker}",
                "unit": "annualised standard deviation",
                "window": config.windows.volatility_long,
                "source": "computed from returns",
            })

        elif col.startswith("avg_corr"):
            info.update({
                "type": "correlation",
                "description": "Average pairwise correlation across all assets",
                "unit": "correlation coefficient (-1 to 1)",
                "window": config.windows.correlation,
                "source": "computed from pairwise correlations",
            })

        elif "_corr" in col:
            # Pair columns look like 'SPY_TLT_corr63d'
            parts = col.split("_corr")[0].split("_")
            asset1, asset2 = parts[0], parts[1]
            info.update({
                "type": "correlation",
                "description": f"{config.windows.correlation}-day rolling correlation between {asset1} and {asset2}",
                "unit": "correlation coefficient (-1 to 1)",
                "window": config.windows.correlation,
                "source": "computed from returns",
            })

        elif col.endswith("_dd"):
            ticker = col.replace("_dd", "")
            info.update({
                "type": "drawdown",
                "description": f"Rolling drawdown from {config.windows.drawdown}-day peak for {ticker}",
                "unit": "decimal (0 = at peak, -0.1 = 10% below peak)",
                "window": config.windows.drawdown,
                "source": "computed from prices",
            })

        elif col in macro_series:
            descriptions = {
                "VIXCLS": "CBOE VIX - implied volatility index (fear gauge)",
                "T10Y2Y": "10Y-2Y Treasury term spread (recession signal)",
                "BAA10Y": "Baa corporate - 10Y Treasury spread (credit spread)",
                "NFCI": "Chicago Fed National Financial Conditions Index",
            }
            info.update({
                "type": "macro",
                "description": descriptions.get(col, f"FRED series {col}"),
                "unit": "varies by series",
                "window": None,
                "source": "FRED (latest vintage, forward-filled)",
            })

        else:
            info.update({
                "type": "unknown",
                "description": f"Feature {col}",
                "unit": "unknown",
                "window": None,
                "source": "unknown",
            })

        metadata.append(info)

    return metadata