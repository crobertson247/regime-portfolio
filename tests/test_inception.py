"""
Inception-handling tests.

These cover the cases that come up when assets start on different dates:
ragged start dates, pre-inception NaNs that must not be forward-filled, the
common coverage period, and the asset-availability summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.data.calendars import determine_common_coverage, get_asset_availability
from regime.data.clean import analyze_missingness, clean_price_panel


class TestRaggedInception:
    """Tests for handling ragged asset inception dates."""

    @pytest.fixture
    def ragged_price_data(self) -> dict[str, pd.DataFrame]:
        """
        Create price data mimicking the ETF universe with ragged starts.

        SPY: 1993 (full coverage in test window)
        TLT: 2002-07
        LQD: 2002-07
        GLD: 2004-11
        DBC: 2006-02
        BIL: 2007-05 (binding constraint)
        """
        dates = pd.date_range("2006-01-01", "2010-12-31", freq="B")
        n_days = len(dates)

        # Find inception indices
        def find_inception_idx(target_date: str) -> int:
            target = pd.Timestamp(target_date)
            for i, d in enumerate(dates):
                if d >= target:
                    return i
            return n_days

        inceptions = {
            "SPY": 0,  # Full coverage
            "TLT": 0,  # Full coverage in this window
            "LQD": 0,  # Full coverage in this window
            "GLD": 0,  # Full coverage in this window (2004-11 < 2006-01)
            "DBC": find_inception_idx("2006-02-03"),
            "BIL": find_inception_idx("2007-05-30"),
        }

        price_data = {}
        for ticker, inception_idx in inceptions.items():
            prices = [np.nan] * inception_idx + list(
                np.random.uniform(100, 110, n_days - inception_idx)
            )
            df = pd.DataFrame({"Close": prices}, index=dates)
            price_data[ticker] = df

        return price_data

    def test_pre_inception_nans_not_filled(
        self,
        ragged_price_data: dict[str, pd.DataFrame],
    ) -> None:
        """Verify pre-inception NaNs are preserved, not forward-filled."""
        dates = ragged_price_data["SPY"].index

        panel = clean_price_panel(ragged_price_data, dates, price_column="Close")

        # BIL should have NaNs before ~May 2007
        bil_inception = pd.Timestamp("2007-05-30")
        pre_inception = panel.index < bil_inception

        # All pre-inception values should be NaN
        pre_inception_vals = panel.loc[pre_inception, "BIL"]
        assert pre_inception_vals.isna().all(), (
            f"Pre-inception values should be NaN, but found "
            f"{pre_inception_vals.notna().sum()} non-null values"
        )

        # Post-inception values should not be NaN
        post_inception_vals = panel.loc[~pre_inception, "BIL"]
        assert post_inception_vals.notna().all(), (
            f"Post-inception values should not be NaN, but found "
            f"{post_inception_vals.isna().sum()} null values"
        )

    def test_common_coverage_uses_latest_start(
        self,
        ragged_price_data: dict[str, pd.DataFrame],
    ) -> None:
        """Verify common coverage starts at latest asset inception."""
        common_start, common_end = determine_common_coverage(ragged_price_data)

        # Common start should be approximately when BIL starts (~May 2007)
        # Allow some tolerance for exact business day
        assert common_start >= pd.Timestamp("2007-05-01")
        assert common_start <= pd.Timestamp("2007-06-30")

    def test_asset_availability_accurate(
        self,
        ragged_price_data: dict[str, pd.DataFrame],
    ) -> None:
        """Verify asset availability reporting is accurate."""
        availability = get_asset_availability(ragged_price_data)

        # SPY should have 100% coverage
        spy_row = availability[availability["ticker"] == "SPY"].iloc[0]
        assert spy_row["pct_coverage"] == 100.0

        # BIL should have less coverage (starts later)
        bil_row = availability[availability["ticker"] == "BIL"].iloc[0]
        assert bil_row["pct_coverage"] < 100.0

        # First dates should be accurate
        assert spy_row["first_date"] < bil_row["first_date"]


class TestMissingnessAnalysis:
    """Tests for missingness analysis."""

    def test_distinguishes_pre_inception_from_gaps(self) -> None:
        """Verify missingness analysis distinguishes pre-inception NaNs from gaps."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Create panel with both pre-inception NaNs and post-inception gaps
        panel = pd.DataFrame(index=dates)

        # Asset 1: Starts at day 20, has gap at days 50-52
        asset1_data = [np.nan] * 20 + [100.0] * 30 + [np.nan] * 3 + [100.0] * 47
        panel["ASSET1"] = asset1_data

        # Asset 2: Full coverage, no gaps
        panel["ASSET2"] = [100.0] * 100

        # Analyze
        missingness = analyze_missingness(panel, dates)

        # ASSET1 analysis
        asset1 = missingness[missingness["column"] == "ASSET1"].iloc[0]
        assert asset1["pre_inception_nans"] == 20
        assert asset1["post_inception_gaps"] == 3

        # ASSET2 analysis
        asset2 = missingness[missingness["column"] == "ASSET2"].iloc[0]
        assert asset2["pre_inception_nans"] == 0
        assert asset2["post_inception_gaps"] == 0

    def test_coverage_percentage_excludes_pre_inception(self) -> None:
        """Verify coverage percentage is calculated within coverage period."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Asset with pre-inception NaNs but 100% coverage after
        panel = pd.DataFrame({
            "ASSET": [np.nan] * 50 + [100.0] * 50,
        }, index=dates)

        missingness = analyze_missingness(panel, dates)
        asset_row = missingness[missingness["column"] == "ASSET"].iloc[0]

        # Coverage within the post-inception period should be 100%
        assert asset_row["coverage_pct"] == 100.0


class TestInceptionEdgeCases:
    """Tests for edge cases in inception handling."""

    def test_all_assets_same_inception(self) -> None:
        """Test when all assets have the same inception date."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        price_data = {
            "SPY": pd.DataFrame({"Close": [100.0] * 100}, index=dates),
            "TLT": pd.DataFrame({"Close": [100.0] * 100}, index=dates),
        }

        common_start, common_end = determine_common_coverage(price_data)

        assert common_start == dates[0]
        assert common_end == dates[-1]

    def test_single_asset(self) -> None:
        """Test with single asset."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        price_data = {
            "SPY": pd.DataFrame({"Close": [100.0] * 100}, index=dates),
        }

        common_start, common_end = determine_common_coverage(price_data)

        assert common_start == dates[0]
        assert common_end == dates[-1]

    def test_no_overlap_raises(self) -> None:
        """Test that non-overlapping assets raise an error."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Asset 1: Only first half
        prices1 = pd.DataFrame({
            "Close": [100.0] * 50 + [np.nan] * 50,
        }, index=dates)

        # Asset 2: Only second half
        prices2 = pd.DataFrame({
            "Close": [np.nan] * 60 + [100.0] * 40,
        }, index=dates)

        price_data = {"ASSET1": prices1, "ASSET2": prices2}

        with pytest.raises(ValueError, match="No overlapping coverage"):
            determine_common_coverage(price_data)

    def test_empty_asset_raises(self) -> None:
        """Test that all-NaN asset raises an error."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        price_data = {
            "SPY": pd.DataFrame({"Close": [100.0] * 100}, index=dates),
            "EMPTY": pd.DataFrame({"Close": [np.nan] * 100}, index=dates),
        }

        with pytest.raises(ValueError):
            determine_common_coverage(price_data)


class TestInceptionWithFeatures:
    """Tests for inception handling in feature computation."""

    def test_features_respect_inception(self, sample_prices: pd.DataFrame) -> None:
        """Verify features don't produce values before inception."""
        from regime.data.features import compute_log_returns

        # Create price data with ragged start
        dates = sample_prices.index
        modified = sample_prices.copy()
        modified.iloc[:50, 0] = np.nan  # First asset starts at day 50

        # Compute returns
        returns = compute_log_returns(modified)

        # Returns for first asset should be NaN before inception + 1
        first_asset_ret = returns.iloc[:, 0]
        assert first_asset_ret.iloc[:51].isna().all(), (
            "Returns should be NaN before inception and first valid day"
        )

    def test_rolling_features_respect_min_periods(
        self,
        sample_prices: pd.DataFrame,
    ) -> None:
        """Verify rolling features don't produce values before min_periods."""
        from regime.data.features import compute_rolling_volatility, compute_log_returns

        returns = compute_log_returns(sample_prices)

        # Compute volatility with min_periods=15
        vol = compute_rolling_volatility(
            returns,
            window=21,
            min_periods=15,
            annualize=True,
        )

        # First valid vol is at the min_periods boundary (row 15). returns[0]
        # is NaN, but it sits inside the rolling window and is simply not
        # counted toward min_periods, so it adds no extra leading NaN row.
        assert vol.iloc[:15].isna().all().all(), (
            "Volatility should be NaN before min_periods is met"
        )
        assert vol.iloc[15:].notna().any().any(), (
            "Volatility should be defined once min_periods is met"
        )
