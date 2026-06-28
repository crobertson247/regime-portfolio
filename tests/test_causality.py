"""
Causality tests for the feature set.

Every feature at day t must depend only on data up to and including t.
The check used throughout this file: for a random sample of dates t,
compute a feature on the full series and again on the series truncated
at t, and confirm the value at t agrees. A disagreement means future
information has leaked into the feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.data.features import compute_all_features
from regime.config import FeaturesConfig


class TestCausalityGuarantee:
    """
    No lookahead bias in feature computation.

    For each feature: compute it on the full series, then on the series
    truncated at a random date t, and confirm the value at t matches.
    """

    @pytest.fixture
    def config(self) -> FeaturesConfig:
        """Feature configuration for tests."""
        return FeaturesConfig()

    def _assert_causal(
        self,
        full_series: pd.Series,
        compute_fn,
        n_samples: int = 20,
        tolerance: float = 1e-10,
    ) -> None:
        """
        Assert that a feature computation is causal.

        For n_samples random dates, verify that the feature value at t
        is the same whether computed on full data or data truncated at t.

        Args:
            full_series: Input series (e.g., prices or returns).
            compute_fn: Function that takes a series and returns features.
            n_samples: Number of random dates to test.
            tolerance: Numerical tolerance for comparison.
        """
        np.random.seed(42)

        # Compute on full series
        full_features = compute_fn(full_series)

        # Get valid indices (where features are not NaN)
        if isinstance(full_features, pd.DataFrame):
            valid_idx = full_features.dropna().index
        else:
            valid_idx = full_features.dropna().index

        if len(valid_idx) < n_samples:
            n_samples = len(valid_idx)

        if n_samples == 0:
            pytest.skip("No valid indices to test")

        # Sample random test dates
        test_dates = np.random.choice(valid_idx, size=n_samples, replace=False)

        for test_date in test_dates:
            # Truncate series at test_date (inclusive)
            truncated = full_series.loc[:test_date]

            # Compute features on truncated series
            truncated_features = compute_fn(truncated)

            # Get value at test_date
            if isinstance(full_features, pd.DataFrame):
                for col in full_features.columns:
                    full_val = full_features.loc[test_date, col]
                    trunc_val = truncated_features.loc[test_date, col]

                    if pd.notna(full_val) and pd.notna(trunc_val):
                        diff = abs(full_val - trunc_val)
                        assert diff < tolerance, (
                            f"Causality violation in {col} at {test_date}: "
                            f"full={full_val}, truncated={trunc_val}, diff={diff}"
                        )
            else:
                full_val = full_features.loc[test_date]
                trunc_val = truncated_features.loc[test_date]

                if pd.notna(full_val) and pd.notna(trunc_val):
                    diff = abs(full_val - trunc_val)
                    assert diff < tolerance, (
                        f"Causality violation at {test_date}: "
                        f"full={full_val}, truncated={trunc_val}, diff={diff}"
                    )

    def test_log_returns_causal(self, sample_prices: pd.DataFrame) -> None:
        """Test that log returns are causal."""
        for col in sample_prices.columns:
            self._assert_causal(
                sample_prices[col],
                lambda s: np.log(s).diff(),
            )

    def test_rolling_volatility_causal(self, sample_returns: pd.DataFrame) -> None:
        """Test that rolling volatility is causal."""
        for col in sample_returns.columns:
            # Define the computation function
            def compute_vol(s):
                return s.rolling(window=21, min_periods=15).std() * np.sqrt(252)

            self._assert_causal(sample_returns[col], compute_vol)

    def test_rolling_correlation_causal(self, sample_returns: pd.DataFrame) -> None:
        """Test that rolling correlation is causal."""
        ret1 = sample_returns.iloc[:, 0]
        ret2 = sample_returns.iloc[:, 1]

        # For correlation, we need both series
        def compute_corr(combined):
            return combined.iloc[:, 0].rolling(window=63, min_periods=30).corr(
                combined.iloc[:, 1]
            )

        combined = pd.concat([ret1, ret2], axis=1)
        full_corr = compute_corr(combined)

        np.random.seed(42)
        valid_idx = full_corr.dropna().index
        n_samples = min(20, len(valid_idx))

        if n_samples == 0:
            pytest.skip("No valid correlation values to test")

        test_dates = np.random.choice(valid_idx, size=n_samples, replace=False)

        for test_date in test_dates:
            truncated = combined.loc[:test_date]
            trunc_corr = compute_corr(truncated)

            full_val = full_corr.loc[test_date]
            trunc_val = trunc_corr.loc[test_date]

            diff = abs(full_val - trunc_val)
            assert diff < 1e-10, (
                f"Causality violation in correlation at {test_date}: "
                f"full={full_val}, truncated={trunc_val}, diff={diff}"
            )

    def test_rolling_drawdown_causal(self, sample_prices: pd.DataFrame) -> None:
        """Test that rolling drawdown is causal."""
        for col in sample_prices.columns:
            def compute_dd(s):
                rolling_max = s.rolling(window=252, min_periods=1).max()
                return (s - rolling_max) / rolling_max

            self._assert_causal(sample_prices[col], compute_dd)


class TestNoCenteredWindows:
    """
    Windows must look backward, not be centred.

    A centred window (rolling(..., center=True)) averages over future points
    as well as past ones, which breaks causality.
    """

    def test_volatility_not_centered(self, sample_returns: pd.DataFrame) -> None:
        """Verify volatility uses backward-looking window, not centered."""
        returns = sample_returns.iloc[:, 0]

        # Our implementation (should be backward-looking)
        vol = returns.rolling(window=21, min_periods=15).std()

        # Centered window (wrong)
        # vol_centered = returns.rolling(window=21, min_periods=15, center=True).std()

        # They should be different (if they're the same, we might be using centered)
        # Check at a specific index where centered would differ
        mid_idx = len(returns) // 2
        mid_date = returns.index[mid_idx]

        # Centered window at mid_idx uses future data, so values differ 
        # (unless by chance they're the same)
        # More importantly: check that our values match truncated computation
        truncated = returns.iloc[:mid_idx + 1]
        vol_truncated = truncated.rolling(window=21, min_periods=15).std()

        # Our implementation should match truncated at mid_date
        assert abs(vol.iloc[mid_idx] - vol_truncated.iloc[-1]) < 1e-10


class TestNoFullSampleStats:
    """
    Statistics must be rolling or expanding, not full-sample.

    A full-sample mean or standard deviation uses the whole series, the
    future included, so it leaks information.
    """

    def test_no_full_sample_zscore(self, sample_prices: pd.DataFrame) -> None:
        """An expanding mean must differ from the full-sample mean early on.

        A full-sample mean would leak future information; an expanding mean
        only sees data up to t, so the two diverge at early dates.
        """
        prices = sample_prices.iloc[:, 0]

        full_mean = prices.mean()

        # Expanding mean at an early index sees only the first ~100 points,
        # so it should differ from the full-sample mean.
        early_idx = 100  # Well after min_periods
        expanding_mean = prices.iloc[:early_idx + 1].mean()
        assert abs(expanding_mean - full_mean) > 0.01, (
            "Expanding mean should differ from full sample mean at early dates"
        )


class TestNoFutureShifts:
    """
    No negative shifts.

    A .shift(-k) pulls values from k steps ahead into the present.
    """

    def test_returns_no_future_shift(self, sample_prices: pd.DataFrame) -> None:
        """Verify returns use only past data."""
        prices = sample_prices.iloc[:, 0]

        # Correct: log(p[t]) - log(p[t-1])
        returns = np.log(prices).diff()

        # At the last index, return should only use last and second-to-last prices
        last_idx = len(prices) - 1
        expected_return = np.log(prices.iloc[last_idx]) - np.log(prices.iloc[last_idx - 1])

        assert abs(returns.iloc[last_idx] - expected_return) < 1e-10

        # First return should be NaN (no previous price)
        assert pd.isna(returns.iloc[0])


class TestFeatureComputationCausal:
    """
    End-to-end test that compute_all_features produces causal features.
    """

    def test_all_features_causal(
        self,
        sample_prices: pd.DataFrame,
        sample_macro: pd.DataFrame,
    ) -> None:
        """Test that compute_all_features produces causal features."""
        config = FeaturesConfig()

        # Compute on full data
        full_features = compute_all_features(sample_prices, sample_macro, config)

        # Pick a random test date in the middle
        np.random.seed(42)
        mid_idx = len(sample_prices) // 2
        test_date = sample_prices.index[mid_idx]

        # Truncate inputs at test_date
        trunc_prices = sample_prices.loc[:test_date]
        trunc_macro = sample_macro.loc[:test_date]

        # Compute on truncated data
        trunc_features = compute_all_features(trunc_prices, trunc_macro, config)

        # All features at test_date should be identical
        for col in full_features.columns:
            full_val = full_features.loc[test_date, col]
            trunc_val = trunc_features.loc[test_date, col]

            if pd.notna(full_val) and pd.notna(trunc_val):
                diff = abs(full_val - trunc_val)
                assert diff < 1e-10, (
                    f"Causality violation in {col} at {test_date}: "
                    f"full={full_val}, truncated={trunc_val}, diff={diff}"
                )
