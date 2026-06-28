"""
Calendar-alignment tests.

These check that data is aligned to the NYSE trading calendar rather than
naive business days, that no future dates appear, that price gaps are left as
NaN, and that macro series are forward-filled within the staleness limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.data.calendars import determine_common_coverage, get_trading_calendar
from regime.data.clean import clean_macro_panel, clean_price_panel


class TestTradingCalendar:
    """Tests for NYSE trading calendar."""

    def test_calendar_excludes_weekends(self) -> None:
        """Verify calendar excludes weekends."""
        calendar = get_trading_calendar("NYSE", "2020-01-01", "2020-12-31")

        # Check no Saturdays or Sundays
        weekdays = calendar.dayofweek
        assert not any(weekdays == 5), "Saturday found in trading calendar"
        assert not any(weekdays == 6), "Sunday found in trading calendar"

    def test_calendar_excludes_holidays(self) -> None:
        """Verify calendar excludes major holidays."""
        calendar = get_trading_calendar("NYSE", "2020-01-01", "2020-12-31")

        # Check specific known holidays (2020)
        holidays = [
            "2020-01-01",  # New Year's Day
            "2020-01-20",  # MLK Day
            "2020-02-17",  # Presidents Day
            "2020-04-10",  # Good Friday
            "2020-05-25",  # Memorial Day
            "2020-07-03",  # Independence Day (observed)
            "2020-09-07",  # Labor Day
            "2020-11-26",  # Thanksgiving
            "2020-12-25",  # Christmas
        ]

        for holiday in holidays:
            holiday_ts = pd.Timestamp(holiday)
            assert holiday_ts not in calendar, f"Holiday {holiday} found in calendar"

    def test_calendar_is_monotonic(self) -> None:
        """Verify calendar is monotonically increasing."""
        calendar = get_trading_calendar("NYSE", "2020-01-01", "2020-12-31")
        assert calendar.is_monotonic_increasing

    def test_calendar_no_duplicates(self) -> None:
        """Verify no duplicate dates in calendar."""
        calendar = get_trading_calendar("NYSE", "2020-01-01", "2020-12-31")
        assert not calendar.has_duplicates


class TestPriceAlignment:
    """Tests for price panel alignment."""

    def test_clean_price_panel_no_ffill(self) -> None:
        """Verify clean_price_panel does not forward-fill."""
        # Create price data with a gap
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({
            "Close": np.random.uniform(100, 110, 100),
        }, index=dates)

        # Introduce a gap
        prices.loc[dates[50:55], "Close"] = np.nan

        # Create price_data dict
        price_data = {"SPY": prices}

        # Clean and align
        trading_days = dates  # Simplified for test
        panel = clean_price_panel(price_data, trading_days, price_column="Close")

        # Gap should remain
        assert panel.loc[dates[50]:dates[54], "SPY"].isna().all()

    def test_clean_price_panel_preserves_pre_inception_nan(self) -> None:
        """Verify pre-inception NaNs are preserved, not filled."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Asset 1: Full coverage
        prices1 = pd.DataFrame({
            "Close": np.random.uniform(100, 110, 100),
        }, index=dates)

        # Asset 2: Starts at day 30 (ragged inception)
        prices2 = pd.DataFrame({
            "Close": [np.nan] * 30 + list(np.random.uniform(100, 110, 70)),
        }, index=dates)

        price_data = {
            "SPY": prices1,
            "BIL": prices2,
        }

        panel = clean_price_panel(price_data, dates, price_column="Close")

        # Pre-inception should remain NaN
        assert panel.loc[dates[:30], "BIL"].isna().all()

        # Post-inception should have values
        assert panel.loc[dates[30:], "BIL"].notna().all()


class TestMacroAlignment:
    """Tests for macro panel alignment."""

    def test_clean_macro_panel_forward_fills(self) -> None:
        """Verify clean_macro_panel forward-fills with limit."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Weekly data (every 5 days)
        macro = pd.DataFrame({
            "NFCI": [np.nan] * 100,
        }, index=dates)
        macro.iloc[::5, 0] = np.random.uniform(-1, 1, 20)

        macro_data = {"NFCI": macro}

        panel = clean_macro_panel(macro_data, dates, max_staleness=5)

        # Should have values filled for up to 5 days after each observation
        # First observation is at index 0
        # Indices 1-4 should be filled (5 days max)
        assert panel.loc[dates[1], "NFCI"] == panel.loc[dates[0], "NFCI"]
        assert panel.loc[dates[4], "NFCI"] == panel.loc[dates[0], "NFCI"]


class TestCommonCoverage:
    """Tests for common coverage determination."""

    def test_common_coverage_ragged_start(self) -> None:
        """Verify common coverage handles ragged starts correctly."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Asset 1: Full coverage
        prices1 = pd.DataFrame({"Close": [100.0] * 100}, index=dates)

        # Asset 2: Starts at day 20
        prices2 = pd.DataFrame({
            "Close": [np.nan] * 20 + [100.0] * 80,
        }, index=dates)

        # Asset 3: Starts at day 40 (binding constraint)
        prices3 = pd.DataFrame({
            "Close": [np.nan] * 40 + [100.0] * 60,
        }, index=dates)

        price_data = {
            "SPY": prices1,
            "TLT": prices2,
            "BIL": prices3,
        }

        common_start, common_end = determine_common_coverage(price_data)

        # Common start should be day 40 (when BIL starts)
        assert common_start == dates[40]
        assert common_end == dates[-1]
