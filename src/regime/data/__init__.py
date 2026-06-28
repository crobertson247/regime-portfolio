"""
Data pipeline module for regime portfolio project.

Provides functionality for:
- Data ingestion from yfinance and FRED
- Calendar alignment using NYSE trading calendar
- Data cleaning and validation
- Causal feature computation
- Pipeline orchestration
"""

from regime.data.pipeline import run_pipeline

__all__ = ["run_pipeline"]
