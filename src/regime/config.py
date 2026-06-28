"""
Configuration loading and validation for the regime portfolio data pipeline.

Uses Pydantic for type-safe configuration with validation.
All magic numbers and tuneable parameters are defined in config/data.yaml.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class AssetConfig(BaseModel):
    """Configuration for a single asset in the universe."""

    description: str
    approx_inception: str  # ISO date string

    @field_validator("approx_inception")
    @classmethod
    def validate_inception_date(cls, v: str) -> str:
        """Validate that inception date is a valid ISO date string."""
        try:
            date.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"Invalid inception date format: {v}. Use YYYY-MM-DD.") from e
        return v


class UniverseConfig(BaseModel):
    """Configuration for the asset universe."""

    name: str
    currency: str = "USD"
    exchange: str = "NYSE"
    assets: dict[str, AssetConfig]

    @property
    def tickers(self) -> list[str]:
        """Return list of ticker symbols."""
        return list(self.assets.keys())


class DatesConfig(BaseModel):
    """Configuration for date range."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate date format if provided."""
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError as e:
                raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD.") from e
        return v


class PricesSourceConfig(BaseModel):
    """Configuration for price data source."""

    provider: str = "yfinance"
    auto_adjust: bool = True
    price_column: str = "Close"


class MacroSeriesConfig(BaseModel):
    """Configuration for a single FRED macro series."""

    description: str
    frequency: str = "daily"
    optional: bool = False


class MacroSourceConfig(BaseModel):
    """Configuration for macro data source."""

    provider: str = "fred"
    api_key_env_var: str = "FRED_API_KEY"
    series: dict[str, MacroSeriesConfig]

    @property
    def series_ids(self) -> list[str]:
        """Return list of FRED series IDs."""
        return list(self.series.keys())

    @property
    def required_series_ids(self) -> list[str]:
        """Return list of required (non-optional) FRED series IDs."""
        return [sid for sid, cfg in self.series.items() if not cfg.optional]


class SourcesConfig(BaseModel):
    """Configuration for all data sources."""

    prices: PricesSourceConfig
    macro: MacroSourceConfig


class CalendarConfig(BaseModel):
    """Configuration for calendar and alignment."""

    exchange: str = "NYSE"
    macro_max_staleness_days: int = Field(default=5, ge=1, le=30)
    price_forward_fill: bool = False


class WindowsConfig(BaseModel):
    """Configuration for feature rolling window lengths."""

    volatility_short: int = Field(default=21, ge=5)
    volatility_long: int = Field(default=63, ge=21)
    correlation: int = Field(default=63, ge=21)
    drawdown: int = Field(default=252, ge=63)


class MinPeriodsConfig(BaseModel):
    """Configuration for minimum periods in rolling calculations."""

    volatility: int = Field(default=15, ge=5)
    correlation: int = Field(default=30, ge=10)


class FeaturesConfig(BaseModel):
    """Configuration for feature computation."""

    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    trading_days_per_year: int = Field(default=252, ge=200, le=260)
    min_periods: MinPeriodsConfig = Field(default_factory=MinPeriodsConfig)
    # Assets excluded from correlation features (e.g. near-zero-variance cash proxy).
    correlation_exclude: list[str] = Field(default_factory=list)


class FilesConfig(BaseModel):
    """Configuration for output file names."""

    features: str = "features.parquet"
    prices: str = "prices.parquet"
    aligned_panel: str = "aligned_panel.parquet"
    data_dictionary: str = "data_dictionary.md"
    qa_report: str = "qa_report.md"


class OutputConfig(BaseModel):
    """Configuration for output directories and files."""

    processed_dir: str = "data/processed"
    interim_dir: str = "data/interim"
    raw_dir: str = "data/raw"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    files: FilesConfig = Field(default_factory=FilesConfig)


class CacheConfig(BaseModel):
    """Configuration for data caching."""

    enabled: bool = True
    use_stale_cache: bool = True


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "reports/pipeline.log"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}. Must be one of {valid_levels}.")
        return v.upper()


class PipelineConfig(BaseModel):
    """
    Root configuration for the data pipeline.

    All configuration is loaded from config/data.yaml and validated here.
    No magic numbers should exist outside this configuration.
    """

    universe: UniverseConfig
    dates: DatesConfig = Field(default_factory=DatesConfig)
    sources: SourcesConfig
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated PipelineConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the config file is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raise ValueError(f"Empty configuration file: {path}")

        return cls.model_validate(raw_config)

    def ensure_directories(self, base_path: Path) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            base_path / self.output.processed_dir,
            base_path / self.output.interim_dir,
            base_path / self.output.raw_dir,
            base_path / self.output.reports_dir,
            base_path / self.output.figures_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str | Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration from file.

    Args:
        config_path: Path to config file. Defaults to config/data.yaml.

    Returns:
        Validated PipelineConfig instance.
    """
    if config_path is None:
        # Default to config/data.yaml relative to project root
        config_path = Path(__file__).parent.parent.parent / "config" / "data.yaml"

    return PipelineConfig.from_yaml(config_path)
