#!/usr/bin/env python3
"""
CLI entry point for the regime-portfolio data pipeline.

Usage:
    python scripts/build_dataset.py --config config/data.yaml

This script:
1. Ingests price data from yfinance and macro data from FRED
2. Aligns data to the NYSE trading calendar
3. Computes causal features (returns, volatility, correlations, drawdowns)
4. Generates features.parquet, data_dictionary.md, and QA report
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=PROJECT_ROOT / "config" / "data.yaml",
    help="Path to configuration file (default: config/data.yaml)",
)
@click.option(
    "--force-refresh",
    "-f",
    is_flag=True,
    default=False,
    help="Ignore cache and re-download all data",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging",
)
def main(config: Path, force_refresh: bool, verbose: bool) -> None:
    """
    Build the regime-portfolio feature dataset.

    Runs the complete data pipeline:
    - Ingests ETF prices from yfinance
    - Ingests macro series from FRED
    - Aligns to NYSE trading calendar
    - Computes causal features
    - Generates outputs
    """
    
    from regime.config import load_config
    from regime.data.pipeline import run_pipeline
    from regime.utils.logging import setup_logging

    # Load config to get logging settings
    pipeline_config = load_config(config)

    # Override logging level if verbose
    log_level = "DEBUG" if verbose else pipeline_config.logging.level

    # Setup logging
    setup_logging(
        level=log_level,
        log_format=pipeline_config.logging.format,
        log_file=pipeline_config.logging.file,
        base_path=PROJECT_ROOT,
    )

    click.echo(f"Configuration: {config}")
    click.echo(f"Project root: {PROJECT_ROOT}")
    click.echo(f"Force refresh: {force_refresh}")
    click.echo("")

    try:
        # Run the pipeline
        feature_df = run_pipeline(
            config_path=config,
            base_path=PROJECT_ROOT,
            force_refresh=force_refresh,
        )

        click.echo("")
        click.echo(click.style("Pipeline completed successfully!", fg="green", bold=True))
        click.echo(f"  Features shape: {feature_df.shape}")
        click.echo(f"  Output: {PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'}")

    except Exception as e:
        click.echo(click.style(f"Pipeline failed: {e}", fg="red", bold=True))
        raise click.Abort()


if __name__ == "__main__":
    main()
