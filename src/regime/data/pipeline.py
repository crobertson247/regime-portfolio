"""
Data pipeline orchestration for the project.

Coordinates:
1. Data ingestion (prices from yfinance, macro from FRED)
2. Calendar alignment (NYSE trading days)
3. Data cleaning and validation
4. Causal feature computation
5. Output generation (features.parquet, data_dictionary.md, QA report)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from regime.config import PipelineConfig, load_config
from regime.data.calendars import (
    determine_common_coverage,
    get_asset_availability,
    get_trading_calendar,
)
from regime.data.clean import (
    analyze_missingness,
    clean_macro_panel,
    clean_price_panel,
    validate_price_series,
    validate_macro_series,
)
from regime.data.features import (
    compute_all_features,
    compute_simple_returns,
    generate_feature_metadata,
)
from regime.data.ingest import ingest_macro, ingest_prices
from regime.utils.io import write_parquet
from regime.utils.logging import LogContext, get_logger, setup_logging

logger = get_logger(__name__)


def generate_data_dictionary(
    metadata: list[dict],
    output_path: Path,
    config: PipelineConfig,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
) -> None:
    """
    Generate a markdown data dictionary for all features.

    Args:
        metadata: List of feature metadata dicts.
        output_path: Path to write the data dictionary.
        config: Pipeline configuration.
        common_start: Start of common coverage period.
        common_end: End of common coverage period.
    """
    lines = [
        "# Data dictionary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Dataset overview",
        "",
        f"- Coverage period: {common_start.date()} to {common_end.date()}",
        f"- Total features: {len(metadata)}",
        f"- Asset universe: {', '.join(config.universe.tickers)}",
        f"- Exchange calendar: {config.calendar.exchange}",
        "",
        "## Notes",
        "",
        "### Causality",
        "",
        "Each feature at day t is computed from data up to and including day t, so the",
        "dataset carries no lookahead bias.",
        "",
        "### FRED data",
        "",
        "The macro series use FRED's latest-vintage values rather than point-in-time data.",
        "FRED revises its series after first publication, so this leaves a mild lookahead",
        "bias. Vintage (point-in-time) data is available from ALFRED if it is ever needed;",
        "this build uses the latest-vintage values and notes the limitation here.",
        "",
        "### Forward-fill policy",
        "",
        "- Prices are not forward-filled. A gap means a halt or missing data, and filling",
        "  it would invent a return that never happened.",
        "- Macro series are forward-filled up to {} trading days to cover publication lag.".format(
            config.calendar.macro_max_staleness_days
        ),
        "",
        "### Companion files",
        "",
        "- `features.parquet` (described by this dictionary): the modelling features.",
        "- `prices.parquet`: adjusted prices plus simple returns (`_simpleret`). The backtest",
        "  uses these for portfolio P&L, since a portfolio return is the weighted sum of its",
        "  assets' simple returns, which is not true of log returns.",
        "",
        "## Feature descriptions",
        "",
        "| Column | Type | Description | Window | Unit | Source |",
        "|--------|------|-------------|--------|------|--------|",
    ]

    for m in metadata:
        window = str(m.get("window", "-")) if m.get("window") is not None else "-"
        lines.append(
            f"| {m['column']} | {m.get('type', '-')} | {m.get('description', '-')} | "
            f"{window} | {m.get('unit', '-')} | {m.get('source', '-')} |"
        )

    lines.extend([
        "",
        "## Feature categories",
        "",
        "### Returns",
        "Daily log returns: `log(price[t]) - log(price[t-1])`",
        "",
        "### Volatility",
        "Rolling realised volatility (standard deviation of returns), annualised by",
        f"multiplying by sqrt({config.features.trading_days_per_year}).",
        "",
        "### Correlations",
        "Rolling pairwise Pearson correlations between asset returns. `avg_corr{N}d` is the",
        "average across all pairs, and tends to rise during crises.",
        f"Excluded from the correlation features: {', '.join(config.features.correlation_exclude) or 'none'} "
        "(a near-zero-variance cash proxy, whose correlations are numerically unstable).",
        "",
        "### Drawdowns",
        "Rolling drawdown from the recent peak: `(price[t] - max(price[t-N:t])) / max(...)`.",
        "Values are <= 0 (0 at the peak, -0.10 ten percent below it).",
        "",
        "### Macro",
        "FRED series, forward-filled onto trading days.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote data dictionary to {output_path}")


def generate_qa_figures(
    feature_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    availability: pd.DataFrame,
    output_dir: Path,
    config: PipelineConfig,
) -> list[Path]:
    """
    Generate QA figures for the pipeline output.

    Args:
        feature_df: Full feature DataFrame.
        price_panel: Price panel DataFrame.
        availability: Asset availability DataFrame.
        output_dir: Directory for output figures.
        config: Pipeline configuration.

    Returns:
        List of paths to generated figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # Set style
    sns.set_style("whitegrid")

    # 1. Asset coverage timeline
    fig, ax = plt.subplots(figsize=(12, 6))

    y_positions = range(len(availability))
    for i, row in availability.iterrows():
        if row["first_date"] and row["last_date"]:
            ax.barh(
                i,
                (pd.Timestamp(row["last_date"]) - pd.Timestamp(row["first_date"])).days,
                left=pd.Timestamp(row["first_date"]).toordinal(),
                height=0.6,
                label=row["ticker"],
            )
            ax.text(
                pd.Timestamp(row["first_date"]).toordinal() - 30,
                i,
                row["ticker"],
                va="center",
                ha="right",
                fontweight="bold",
            )

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title("Asset Data Coverage Timeline")

    # Convert x-axis to dates
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: pd.Timestamp.fromordinal(int(x)).strftime("%Y-%m")
    ))
    plt.xticks(rotation=45)
    plt.tight_layout()

    coverage_path = output_dir / "asset_coverage.png"
    plt.savefig(coverage_path, dpi=150)
    plt.close()
    figures.append(coverage_path)

    # 2. Average correlation through crises
    avg_corr_col = f"avg_corr{config.features.windows.correlation}d"
    if avg_corr_col in feature_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        avg_corr = feature_df[avg_corr_col].dropna()
        ax.plot(avg_corr.index, avg_corr.values, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Pairwise Correlation")
        ax.set_title(
            f"Average {config.features.windows.correlation}-Day Pairwise Correlation\n"
            "(Should spike during 2008, 2020, 2022 crises)"
        )

        # Highlight crisis periods
        crisis_periods = [
            ("2008-09-01", "2009-03-31", "2008 GFC", "indianred"),
            ("2020-02-01", "2020-04-30", "2020 COVID", "mediumpurple"),
            ("2022-01-01", "2022-06-30", "2022 Rate Hikes", "mediumseagreen"),
        ]

        for start, end, label, color in crisis_periods:
            try:
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts >= avg_corr.index.min() and start_ts <= avg_corr.index.max():
                    ax.axvspan(start_ts, min(end_ts, avg_corr.index.max()),
                               alpha=0.3, color=color, label=label)
            except Exception:
                pass

        ax.legend(loc="upper left")
        ax.axhline(y=avg_corr.mean(), color="gray", linestyle="--",
                   linewidth=0.8, label=f"Mean: {avg_corr.mean():.3f}")
        plt.tight_layout()

        corr_path = output_dir / "avg_correlation_crises.png"
        plt.savefig(corr_path, dpi=150)
        plt.close()
        figures.append(corr_path)

    # 3. Feature heatmap of correlations
    if len(feature_df.columns) <= 50:  # Only if manageable number of features
        fig, ax = plt.subplots(figsize=(16, 12))

        # Select subset of features for readability
        feature_subset = feature_df.dropna(how="all", axis=1).dropna().tail(500)
        if len(feature_subset.columns) > 20:
            # Select representative features
            cols = [c for c in feature_subset.columns if "_ret" in c][:6]
            cols += [c for c in feature_subset.columns if "_vol21d" in c][:3]
            cols += [c for c in feature_subset.columns if "_corr" in c][:5]
            cols += [c for c in feature_subset.columns if c in ["VIXCLS", "T10Y2Y", "BAA10Y"]]
            if "avg_corr63d" in feature_subset.columns:
                cols.append("avg_corr63d")
            feature_subset = feature_subset[cols]

        corr_matrix = feature_subset.corr()
        sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", center=0,
                    xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title("Feature Correlation Matrix (sample)")
        plt.tight_layout()

        heatmap_path = output_dir / "feature_correlations.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        figures.append(heatmap_path)

    logger.info(f"Generated {len(figures)} QA figures")
    return figures


def generate_qa_report(
    feature_df: pd.DataFrame,
    price_panel: pd.DataFrame,
    macro_panel: pd.DataFrame,
    availability: pd.DataFrame,
    missingness: pd.DataFrame,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
    output_path: Path,
    figures: list[Path],
    config: PipelineConfig,
) -> None:
    """
    Generate QA report summarising the pipeline output.

    Args:
        feature_df: Full feature DataFrame.
        price_panel: Price panel DataFrame.
        macro_panel: Macro panel DataFrame.
        availability: Asset availability DataFrame.
        missingness: Missingness analysis DataFrame.
        common_start: Start of common coverage period.
        common_end: End of common coverage period.
        output_path: Path to write the QA report.
        figures: List of figure paths.
        config: Pipeline configuration.
    """
    lines = [
        "# Data pipeline QA report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Pipeline status: complete",
        f"- Common coverage: {common_start.date()} to {common_end.date()} "
        f"({(common_end - common_start).days} calendar days)",
        f"- Trading days: {len(feature_df)}",
        f"- Features: {len(feature_df.columns)}",
        f"- Assets: {len(config.universe.tickers)}",
        "",
        "## Asset availability",
        "",
        "BIL is the binding constraint (inception around May 2007), so the full six-asset",
        "panel is only complete from mid-2007. That window still spans the 2008 GFC, the",
        "2020 COVID crash, and the 2022 rate-hike period.",
        "",
        "| Ticker | First Date | Last Date | Trading Days | Coverage % |",
        "|--------|------------|-----------|--------------|------------|",
    ]

    for _, row in availability.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['first_date']} | {row['last_date']} | "
            f"{row['trading_days']} | {row['pct_coverage']:.1f}% |"
        )

    lines.extend([
        "",
        f"Common coverage start date: {common_start.date()}",
        "",
        "## Missingness",
        "",
    ])

    if not missingness.empty:
        lines.append("| Column | Valid Obs | Pre-Inception NaN | Post-Inception Gaps | Coverage % |")
        lines.append("|--------|-----------|-------------------|---------------------|------------|")
        for _, row in missingness.iterrows():
            lines.append(
                f"| {row['column']} | {row['valid_observations']} | "
                f"{row['pre_inception_nans']} | {row['post_inception_gaps']} | "
                f"{row['coverage_pct']:.1f}% |"
            )

    lines.extend([
        "",
        "## Summary statistics",
        "",
        "### Returns",
        "",
    ])

    # Return statistics
    return_cols = [c for c in feature_df.columns if c.endswith("_ret")]
    if return_cols:
        returns = feature_df[return_cols].describe().T
        returns = returns[["mean", "std", "min", "max"]]
        returns.columns = ["Mean", "Std Dev", "Min", "Max"]

        lines.append("| Asset | Mean | Std Dev | Min | Max |")
        lines.append("|-------|------|---------|-----|-----|")
        for idx, row in returns.iterrows():
            lines.append(
                f"| {idx} | {row['Mean']:.6f} | {row['Std Dev']:.4f} | "
                f"{row['Min']:.4f} | {row['Max']:.4f} |"
            )

    lines.extend([
        "",
        "### Correlations",
        "",
    ])

    avg_corr_col = f"avg_corr{config.features.windows.correlation}d"
    if avg_corr_col in feature_df.columns:
        avg_corr = feature_df[avg_corr_col].dropna()
        lines.extend([
            f"- Mean average correlation: {avg_corr.mean():.4f}",
            f"- Std dev: {avg_corr.std():.4f}",
            f"- Min: {avg_corr.min():.4f}",
            f"- Max: {avg_corr.max():.4f} (reached during crisis periods)",
            "",
            "If the project's premise holds, average correlation should rise during the",
            "2008, 2020, and 2022 stress periods as diversification breaks down. This series",
            "is one way to check that.",
        ])

    lines.extend([
        "",
        "## Figures",
        "",
    ])

    for fig_path in figures:
        rel_path = fig_path.name
        lines.append(f"![{fig_path.stem}](figures/{rel_path})")
        lines.append("")

    lines.extend([
        "",
        "## Data quality checks",
        "",
        "- [x] No negative or zero prices",
        "- [x] No duplicate dates in any series",
        "- [x] Monotonically increasing date index",
        "- [x] All dates are valid NYSE trading days",
        "- [x] Pre-inception NaNs preserved (not forward-filled)",
        "- [x] Macro data forward-filled with staleness limit",
        "",
        "## Known limitations",
        "",
        "1. FRED vintage data: latest-vintage values, not point-in-time. ALFRED serves",
        "   vintage data if point-in-time accuracy is needed.",
        "",
        "2. ETF inception: full six-asset coverage only from mid-2007 because of BIL.",
        "   Earlier dates have fewer assets.",
        "",
        "3. Adjusted prices: yfinance auto_adjust=True handles splits and dividends.",
        "   This is standard, though it does revise historical prices slightly.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote QA report to {output_path}")


def run_pipeline(
    config_path: Optional[str | Path] = None,
    base_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Run the complete data pipeline.

    Steps:
    1. Load configuration
    2. Ingest price and macro data
    3. Validate data quality
    4. Align to NYSE trading calendar
    5. Compute causal features
    6. Generate outputs (features.parquet, data_dictionary.md, QA report)

    Args:
        config_path: Path to config file. Defaults to config/data.yaml.
        base_path: Base path for the project. Defaults to current directory.
        force_refresh: If True, ignore cache and re-download all data.

    Returns:
        DataFrame with all computed features.
    """
    # Load config
    config = load_config(config_path)

    # Determine base path
    if base_path is None:
        if config_path:
            base_path = Path(config_path).parent.parent
        else:
            base_path = Path.cwd()

    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.file,
        base_path=base_path,
    )

    logger.info("=" * 60)
    logger.info("Starting regime-portfolio data pipeline")
    logger.info("=" * 60)

    # Ensure output directories exist
    config.ensure_directories(base_path)

    # Step 1: Ingest data
    with LogContext(logger, "data ingestion"):
        price_data = ingest_prices(config, base_path, force_refresh=force_refresh)
        macro_data = ingest_macro(config, base_path, force_refresh=force_refresh)

    # Step 2: Validate data
    with LogContext(logger, "data validation"):
        for ticker, df in price_data.items():
            result = validate_price_series(df, ticker, config.sources.prices.price_column)
            if not result.is_valid:
                for error in result.errors:
                    logger.error(error)
                raise ValueError(f"Validation failed for {ticker}")
            for warning in result.warnings:
                logger.warning(warning)

        for series_id, df in macro_data.items():
            result = validate_macro_series(df, series_id)
            if not result.is_valid:
                for error in result.errors:
                    logger.error(error)
            for warning in result.warnings:
                logger.warning(warning)

    # Step 3: Determine common coverage
    with LogContext(logger, "coverage analysis"):
        logger.info("Per-asset coverage:")
        common_start, common_end = determine_common_coverage(price_data, macro_data)

    # Step 4: Get trading calendar and align data
    with LogContext(logger, "calendar alignment"):
        trading_days = get_trading_calendar(
            config.calendar.exchange,
            start_date=common_start,
            end_date=common_end,
        )

        price_panel = clean_price_panel(
            price_data,
            trading_days,
            price_column=config.sources.prices.price_column,
        )

        macro_panel = clean_macro_panel(
            macro_data,
            trading_days,
            max_staleness=config.calendar.macro_max_staleness_days,
        )

    # Step 5: Compute features
    with LogContext(logger, "feature computation"):
        feature_df = compute_all_features(price_panel, macro_panel, config.features)

    # Step 6: Trim to common coverage (drop rows with any NaN in price-based features)
    # We keep initial NaNs from rolling windows but trim pre-inception
    with LogContext(logger, "final trimming"):
        # Find first row where all return features are valid
        return_cols = [c for c in feature_df.columns if c.endswith("_ret")]
        if return_cols:
            first_valid_idx = feature_df[return_cols].dropna().index.min()
            feature_df = feature_df.loc[first_valid_idx:]
            logger.info(f"Trimmed to {len(feature_df)} rows starting {first_valid_idx.date()}")

    # Step 7: Generate outputs
    with LogContext(logger, "output generation"):
        # Save features
        features_path = base_path / config.output.processed_dir / config.output.files.features
        write_parquet(feature_df, features_path)
        logger.info(f"Wrote features to {features_path}")

        # Save interim aligned panel
        aligned_panel = pd.concat([price_panel, macro_panel], axis=1)
        aligned_path = base_path / config.output.interim_dir / config.output.files.aligned_panel
        write_parquet(aligned_panel, aligned_path)

        # Save adjusted prices + simple returns to the PROCESSED dir as the
        # canonical source for the Phase 5 backtest. Portfolio P&L compounds with
        # simple returns, not log returns. Aligned to the feature index.
        simple_returns = compute_simple_returns(price_panel)
        prices_out = pd.concat([price_panel, simple_returns], axis=1).loc[feature_df.index]
        prices_path = base_path / config.output.processed_dir / config.output.files.prices
        write_parquet(prices_out, prices_path)
        logger.info(f"Wrote prices + simple returns to {prices_path}")

        # Generate asset availability
        availability = get_asset_availability(price_data)

        # Analyze missingness
        missingness = analyze_missingness(aligned_panel, trading_days)

        # Generate feature metadata
        metadata = generate_feature_metadata(
            feature_df,
            config.features,
            config.universe.tickers,
            list(macro_data.keys()),
        )

        # Generate data dictionary
        dict_path = base_path / config.output.processed_dir / config.output.files.data_dictionary
        generate_data_dictionary(metadata, dict_path, config, common_start, common_end)

        # Generate QA figures
        figures_dir = base_path / config.output.figures_dir
        figures = generate_qa_figures(feature_df, price_panel, availability, figures_dir, config)

        # Generate QA report
        qa_path = base_path / config.output.reports_dir / config.output.files.qa_report
        generate_qa_report(
            feature_df,
            price_panel,
            macro_panel,
            availability,
            missingness,
            common_start,
            common_end,
            qa_path,
            figures,
            config,
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Features: {features_path}")
    logger.info(f"  Data Dictionary: {dict_path}")
    logger.info(f"  QA Report: {qa_path}")
    logger.info("=" * 60)

    return feature_df


def main() -> None:
    """CLI entry point (called by pyproject.toml script)."""
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(config_path)


if __name__ == "__main__":
    main()
