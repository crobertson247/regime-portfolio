#!/usr/bin/env python3
"""
Run regime detection (Phase 3) on the Phase 2 feature matrix.

Usage:
    python scripts/run_detection.py [--config config/detection.yaml]

Steps:
1. Load features.parquet and the detection config.
2. Standardise the detection features causally.
3. Label every trading day with a regime under causal walk-forward inference.
4. Save the labels, a QA figure, and a short report that checks the crisis
   regime concentrates in the 2008, 2020 and 2022 stress periods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Stress windows used to check the labels (start, end), inclusive.
STRESS_WINDOWS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2020 COVID": ("2020-02-15", "2020-04-30"),
    "2022 tightening": ("2022-01-01", "2022-10-31"),
}


def _runs(labels):
    """Yield (label, length) for consecutive runs of a label series."""
    import numpy as np

    vals = labels.to_numpy()
    if len(vals) == 0:
        return []
    change = np.flatnonzero(np.diff(vals) != 0) + 1
    bounds = [0, *change.tolist(), len(vals)]
    return [(vals[bounds[i]], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]


def build_report(regimes, n_states):
    import numpy as np
    import pandas as pd

    from regime.detection.base import regime_name

    labelled = regimes.dropna().astype(int)
    crisis = n_states - 1
    lines = [
        "# Regime Detection QA Report",
        "",
        f"- Labelled period: {labelled.index.min().date()} to {labelled.index.max().date()}",
        f"- Labelled days: {len(labelled)}",
        "",
        "## Regime distribution",
        "",
        "| Regime | Days | Share |",
        "|--------|------|-------|",
    ]
    for s in range(n_states):
        days = int((labelled == s).sum())
        lines.append(f"| {regime_name(s, n_states)} | {days} | {days / len(labelled) * 100:.1f}% |")

    # Persistence
    runs = _runs(labelled)
    mean_run = np.mean([length for _, length in runs]) if runs else 0.0
    switches = max(len(runs) - 1, 0)
    years = (labelled.index.max() - labelled.index.min()).days / 365.25
    lines += [
        "",
        "## Persistence",
        "",
        f"- Mean regime run length: {mean_run:.1f} trading days",
        f"- Regime switches: {switches} ({switches / years:.1f} per year)",
        "",
        "## Crisis concentration in stress windows",
        "",
        "Crisis-regime share inside each stress window versus the calm baseline "
        "outside all windows. The premise is that crisis labelling concentrates "
        "in the stress periods.",
        "",
        "| Window | Days | Crisis share |",
        "|--------|------|--------------|",
    ]
    in_any = pd.Series(False, index=labelled.index)
    for name, (a, b) in STRESS_WINDOWS.items():
        mask = (labelled.index >= a) & (labelled.index <= b)
        in_any |= mask
        sub = labelled[mask]
        share = (sub == crisis).mean() * 100 if len(sub) else float("nan")
        lines.append(f"| {name} | {len(sub)} | {share:.1f}% |")
    outside = labelled[~in_any.to_numpy()]
    base = (outside == crisis).mean() * 100 if len(outside) else float("nan")
    lines.append(f"| outside all windows | {len(outside)} | {base:.1f}% |")
    lines.append("")
    return "\n".join(lines), labelled


def make_figure(features, regimes, n_states, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labelled = regimes.dropna().astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Average correlation with crisis periods shaded
    ax = axes[0]
    if "avg_corr63d" in features.columns:
        ax.plot(features.index, features["avg_corr63d"], lw=0.7, color="steelblue")
        ax.set_ylabel("avg pairwise corr (63d)")
    crisis = n_states - 1
    in_crisis = (labelled == crisis)
    _shade(ax, labelled.index, in_crisis.to_numpy())
    ax.set_title("Average correlation with detected crisis regime shaded")

    # Regime step plot
    ax = axes[1]
    ax.step(labelled.index, labelled.to_numpy(), where="post", lw=0.8, color="black")
    ax.set_yticks(range(n_states))
    from regime.detection.base import regime_name

    ax.set_yticklabels([regime_name(s, n_states) for s in range(n_states)])
    ax.set_ylabel("regime")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _shade(ax, index, mask):
    """Shade contiguous True spans of mask on a time axis."""
    import numpy as np

    if not mask.any():
        return
    edges = np.diff(mask.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    for s, e in zip(starts, ends):
        ax.axvspan(index[s], index[min(e, len(index) - 1)], color="indianred", alpha=0.25)


@click.command()
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None,
              help="Detection config (default: config/detection.yaml)")
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "data" / "processed" / "features.parquet",
              help="Path to features.parquet")
def main(config, features_path):
    """Detect regimes and write labels, a figure and a QA report."""
    import pandas as pd

    from regime.detection import (
        HMMDetector,
        label_walk_forward,
        load_detection_config,
        prepare_detection_features,
    )
    from regime.utils.logging import setup_logging

    setup_logging(base_path=PROJECT_ROOT)
    cfg = load_detection_config(config)

    features = pd.read_parquet(features_path)
    z = prepare_detection_features(features, cfg.features, cfg.standardize_min_periods).dropna()

    def factory():
        return HMMDetector(
            n_states=cfg.n_states,
            severity_indices=cfg.severity_indices(),
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tol,
            n_restarts=cfg.n_restarts,
            random_state=cfg.random_state,
            min_covar=cfg.min_covar,
        )

    regimes = label_walk_forward(z, factory, cfg.min_train, cfg.refit_every)

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_frame().to_parquet(out_dir / "regimes.parquet")

    report, labelled = build_report(regimes, cfg.n_states)
    report_path = PROJECT_ROOT / "reports" / "detection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    fig_path = PROJECT_ROOT / "reports" / "figures" / "regimes.png"
    make_figure(features, regimes, cfg.n_states, fig_path)

    click.echo(report)
    click.echo(f"\nLabels:  {out_dir / 'regimes.parquet'}")
    click.echo(f"Report:  {report_path}")
    click.echo(f"Figure:  {fig_path}")


if __name__ == "__main__":
    main()
