#!/usr/bin/env python3
"""
Run regime-conditioned allocation (Phase 4).

Usage:
    python scripts/run_allocation.py [--config config/allocation.yaml]
                                     [--regimes hmm|jump|changepoint]
                                     [--no-baselines]

Reads the Phase 2 returns and a Phase 3 regime label file, then produces a daily
long-only weight series two ways: the regime-switching strategy (a different
objective per regime) and the static regime-blind baselines. Weights are written
to data/processed/weights_*.parquet, with a QA report and a figure in
reports/allocation/. This is weight generation only; turning weights into
returns, costs and risk-adjusted performance is Phase 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STRESS_WINDOWS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2020 COVID": ("2020-02-15", "2020-04-30"),
    "2022 tightening": ("2022-01-01", "2022-10-31"),
}


def load_returns(features_path, assets):
    import pandas as pd

    from regime.allocation.estimators import to_simple

    feats = pd.read_parquet(features_path)
    cols = [f"{a}_ret" for a in assets]
    simple = to_simple(feats[cols].to_numpy())
    return pd.DataFrame(simple, index=feats.index, columns=assets)


def turnover(weights):
    """Daily one-way turnover series: 0.5 * sum |w_t - w_{t-1}|."""
    w = weights.dropna()
    return 0.5 * w.diff().abs().sum(axis=1)


def build_report(name, weights, regimes, cash_asset):
    import numpy as np
    import pandas as pd

    w = weights.dropna()
    to = turnover(weights)
    years = (w.index.max() - w.index.min()).days / 365.25
    rebalances = int((to > 1e-9).sum())
    lines = [
        f"# Allocation QA Report ({name})",
        "",
        f"- Weighted period: {w.index.min().date()} to {w.index.max().date()}",
        f"- Rebalances: {rebalances} ({rebalances / years:.1f} per year)",
        f"- Annual turnover: {to.sum() / years:.2f}",
        f"- Mean {cash_asset} (cash) weight: {w[cash_asset].mean() * 100:.1f}%",
        "",
        "## Average weight by asset",
        "",
        "| Asset | Mean weight |",
        "|-------|-------------|",
    ]
    for a in weights.columns:
        lines.append(f"| {a} | {w[a].mean() * 100:.1f}% |")

    if regimes is not None:
        reg = regimes.reindex(w.index)
        from regime.detection.base import regime_name

        lines += ["", "## Average weight by regime", "",
                  "| Regime | Days | " + " | ".join(weights.columns) + " |",
                  "|" + "---|" * (len(weights.columns) + 2)]
        for s in sorted(int(x) for x in reg.dropna().unique()):
            mask = (reg == s).to_numpy()
            sub = w[mask]
            cells = " | ".join(f"{sub[a].mean() * 100:.0f}%" for a in weights.columns)
            lines.append(f"| {regime_name(s)} | {len(sub)} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def make_figure(weights, regimes, n_states, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    w = weights.dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(w.index, *[w[a].to_numpy() for a in w.columns], labels=list(w.columns), alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("portfolio weight")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=len(w.columns), fontsize=8)

    if regimes is not None:
        reg = regimes.reindex(w.index).to_numpy()
        crisis = (reg == n_states - 1)
        edges = np.diff(crisis.astype(int))
        starts = np.flatnonzero(edges == 1) + 1
        ends = np.flatnonzero(edges == -1) + 1
        if crisis[0]:
            starts = np.r_[0, starts]
        if crisis[-1]:
            ends = np.r_[ends, len(crisis)]
        for s, e in zip(starts, ends):
            ax.axvspan(w.index[s], w.index[min(e, len(w.index) - 1)], color="black", alpha=0.12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None)
@click.option("--regimes", "regimes_method", default="hmm",
              help="which detector's labels to switch on (hmm | jump | changepoint)")
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "data" / "processed" / "features.parquet")
@click.option("--no-baselines", is_flag=True, help="skip the static regime-blind baselines")
def main(config, regimes_method, features_path, no_baselines):
    """Generate regime-switching and baseline weight series with QA."""
    import pandas as pd

    from regime.allocation import allocate_walk_forward, load_allocation_config
    from regime.allocation.regime_switching import RegimeSwitchingAllocator
    from regime.utils.logging import setup_logging

    setup_logging(base_path=PROJECT_ROOT)
    cfg = load_allocation_config(config)
    returns = load_returns(features_path, cfg.assets)
    cash = cfg.assets[-1]

    regime_path = PROJECT_ROOT / "data" / "processed" / f"regimes_{regimes_method}.parquet"
    regimes = pd.read_parquet(regime_path)["regime"]
    n_states = int(regimes.dropna().max()) + 1

    out_proc = PROJECT_ROOT / "data" / "processed"
    out_rep = PROJECT_ROOT / "reports" / "allocation"
    out_fig = PROJECT_ROOT / "reports" / "figures"
    for d in (out_proc, out_rep, out_fig):
        d.mkdir(parents=True, exist_ok=True)

    # --- regime-switching strategy ------------------------------------------
    switch = RegimeSwitchingAllocator(cfg.regime_allocators())
    click.echo(f"\n=== regime-switching (on {regimes_method} labels) ===")
    w_switch = allocate_walk_forward(
        returns, lambda r: switch.for_regime(r), regimes,
        lookback=cfg.lookback, rebalance_every=cfg.rebalance_every,
        on_regime_change=cfg.on_regime_change,
    )
    tag = f"regime_{regimes_method}"
    w_switch.to_parquet(out_proc / f"weights_{tag}.parquet")
    rep = build_report(tag, w_switch, regimes, cash)
    (out_rep / f"report_{tag}.md").write_text(rep, encoding="utf-8")
    make_figure(w_switch, regimes, n_states, out_fig / f"weights_{tag}.png",
                f"Regime-switching weights (on {regimes_method} regimes)")
    click.echo(rep)

    # --- static baselines ----------------------------------------------------
    if not no_baselines:
        for name in cfg.baselines:
            click.echo(f"\n=== baseline: {name} ===")
            alloc = cfg.build_allocator(name)
            w = allocate_walk_forward(
                returns, lambda r, a=alloc: a, None,
                lookback=cfg.lookback, rebalance_every=cfg.rebalance_every,
                on_regime_change=False,
            )
            w.to_parquet(out_proc / f"weights_static_{name}.parquet")
            rep = build_report(f"static {name}", w, regimes, cash)
            (out_rep / f"report_static_{name}.md").write_text(rep, encoding="utf-8")
            click.echo(rep)

    click.echo("\nOutputs in data/processed/weights_*.parquet and reports/allocation/")


if __name__ == "__main__":
    main()
