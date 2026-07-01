#!/usr/bin/env python3
"""
Walk-forward backtest and evaluation (Phase 5).

Usage:
    python scripts/run_backtest.py [--config config/backtest.yaml] [--cost-bps 10]

Reads the Phase 2 returns and every weight series in data/processed
(weights_*.parquet from Phase 4), applies each to realised returns net of
transaction costs, and scores them. It reports annualised return, volatility,
Sharpe ratio, maximum drawdown and turnover over the full sample and the three
stress windows, then the deflated Sharpe ratio and the probability of backtest
overfitting across the whole set, plus a cost-sensitivity sweep. Equity and
drawdown figures and a markdown report are written to reports/backtest/.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_returns(features_path, assets):
    import pandas as pd

    from regime.allocation.estimators import to_simple

    feats = pd.read_parquet(features_path)
    cols = [f"{a}_ret" for a in assets]
    return pd.DataFrame(to_simple(feats[cols].to_numpy()), index=feats.index, columns=assets)


def pretty(name):
    return name.replace("regime_", "regime-switch (").replace("static_", "").replace("_", " ") + (
        ")" if name.startswith("regime_") else ""
    )


def window_stats(net, a, b):
    from regime.backtest.metrics import max_drawdown

    sub = net.loc[(net.index >= a) & (net.index <= b)].dropna()
    if len(sub) == 0:
        return float("nan"), float("nan")
    total = float((1.0 + sub).prod() - 1.0)
    return total, max_drawdown(sub)


def make_figures(nets, regimes, stress, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    # equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, net in nets.items():
        eq = (1.0 + net.dropna()).cumprod()
        ax.plot(eq.index, eq.to_numpy(), lw=1.2 if name.startswith("regime_") else 0.9,
                label=pretty(name), alpha=0.9 if name.startswith("regime_") else 0.7)
    for a, b in stress.values():
        ax.axvspan(a, b, color="grey", alpha=0.15)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title("Net-of-cost equity curves; stress windows shaded")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves.png", dpi=150)
    plt.close(fig)

    # drawdown of the first regime strategy vs equal weight
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for name in nets:
        if name.startswith("regime_") or name == "static_equal_weight":
            net = nets[name].dropna()
            eq = (1.0 + net).cumprod()
            dd = (eq / eq.cummax() - 1.0) * 100
            ax.plot(dd.index, dd.to_numpy(), lw=1.0, label=pretty(name))
    for a, b in stress.values():
        ax.axvspan(a, b, color="grey", alpha=0.15)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Drawdown: regime-switching vs equal weight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "drawdowns.png", dpi=150)
    plt.close(fig)


@click.command()
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None)
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "data" / "processed" / "features.parquet")
@click.option("--cost-bps", type=float, default=None, help="override the transaction cost")
def main(config, features_path, cost_bps):
    """Backtest every weight series and write the evaluation report."""
    import numpy as np
    import pandas as pd

    from regime.backtest import (
        deflated_sharpe_ratio, load_backtest_config, pbo_cscv, run_backtest, sharpe_ratio, summarise,
    )
    from regime.utils.logging import setup_logging

    setup_logging(base_path=PROJECT_ROOT)
    cfg = load_backtest_config(config)
    cost = cfg.cost_bps if cost_bps is None else cost_bps
    returns = load_returns(features_path, cfg.assets)
    rf = returns[cfg.rf_asset]

    weight_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "processed" / "weights_*.parquet")))
    if not weight_files:
        raise SystemExit("No weights_*.parquet found. Run scripts/run_allocation.py first.")

    bts, nets = {}, {}
    for p in weight_files:
        name = Path(p).stem.replace("weights_", "")
        bts[name] = run_backtest(pd.read_parquet(p), returns, cost_bps=cost)
        nets[name] = bts[name]["net"]

    # Common evaluation window: the first date on which every strategy is live.
    # Metrics are computed here so no strategy is credited for a period it could
    # not trade (e.g. before the detector has warmed up).
    common = None
    for net in nets.values():
        idx = net.dropna().index
        common = idx if common is None else common.intersection(idx)
    nets = {k: v.reindex(common) for k, v in nets.items()}
    rows = {k: summarise(bts[k].reindex(common), rf) for k in bts}
    trial_sharpes = [sharpe_ratio(nets[k], rf, annualise=False) for k in nets]

    lines = [
        f"# Backtest evaluation (cost {cost:.0f} bps)",
        "",
        f"- Assets: {', '.join(cfg.assets)}",
        f"- Evaluated window: {common.min().date()} to {common.max().date()} ({len(common)} days)",
        "",
        "## Performance (full sample, net of cost)",
        "",
        "| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Calmar | Turnover | Deflated Sharpe |",
        "|----------|-------------|----------|--------|--------|--------|----------|-----------------|",
    ]
    for name in nets:
        m = rows[name]
        dsr = deflated_sharpe_ratio(nets[name], trial_sharpes, rf)
        calmar = m["ann_return"] / m["max_drawdown"] if m["max_drawdown"] > 0 else float("nan")
        lines.append(
            f"| {pretty(name)} | {m['ann_return']*100:.1f}% | {m['ann_vol']*100:.1f}% | "
            f"{m['sharpe']:.2f} | {m['max_drawdown']*100:.1f}% | {calmar:.2f} | {m['ann_turnover']:.2f} | {dsr:.2f} |"
        )

    # stress-window returns
    lines += ["", "## Return within stress windows (net of cost)", "",
              "| Strategy | " + " | ".join(cfg.stress_windows) + " |",
              "|" + "---|" * (len(cfg.stress_windows) + 1)]
    for name in nets:
        cells = []
        for a, b in cfg.stress_windows.values():
            ret, _ = window_stats(nets[name], a, b)
            cells.append(f"{ret*100:+.1f}%")
        lines.append(f"| {pretty(name)} | " + " | ".join(cells) + " |")

    # significance
    m = np.column_stack([nets[k].reindex(common).to_numpy() for k in nets])
    pbo, _ = pbo_cscv(m, n_blocks=cfg.cscv_blocks)
    lines += ["", "## Significance / overfitting", "",
              f"- Strategies compared: {len(nets)}",
              f"- Probability of backtest overfitting (CSCV, {cfg.cscv_blocks} blocks): {pbo:.2f}",
              "- Deflated Sharpe ratio is the last column of the performance table "
              "(probability the Sharpe beats the expected best of the trials).", ""]

    # cost sensitivity for the regime strategies
    lines += ["## Cost sensitivity (Sharpe ratio)", "",
              "| Strategy | " + " | ".join(f"{c:.0f} bps" for c in cfg.cost_grid_bps) + " |",
              "|" + "---|" * (len(cfg.cost_grid_bps) + 1)]
    show = [k for k in nets if k.startswith("regime_")] + ["static_equal_weight", "static_risk_parity"]
    for name in [k for k in show if k in nets]:
        w = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / f"weights_{name}.parquet")
        cells = []
        for c in cfg.cost_grid_bps:
            bt = run_backtest(w, returns, cost_bps=c)
            cells.append(f"{sharpe_ratio(bt['net'], rf):.2f}")
        lines.append(f"| {pretty(name)} | " + " | ".join(cells) + " |")
    lines.append("")

    report = "\n".join(lines)
    out_rep = PROJECT_ROOT / "reports" / "backtest"
    out_rep.mkdir(parents=True, exist_ok=True)
    (out_rep / "report.md").write_text(report, encoding="utf-8")
    make_figures(nets, None, cfg.stress_windows, PROJECT_ROOT / "reports" / "figures")
    click.echo(report)
    click.echo("\nOutputs in reports/backtest/report.md and reports/figures/{equity_curves,drawdowns}.png")


if __name__ == "__main__":
    main()
