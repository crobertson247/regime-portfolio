#!/usr/bin/env python3
"""
Run regime detection (Phase 3) on the Phase 2 feature matrix.

Usage:
    python scripts/run_detection.py [--config config/detection.yaml] [--method hmm|jump|changepoint|all]

For a single method it labels every trading day under causal walk-forward
inference and writes the labels, a QA figure and a report that checks the crisis
regime concentrates in the 2008, 2020 and 2022 stress periods. With --method all
it runs the three detectors and adds a comparison (regime mix, crisis
concentration and pairwise agreement).
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


def _runs(labels):
    import numpy as np

    vals = labels.to_numpy()
    if len(vals) == 0:
        return []
    change = np.flatnonzero(np.diff(vals) != 0) + 1
    bounds = [0, *change.tolist(), len(vals)]
    return [(vals[bounds[i]], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]


def build_report(regimes, n_states, method):
    import numpy as np
    import pandas as pd

    from regime.detection.base import regime_name

    labelled = regimes.dropna().astype(int)
    crisis = n_states - 1
    lines = [
        f"# Regime Detection QA Report ({method})",
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
    return "\n".join(lines)


def _shade(ax, index, mask):
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


def make_figure(features, regimes, n_states, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from regime.detection.base import regime_name

    labelled = regimes.dropna().astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    if "avg_corr63d" in features.columns:
        ax.plot(features.index, features["avg_corr63d"], lw=0.7, color="steelblue")
        ax.set_ylabel("avg pairwise corr (63d)")
    in_crisis = (labelled == n_states - 1)
    _shade(ax, labelled.index, in_crisis.to_numpy())
    ax.set_title("Average correlation with detected crisis regime shaded")
    ax = axes[1]
    ax.step(labelled.index, labelled.to_numpy(), where="post", lw=0.8, color="black")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([regime_name(s, n_states) for s in range(n_states)])
    ax.set_ylabel("regime")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_report(regimes_by_method, n_states):
    import numpy as np
    import pandas as pd

    crisis = n_states - 1
    methods = list(regimes_by_method)
    lines = ["# Detector comparison", "", "## Crisis share by detector", "",
             "| Window | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    rows = {**STRESS_WINDOWS, "outside all windows": None}
    common = None
    for r in regimes_by_method.values():
        idx = r.dropna().index
        common = idx if common is None else common.intersection(idx)
    for name, win in rows.items():
        cells = []
        for m in methods:
            lab = regimes_by_method[m].dropna().astype(int)
            if win is None:
                mask = pd.Series(True, index=lab.index)
                for a, b in STRESS_WINDOWS.values():
                    mask &= ~((lab.index >= a) & (lab.index <= b))
            else:
                a, b = win
                mask = (lab.index >= a) & (lab.index <= b)
            sub = lab[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
            cells.append(f"{(sub == crisis).mean() * 100:.0f}%" if len(sub) else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Pairwise label agreement (common days)", "",
              "| Pair | Agreement |", "|------|-----------|"]
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = regimes_by_method[methods[i]].reindex(common).astype(int)
            b = regimes_by_method[methods[j]].reindex(common).astype(int)
            agree = (a.to_numpy() == b.to_numpy()).mean() * 100
            lines.append(f"| {methods[i]} vs {methods[j]} | {agree:.0f}% |")
    lines.append("")
    return "\n".join(lines)


def label_one(method, z, features, cfg, project_root):
    from regime.detection import label_walk_forward

    factory = cfg.build_factory(method)
    regimes = label_walk_forward(z, factory, cfg.min_train, cfg.refit_every)
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_frame().to_parquet(out_dir / f"regimes_{method}.parquet")
    report = build_report(regimes, cfg.n_states, method)
    (project_root / "reports" / "detection").mkdir(parents=True, exist_ok=True)
    (project_root / "reports" / "detection" / f"report_{method}.md").write_text(report, encoding="utf-8")
    make_figure(features, regimes, cfg.n_states, project_root / "reports" / "figures" / f"regimes_{method}.png")
    return regimes, report


@click.command()
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None)
@click.option("--method", "-m", default=None,
              help="hmm | jump | changepoint | all (default: config method)")
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "data" / "processed" / "features.parquet")
def main(config, method, features_path):
    """Detect regimes with one or all detectors and write labels, figures, reports."""
    import pandas as pd

    from regime.detection import prepare_detection_features, load_detection_config
    from regime.utils.logging import setup_logging

    setup_logging(base_path=PROJECT_ROOT)
    cfg = load_detection_config(config)
    method = (method or cfg.method).lower()

    features = pd.read_parquet(features_path)
    z = prepare_detection_features(features, cfg.features, cfg.standardize_min_periods).dropna()

    methods = ["hmm", "jump", "changepoint"] if method == "all" else [method]
    regimes_by_method = {}
    for m in methods:
        click.echo(f"\n=== {m} ===")
        regimes, report = label_one(m, z, features, cfg, PROJECT_ROOT)
        regimes_by_method[m] = regimes
        click.echo(report)

    if len(methods) > 1:
        comp = compare_report(regimes_by_method, cfg.n_states)
        (PROJECT_ROOT / "reports" / "detection" / "comparison.md").write_text(comp, encoding="utf-8")
        click.echo("\n" + comp)
    click.echo("\nOutputs in data/processed/regimes_*.parquet and reports/detection/")


if __name__ == "__main__":
    main()
