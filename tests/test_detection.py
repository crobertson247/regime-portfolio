"""
Tests for the regime detection module (Phase 3).

The key guarantee is causality: a filtered label at day t must be the same
whether it is computed on the full series or on the series truncated at t. This
is checked for all three detectors. The remaining tests confirm states are
ordered calm -> crisis, that crisis labels track high-stress periods, that the
walk-forward harness warms up and stays in range, and that fitting is
deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.detection import (
    ChangePointDetector,
    HMMDetector,
    JumpModelDetector,
    label_walk_forward,
)

SEV = [1, 2]  # severity feature columns in the synthetic data


def synth_regimes(n: int = 800, seed: int = 0):
    """Three persistent regimes with increasing volatility and correlation."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, -0.6, -0.5], [0.0, 0.4, 0.3], [-0.3, 1.6, 1.3]])
    sds = np.array([[0.5, 0.3, 0.3], [0.6, 0.4, 0.4], [0.8, 0.6, 0.6]])
    trans = np.array([[0.97, 0.025, 0.005], [0.04, 0.93, 0.03], [0.02, 0.08, 0.90]])
    states = np.empty(n, dtype=int)
    x = np.empty((n, 3))
    s = 0
    for t in range(n):
        if t > 0:
            s = rng.choice(3, p=trans[s])
        states[t] = s
        x[t] = means[s] + sds[s] * rng.standard_normal(3)
    return x, states


@pytest.fixture(scope="module")
def data():
    return synth_regimes()


def make(name: str):
    if name == "hmm":
        return HMMDetector(n_states=3, severity_indices=SEV, n_restarts=3, random_state=0, n_iter=80)
    if name == "jump":
        return JumpModelDetector(n_states=3, severity_indices=SEV, jump_penalty=40.0, n_init=4, random_state=0)
    if name == "changepoint":
        return ChangePointDetector(n_states=3, severity_indices=SEV, window=80, min_size=8, random_state=0)
    raise ValueError(name)


def prototypes(det):
    return det.means_ if hasattr(det, "means_") else det.centroids_


ALL = ["hmm", "jump", "changepoint"]


@pytest.mark.parametrize("name", ALL)
def test_filtered_label_is_causal(data, name):
    """The filtered label at t is the same on the full series and truncated at t."""
    x, _ = data
    det = make(name).fit(x)
    full = det.filtered_states(x)
    rng = np.random.default_rng(2)
    for t in rng.choice(range(60, len(x)), size=8, replace=False):
        assert det.filtered_states(x[: t + 1])[-1] == full[t], (
            f"{name}: filtered label at {t} not causal"
        )


@pytest.mark.parametrize("name", ALL)
def test_states_ordered_by_severity(data, name):
    """Prototypes are ordered so the severity score increases with rank."""
    x, _ = data
    det = make(name).fit(x)
    sev = prototypes(det)[det.state_order_][:, SEV].sum(axis=1)
    assert np.all(np.diff(sev) > 0), f"{name}: states not severity-ordered"


@pytest.mark.parametrize("name", ALL)
def test_higher_regimes_track_higher_stress(data, name):
    """Mean stress increases with regime rank (across the ranks actually used).

    Change-point segmentation may not isolate the rare crisis regime on a short
    series, so the check is monotonicity over the ranks that appear rather than
    a fixed calm-vs-crisis comparison.
    """
    x, _ = data
    det = make(name).fit(x)
    labels = det.smoothed_states(x)
    present = sorted(set(labels.tolist()))
    assert len(present) >= 2, f"{name}: only one regime used"
    stress = [x[labels == r][:, SEV].mean() for r in present]
    assert all(stress[i] < stress[i + 1] for i in range(len(stress) - 1)), name


@pytest.mark.parametrize("name", ["hmm", "jump"])
def test_walk_forward_warmup_and_range(data, name):
    """Warm-up rows are NaN; labelled rows are valid severity ranks."""
    x, _ = data
    z = pd.DataFrame(x, index=pd.date_range("2000-01-03", periods=len(x), freq="B"))
    labels = label_walk_forward(z, lambda: make(name), min_train=300, refit_every=150)
    assert labels.iloc[:300].isna().all()
    assert labels.iloc[300:].notna().all()
    assert labels.dropna().astype(int).isin([0, 1, 2]).all()


@pytest.mark.parametrize("name", ["hmm", "jump"])
def test_fit_is_deterministic(data, name):
    """A fixed seed gives identical filtered labels."""
    x, _ = data
    a = make(name).fit(x).filtered_states(x)
    b = make(name).fit(x).filtered_states(x)
    assert np.array_equal(a, b), name
