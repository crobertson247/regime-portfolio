"""
Tests for the regime detection module (Phase 3).

The most important test is causality: the filtered posterior at day t must be
identical whether it is computed on the full series or on the series truncated
at t. The remaining tests check that states are ordered from calm to crisis,
that the walk-forward harness warms up and labels correctly, and that fitting is
deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.detection import HMMDetector, label_walk_forward

# Severity features in the synthetic data are columns 1 and 2.
SEV = [1, 2]


def synth_regimes(n: int = 800, seed: int = 0):
    """Three persistent regimes with increasing volatility and correlation."""
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            [0.0, -0.6, -0.5],  # calm
            [0.0, 0.4, 0.3],    # volatile
            [-0.3, 1.6, 1.3],   # crisis
        ]
    )
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


def _detector(seed: int = 0, restarts: int = 3) -> HMMDetector:
    return HMMDetector(
        n_states=3, severity_indices=SEV, n_restarts=restarts, random_state=seed, n_iter=80
    )


def test_filtered_posterior_is_causal(data):
    """Filtered posterior at t uses only data up to t."""
    x, _ = data
    det = _detector().fit(x)
    full = det.filtered_proba(x)

    rng = np.random.default_rng(1)
    test_idx = rng.choice(range(50, len(x)), size=15, replace=False)
    for t in test_idx:
        truncated = det.filtered_proba(x[: t + 1])
        assert np.allclose(full[t], truncated[-1], atol=1e-9), (
            f"filtered posterior at {t} differs between full and truncated series"
        )


def test_states_ordered_by_severity(data):
    """Calm < volatile < crisis: the severity score is non-decreasing in rank."""
    x, _ = data
    det = _detector().fit(x)
    ordered_means = det.means_[det.state_order_]
    severity = ordered_means[:, SEV].sum(axis=1)
    assert np.all(np.diff(severity) > 0), "states are not ordered by severity"


def test_crisis_label_tracks_high_stress(data):
    """Days labelled crisis have higher average stress features than calm days."""
    x, _ = data
    det = _detector().fit(x)
    labels = det.smoothed_states(x)
    calm_stress = x[labels == 0][:, SEV].mean()
    crisis_stress = x[labels == 2][:, SEV].mean()
    assert crisis_stress > calm_stress


def test_walk_forward_warmup_and_range(data):
    """Warm-up rows are NaN; labelled rows are valid severity ranks."""
    x, _ = data
    z = pd.DataFrame(x, index=pd.date_range("2000-01-03", periods=len(x), freq="B"))
    labels = label_walk_forward(
        z, lambda: _detector(restarts=2), min_train=300, refit_every=120
    )
    assert labels.iloc[:300].isna().all()
    assert labels.iloc[300:].notna().all()
    assert labels.dropna().astype(int).isin([0, 1, 2]).all()


def test_fit_is_deterministic(data):
    """A fixed seed gives identical filtered labels."""
    x, _ = data
    a = _detector(seed=7).fit(x).filtered_states(x)
    b = _detector(seed=7).fit(x).filtered_states(x)
    assert np.array_equal(a, b)
