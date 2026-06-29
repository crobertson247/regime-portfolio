"""
Causal walk-forward regime labelling.

The detector is refitted on an expanding window at regular refit points. Between
refits, the most recent model is filtered forward one day at a time. The label
for day t therefore depends only on data up to and including t: the model
parameters come from a window that ends at or before the most recent refit
point r (with r <= t), and the filtered estimate at t uses observations up to t.
No future information reaches any past label.

A warm-up of min_train days passes before the first label is emitted, so the
first model is fitted on enough history to identify regimes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from regime.detection.base import RegimeDetector
from regime.utils.logging import get_logger

logger = get_logger(__name__)


def label_walk_forward(
    z: pd.DataFrame,
    detector_factory: Callable[[], RegimeDetector],
    min_train: int = 252,
    refit_every: int = 63,
) -> pd.Series:
    """Produce point-in-time regime labels over the whole sample.

    Args:
        z: standardised detection features, indexed by trading day, with no
            missing values.
        detector_factory: returns a fresh, unfitted detector each call.
        min_train: warm-up length before the first label.
        refit_every: trading days between refits.

    Returns:
        Series of severity-ranked integer labels aligned to z.index. Rows before
        min_train are NaN (warm-up).
    """
    if z.isna().any().any():
        raise ValueError("standardised features contain NaNs; drop them first")

    x_all = z.to_numpy(dtype=float)
    n = len(z)
    labels = np.full(n, np.nan)

    if min_train >= n:
        raise ValueError(f"min_train ({min_train}) >= sample length ({n})")

    refit_points = list(range(min_train, n, refit_every))
    for i, r in enumerate(refit_points):
        det = detector_factory()
        det.fit(x_all[:r])  # parameters depend only on data before r
        end = refit_points[i + 1] if i + 1 < len(refit_points) else n
        # Filtered posteriors are causal: row t uses x_all[:t+1] only.
        states = det.filtered_states(x_all[:end])
        labels[r:end] = states[r:end]

    logger.info(
        "Walk-forward labelling complete: %d refits, %d labelled days from %s",
        len(refit_points),
        int(np.isfinite(labels).sum()),
        z.index[min_train].date(),
    )
    return pd.Series(labels, index=z.index, name="regime")
