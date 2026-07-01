"""Equal-weight (1/N) allocator, the regime-blind baseline."""

from __future__ import annotations

from typing import Optional

import numpy as np

from regime.allocation.base import Allocator


class EqualWeightAllocator(Allocator):
    """Hold every asset in equal proportion.

    No estimation, so nothing to go wrong; it is the reference every other
    allocator has to beat after costs.
    """

    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        return self._equal(np.asarray(returns).shape[1])
