"""Online statistics primitives.

The hot path runs once per reward call, so rolling statistics must be O(1) per
update rather than recomputing over the window each time. ``RunningStats``
implements Welford's algorithm for a numerically stable mean and variance, with
a matching ``remove`` step so it can back a fixed-size sliding window: evict the
oldest value, add the newest, and the mean and variance stay correct without a
full rescan.
"""

from __future__ import annotations

import math


class RunningStats:
    """Welford's online mean and variance with support for removal.

    ``add`` and ``remove`` are both O(1). Variance is reported as the population
    variance (divides by n), which matches ``statistics.pvariance``.
    """

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    def remove(self, x: float) -> None:
        """Reverse of ``add``. Removing the last remaining value resets to empty."""
        if self.n <= 1:
            self.n = 0
            self.mean = 0.0
            self._m2 = 0.0
            return
        n_new = self.n - 1
        mean_new = (self.n * self.mean - x) / n_new
        self._m2 -= (x - mean_new) * (x - self.mean)
        # Floating point drift can push m2 slightly negative; clamp it.
        if self._m2 < 0.0:
            self._m2 = 0.0
        self.mean = mean_new
        self.n = n_new

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
