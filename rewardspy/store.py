"""In-memory time-series store for reward observations.

``MetricStore`` keeps the full recent history in a bounded deque and maintains
O(1) rolling statistics over a sliding window. It is the single source of truth
that detectors read from and the dashboard renders.

Memory is bounded on purpose. Training runs can have millions of steps, so the
store keeps at most ``max_records`` records (10k by default), which is plenty
for statistical analysis. The full history is persisted by the exporters.
"""

from __future__ import annotations

import math
from collections import deque

from .records import Alert, RolloutRecord
from .stats import RunningStats


class ComponentStat:
    """Rolling statistics for one named reward component over the window."""

    __slots__ = ("name", "_stats")

    def __init__(self, name: str) -> None:
        self.name = name
        self._stats = RunningStats()

    def add(self, value: float) -> None:
        self._stats.add(value)

    def remove(self, value: float) -> None:
        self._stats.remove(value)

    @property
    def rolling_mean(self) -> float:
        return self._stats.mean

    @property
    def rolling_std(self) -> float:
        return self._stats.std

    @property
    def count(self) -> int:
        return self._stats.n


class MetricStore:
    """Bounded history plus rolling statistics for a single watched function."""

    def __init__(
        self,
        name: str,
        window_size: int = 100,
        max_records: int = 10_000,
    ) -> None:
        self.name = name
        self.window_size = window_size
        self.records: deque[RolloutRecord] = deque(maxlen=max_records)
        self.alerts: list[Alert] = []
        self.step_counter: int = 0

        self._window: deque[RolloutRecord] = deque(maxlen=window_size)
        self._reward_stats = RunningStats()
        self.component_stats: dict[str, ComponentStat] = {}

        self.lifetime_min: float = math.inf
        self.lifetime_max: float = -math.inf

    # -- writes ----------------------------------------------------------------

    def append(self, record: RolloutRecord) -> None:
        """Record a new observation and update rolling statistics in O(1)."""
        if len(self._window) == self.window_size:
            # This append will evict the oldest from the window deque, so first
            # remove it from the running statistics to keep them in sync.
            self._evict(self._window[0])

        self._window.append(record)
        self.records.append(record)

        self._reward_stats.add(record.scalar_reward)
        for key, value in record.components.items():
            stat = self.component_stats.get(key)
            if stat is None:
                stat = self.component_stats[key] = ComponentStat(key)
            stat.add(value)

        if record.scalar_reward < self.lifetime_min:
            self.lifetime_min = record.scalar_reward
        if record.scalar_reward > self.lifetime_max:
            self.lifetime_max = record.scalar_reward

        self.step_counter += 1

    def _evict(self, record: RolloutRecord) -> None:
        self._reward_stats.remove(record.scalar_reward)
        for key, value in record.components.items():
            stat = self.component_stats.get(key)
            if stat is not None:
                stat.remove(value)

    def add_alert(self, alert: Alert) -> None:
        self.alerts.append(alert)

    # -- rolling reads (O(1)) --------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def window_count(self) -> int:
        return len(self._window)

    @property
    def rolling_mean(self) -> float:
        return self._reward_stats.mean

    @property
    def rolling_std(self) -> float:
        return self._reward_stats.std

    @property
    def rolling_variance(self) -> float:
        return self._reward_stats.variance

    @property
    def observed_min(self) -> float:
        return self.lifetime_min if self.records else 0.0

    @property
    def observed_max(self) -> float:
        return self.lifetime_max if self.records else 0.0

    # -- window reads (computed on demand) -------------------------------------

    def window_rewards(self) -> list[float]:
        return [r.scalar_reward for r in self._window]

    def percentile(self, q: float) -> float:
        """Linear-interpolated percentile of window rewards. ``q`` in [0, 100]."""
        data = sorted(self.window_rewards())
        if not data:
            return 0.0
        if len(data) == 1:
            return data[0]
        rank = (q / 100.0) * (len(data) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return data[low]
        return data[low] + (data[high] - data[low]) * (rank - low)

    def ceiling_rate(self, ceiling: float, tolerance: float = 0.99) -> float:
        """Fraction of window rewards at or above ``ceiling * tolerance``."""
        rewards = self.window_rewards()
        if not rewards:
            return 0.0
        threshold = ceiling * tolerance
        return sum(1 for r in rewards if r >= threshold) / len(rewards)

    def recent(self, n: int) -> list[RolloutRecord]:
        if n <= 0:
            return []
        return list(self.records)[-n:]

    def baseline_std(self, fraction: float = 0.2) -> float:
        """Reward std over the earliest ``fraction`` of recorded history.

        Detectors compare the recent window against this baseline to spot a
        variance collapse. Computed on demand from the kept history.
        """
        total = len(self.records)
        if total < 2:
            return 0.0
        count = max(2, int(total * fraction))
        stats = RunningStats()
        for record in list(self.records)[:count]:
            stats.add(record.scalar_reward)
        return stats.std
