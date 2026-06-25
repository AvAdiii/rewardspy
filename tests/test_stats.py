import random
import statistics

import pytest

from rewardspy.stats import RunningStats


def test_add_matches_reference():
    values = [random.gauss(0.5, 0.2) for _ in range(500)]
    stats = RunningStats()
    for v in values:
        stats.add(v)
    assert stats.n == len(values)
    assert stats.mean == pytest.approx(statistics.fmean(values))
    assert stats.variance == pytest.approx(statistics.pvariance(values))
    assert stats.std == pytest.approx(statistics.pstdev(values))


def test_remove_keeps_sliding_window_correct():
    """Add then evict so only the last `window` values remain, and compare to a
    fresh recompute over exactly those values."""
    window = 50
    values = [random.gauss(0.0, 1.0) for _ in range(300)]
    stats = RunningStats()
    held: list[float] = []
    for v in values:
        if len(held) == window:
            stats.remove(held.pop(0))
        stats.add(v)
        held.append(v)

    assert stats.n == window
    assert stats.mean == pytest.approx(statistics.fmean(held))
    assert stats.variance == pytest.approx(statistics.pvariance(held))


def test_remove_to_empty_resets():
    stats = RunningStats()
    stats.add(3.0)
    stats.remove(3.0)
    assert stats.n == 0
    assert stats.mean == 0.0
    assert stats.variance == 0.0
    assert stats.std == 0.0


def test_empty_is_zero():
    stats = RunningStats()
    assert stats.mean == 0.0
    assert stats.variance == 0.0
    assert stats.std == 0.0
