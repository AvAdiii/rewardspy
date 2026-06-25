import statistics

import pytest

from rewardspy.records import Alert, RolloutRecord
from rewardspy.store import MetricStore


def make_record(step: int, reward: float, components: dict[str, float] | None = None):
    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
        components=components or {},
        output_length=10,
    )


def test_append_tracks_count_and_step():
    store = MetricStore("t")
    for i in range(10):
        store.append(make_record(i, float(i)))
    assert store.count == 10
    assert store.step_counter == 10


def test_rolling_stats_match_window_recompute():
    window = 100
    store = MetricStore("t", window_size=window)
    rewards = [(i % 7) * 0.13 for i in range(500)]
    for i, r in enumerate(rewards):
        store.append(make_record(i, r))

    last = rewards[-window:]
    assert store.rolling_mean == pytest.approx(statistics.fmean(last))
    assert store.rolling_variance == pytest.approx(statistics.pvariance(last))
    assert store.window_count == window


def test_history_is_bounded_but_window_correct():
    store = MetricStore("t", window_size=10, max_records=50)
    for i in range(200):
        store.append(make_record(i, float(i)))
    assert store.count == 50  # bounded history
    assert store.window_count == 10
    assert store.rolling_mean == pytest.approx(statistics.fmean(range(190, 200)))


def test_component_stats_track_per_key():
    store = MetricStore("t", window_size=100)
    for i in range(100):
        store.append(
            make_record(i, 1.0, {"correctness": 0.8, "format": 0.1})
        )
    assert store.component_stats["correctness"].rolling_mean == pytest.approx(0.8)
    assert store.component_stats["format"].rolling_mean == pytest.approx(0.1)


def test_percentiles_and_bounds():
    store = MetricStore("t", window_size=100)
    for i in range(101):
        store.append(make_record(i, float(i)))  # 1..100 in window
    # window holds 1..100
    assert store.percentile(0) == pytest.approx(1.0)
    assert store.percentile(100) == pytest.approx(100.0)
    assert store.percentile(50) == pytest.approx(50.5)
    assert store.observed_min == pytest.approx(0.0)
    assert store.observed_max == pytest.approx(100.0)


def test_ceiling_rate():
    store = MetricStore("t", window_size=10)
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    for i, r in enumerate(rewards):
        store.append(make_record(i, r))
    assert store.ceiling_rate(ceiling=1.0) == pytest.approx(0.8)


def test_baseline_std_uses_earliest_history():
    store = MetricStore("t", window_size=10)
    # First 20 are noisy, rest are constant.
    for i in range(20):
        store.append(make_record(i, float(i % 5)))
    for i in range(20, 100):
        store.append(make_record(i, 1.0))
    baseline = store.baseline_std(fraction=0.2)
    assert baseline > 0.0  # early history had spread


def test_alerts_accumulate():
    store = MetricStore("t")
    store.add_alert(Alert(step=1, timestamp=0.0, detector="x", status="WARNING", message="hi"))
    assert len(store.alerts) == 1
    assert store.alerts[0].detector == "x"


def test_empty_store_reads_are_safe():
    store = MetricStore("t")
    assert store.rolling_mean == 0.0
    assert store.percentile(50) == 0.0
    assert store.observed_max == 0.0
    assert store.ceiling_rate(1.0) == 0.0
    assert store.recent(5) == []
