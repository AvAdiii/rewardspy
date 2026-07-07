import rewardspy
from rewardspy.detectors import DetectionEngine, ProxyEvalDivergenceDetector, Status
from rewardspy.detectors.divergence import _fractional_trend
from rewardspy.records import RolloutRecord
from rewardspy.store import MetricStore


def make_record(step, reward):
    return RolloutRecord(
        call_id=f"c{step}",
        timestamp=float(step),
        step=step,
        scalar_reward=reward,
    )


def test_fractional_trend_directions():
    assert _fractional_trend([0.0, 1.0, 2.0, 3.0]) > 0.9
    assert _fractional_trend([3.0, 2.0, 1.0, 0.0]) < -0.9
    # Noise symmetric around a flat mean has a near-zero fitted slope.
    assert abs(_fractional_trend([1.0, 0.9, 1.1, 1.1, 0.9, 1.0])) < 1e-9


def test_not_applicable_without_eval():
    store = MetricStore("d")
    for i in range(50):
        store.append(make_record(i, 0.5))
    result = ProxyEvalDivergenceDetector().check(store)
    assert result.status == Status.NOT_APPLICABLE


def test_insufficient_eval_data():
    store = MetricStore("d")
    store.append(make_record(0, 0.5))
    store.log_eval(0.5)
    store.log_eval(0.5)
    assert ProxyEvalDivergenceDetector().check(store).status == Status.INSUFFICIENT_DATA


def test_alert_on_reward_up_eval_down():
    store = MetricStore("d")
    detector = ProxyEvalDivergenceDetector()
    for i in range(6):
        # Proxy reward climbs, the held-out score falls: classic hacking.
        for _ in range(5):
            store.append(make_record(store.step_counter, 0.1 * i))
        store.log_eval(1.0 - 0.1 * i)
    result = detector.check(store)
    assert result.status == Status.ALERT
    assert "held-out" in result.message


def test_warning_on_reward_up_eval_flat():
    store = MetricStore("d")
    detector = ProxyEvalDivergenceDetector()
    for i in range(6):
        for _ in range(5):
            store.append(make_record(store.step_counter, 0.1 * i))
        store.log_eval(0.7)  # eval flat while reward climbs
    result = detector.check(store)
    assert result.status == Status.WARNING


def test_ok_when_both_climb_together():
    store = MetricStore("d")
    detector = ProxyEvalDivergenceDetector()
    for i in range(6):
        for _ in range(5):
            store.append(make_record(store.step_counter, 0.1 * i))
        store.log_eval(0.1 * i)  # eval tracks the reward up
    assert detector.check(store).status == Status.OK


def test_engine_registers_divergence_and_raises_alert():
    store = MetricStore("d")
    engine = DetectionEngine(store)
    assert "divergence" in engine.latest
    for i in range(6):
        for _ in range(5):
            rec = make_record(store.step_counter, 0.1 * i)
            store.append(rec)
            engine.process(rec)
        store.log_eval(1.0 - 0.1 * i)  # log_eval triggers engine.process()
    alerts = [a for a in store.alerts if a.detector == "divergence"]
    assert alerts
    assert engine.latest["divergence"].status == Status.ALERT


def test_public_log_eval_resolves_watched_fn():
    @rewardspy.watch(name="le_fn")
    def reward(x):
        return x

    for _ in range(30):
        reward(0.5)
    point = rewardspy.log_eval(reward, 0.8)
    assert point.value == 0.8
    assert reward.store.eval_history[-1].value == 0.8
    # By name and by store, too.
    rewardspy.log_eval("le_fn", 0.9)
    rewardspy.log_eval(reward.store, 0.7)
    assert len(reward.store.eval_history) == 3


def test_public_log_eval_bad_target():
    import pytest

    with pytest.raises(ValueError):
        rewardspy.log_eval(object(), 1.0)
