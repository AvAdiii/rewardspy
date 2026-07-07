"""Proxy vs held-out objective divergence.

Every other detector reads the reward's own distribution, so a clean exploit
that keeps that distribution looking healthy is invisible to them. The one
signal that is not fooled is the gap between the proxy reward climbing and the
true objective it stands in for failing to move.

rewardspy cannot see the true objective on its own, so you feed it in with
``store.log_eval(score)`` (or ``rewardspy.log_eval(reward_fn, score)``) from a
periodic held-out eval that the reward never touches. This detector watches for
the proxy trending up while that held-out score stays flat or falls, which is
reward overoptimization by definition. With no eval logged it stays
NOT_APPLICABLE and costs nothing.
"""

from __future__ import annotations

from ..records import RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult


def _fractional_trend(ys: list[float]) -> float:
    """Least-squares trend over ``ys``, as a fraction of the series scale.

    Returns the fitted change across the whole window divided by the series
    span, so a clean rise returns roughly +1 and pure noise around a flat mean
    returns near 0 (the slope of noise cancels). Scale-free, so proxy reward and
    an eval metric on different units are comparable.
    """
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in range(n))
    if denom == 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in enumerate(ys)) / denom
    total = slope * (n - 1)
    span = max(ys) - min(ys)
    scale = span if span > 1e-9 else (abs(my) if abs(my) > 1e-9 else 1.0)
    return total / scale


class ProxyEvalDivergenceDetector(BaseDetector):
    name = "divergence"
    label = "Divergence"

    # Fractional-trend band that counts as a real move, per sensitivity. A
    # smaller band flags subtler divergence at the cost of more false positives.
    PARAMS = {"low": 0.25, "medium": 0.15, "high": 0.08}

    MIN_EVALS = 4
    WINDOW = 10

    def __init__(self, sensitivity: str = "medium") -> None:
        super().__init__(sensitivity)
        self.band = self.PARAMS[sensitivity]

    def check(
        self, store: MetricStore, record: RolloutRecord | None = None
    ) -> DetectionResult:
        points = store.eval_history
        if not points:
            # No held-out eval was ever logged; nothing to compare against.
            return DetectionResult.not_applicable()
        if len(points) < self.MIN_EVALS:
            return DetectionResult.insufficient_data()

        recent = list(points)[-self.WINDOW :]
        proxy_trend = _fractional_trend([p.reward_mean for p in recent])
        eval_trend = _fractional_trend([p.value for p in recent])
        n = len(recent)

        if proxy_trend > self.band and eval_trend < -self.band:
            return DetectionResult.alert(
                "Proxy reward is climbing while the held-out objective falls.",
                f"Over the last {n} evals the reward trended {proxy_trend:+.0%} but "
                f"the held-out score trended {eval_trend:+.0%}. This is textbook "
                "reward overoptimization: the model is gaming the proxy.",
                severity="HIGH",
            )
        if proxy_trend > self.band and abs(eval_trend) <= self.band:
            return DetectionResult.warning(
                "Proxy reward is climbing while the held-out objective is flat.",
                f"Over the last {n} evals the reward trended {proxy_trend:+.0%} but "
                f"the held-out score barely moved ({eval_trend:+.0%}). The gain may "
                "not be real; watch for divergence.",
                severity="MEDIUM",
            )
        return DetectionResult.ok()
