"""GRPO-native integration.

GRPO scores a group of responses per prompt and normalizes within the group to
get advantages. That makes the group itself a unit worth watching: if the reward
variance inside a group collapses to zero, every response scored the same and
there is no learning signal left, often a sign the policy has converged onto a
single (possibly hacked) strategy.

``GRPOSpy`` wraps your reward function, records every call like ``watch`` does,
and additionally tracks per-group statistics. Group collapse is surfaced as an
alert on the same store the dashboard and CLI read.

    spy = GRPOSpy(reward_fn=my_reward, group_size=8)

    for batch in loader:
        with spy.step(step=global_step):
            for response in batch.responses:
                spy.reward(response, batch.answer)
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from ..records import Alert
from ..wrapper import _parse_result, watch


@dataclass(slots=True)
class GroupRecord:
    """Summary statistics for one GRPO group."""

    step: int
    size: int
    mean: float
    std: float
    variance: float
    reward_min: float
    reward_max: float
    best_minus_mean: float
    all_same: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarize_group(step: int, rewards: list[float], collapse_threshold: float) -> GroupRecord:
    mean = statistics.fmean(rewards)
    variance = statistics.pvariance(rewards) if len(rewards) > 1 else 0.0
    std = variance**0.5
    reward_max = max(rewards)
    return GroupRecord(
        step=step,
        size=len(rewards),
        mean=mean,
        std=std,
        variance=variance,
        reward_min=min(rewards),
        reward_max=reward_max,
        best_minus_mean=reward_max - mean,
        all_same=std < collapse_threshold,
    )


class GRPOSpy:
    def __init__(
        self,
        reward_fn: Callable[..., Any],
        group_size: int | None = None,
        name: str = "grpo",
        window_size: int = 100,
        sensitivity: str = "medium",
        export_path: str | None = None,
        max_reward: float | None = None,
        collapse_threshold: float = 1e-6,
        group_history: int = 500,
        collapse_alert_rate: float = 0.5,
        min_groups: int = 20,
    ) -> None:
        self.reward_fn = watch(
            reward_fn,
            name=name,
            window_size=window_size,
            sensitivity=sensitivity,
            export_path=export_path,
            max_reward=max_reward,
        )
        self.store = self.reward_fn.store
        self.engine = self.reward_fn.engine
        self.group_size = group_size
        self.collapse_threshold = collapse_threshold
        self.collapse_alert_rate = collapse_alert_rate
        self.min_groups = min_groups
        self.groups: deque[GroupRecord] = deque(maxlen=group_history)

        self._buffer: list[float] = []
        self._step = 0
        self._collapse_flagged = False

    def reward(self, *args: Any, **kwargs: Any) -> Any:
        """Score one response. Identical return to the wrapped reward function."""
        result = self.reward_fn(*args, **kwargs)
        scalar, _ = _parse_result(result, None)
        if scalar is not None and scalar == scalar:  # finite, not NaN
            self._buffer.append(float(scalar))
        return result

    def score_group(self, items: Iterable[Any]) -> list[Any]:
        """Score an iterable of arg-tuples as one group."""
        results = []
        for item in items:
            call = tuple(item) if isinstance(item, (tuple, list)) else (item,)
            results.append(self.reward(*call))
        return results

    @contextmanager
    def step(self, step: int | None = None):
        """Group every reward scored inside the block into one GRPO group."""
        self._buffer = []
        try:
            yield self
        finally:
            self.close_group(step)

    def close_group(self, step: int | None = None) -> GroupRecord | None:
        rewards = self._buffer
        self._buffer = []
        if not rewards:
            return None
        self._step = step if step is not None else self._step + 1
        record = _summarize_group(self._step, rewards, self.collapse_threshold)
        self.groups.append(record)
        self._maybe_alert()
        return record

    @property
    def collapse_rate(self) -> float:
        """Fraction of tracked groups whose reward variance has collapsed."""
        if not self.groups:
            return 0.0
        return sum(1 for g in self.groups if g.all_same) / len(self.groups)

    @property
    def mean_group_variance(self) -> float:
        if not self.groups:
            return 0.0
        return statistics.fmean(g.variance for g in self.groups)

    def _maybe_alert(self) -> None:
        if len(self.groups) < self.min_groups:
            return
        rate = self.collapse_rate
        if rate > self.collapse_alert_rate and not self._collapse_flagged:
            self._collapse_flagged = True
            self.store.add_alert(
                Alert(
                    step=self._step,
                    timestamp=time.time(),
                    detector="grpo_group",
                    status="ALERT",
                    message=f"{rate:.0%} of recent groups have zero reward variance.",
                    detail="No learning signal: responses within a group score identically.",
                    severity="HIGH",
                )
            )
        elif rate <= self.collapse_alert_rate:
            self._collapse_flagged = False
