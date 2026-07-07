"""Runs every detector against a store and turns findings into alerts.

The engine holds one instance of each detector, runs them after each new record,
and records an ``Alert`` when a detector's status first becomes actionable. It
deduplicates: an ongoing warning fires once, not once per step, and re-fires only
after the condition clears and returns.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from ..records import Alert, RolloutRecord
from ..store import MetricStore
from .base import BaseDetector, DetectionResult, Status
from .ceiling import CeilingRateDetector
from .components import ComponentDominanceDetector
from .divergence import ProxyEvalDivergenceDetector
from .length import LengthDriftDetector
from .slope import RewardSlopeChangeDetector
from .variance import VarianceCollapseDetector

_SEVERITY_RANK = {Status.OK: 1, Status.WARNING: 2, Status.ALERT: 3}


class DetectionEngine:
    def __init__(
        self,
        store: MetricStore,
        sensitivity: str = "medium",
        max_reward: float | None = None,
        callbacks: Iterable[Callable[[Alert], None]] | None = None,
    ) -> None:
        self.store = store
        self.sensitivity = sensitivity
        self.callbacks: list[Callable[[Alert], None]] = list(callbacks or [])
        self.detectors: list[BaseDetector] = [
            VarianceCollapseDetector(sensitivity),
            RewardSlopeChangeDetector(sensitivity),
            ComponentDominanceDetector(sensitivity),
            CeilingRateDetector(sensitivity, max_reward=max_reward),
            LengthDriftDetector(sensitivity),
            ProxyEvalDivergenceDetector(sensitivity),
        ]
        self.latest: dict[str, DetectionResult] = {
            d.name: DetectionResult.insufficient_data() for d in self.detectors
        }
        self._last_actionable: dict[str, Status | None] = {d.name: None for d in self.detectors}
        # Let consumers (the dashboard) find the engine from the store.
        store.engine = self

    def process(self, record: RolloutRecord | None = None) -> list[Alert]:
        """Run all detectors once and return any newly raised alerts."""
        new_alerts: list[Alert] = []
        for detector in self.detectors:
            result = detector.check(self.store, record)
            self.latest[detector.name] = result

            if result.actionable:
                if self._last_actionable[detector.name] != result.status:
                    alert = self._raise(detector, result)
                    new_alerts.append(alert)
                self._last_actionable[detector.name] = result.status
            elif result.status == Status.OK:
                self._last_actionable[detector.name] = None

        return new_alerts

    def _raise(self, detector: BaseDetector, result: DetectionResult) -> Alert:
        alert = Alert(
            step=self.store.step_counter,
            timestamp=time.time(),
            detector=detector.name,
            status=result.status.value,
            message=result.message,
            detail=result.detail,
            severity=result.severity,
        )
        self.store.add_alert(alert)
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception:
                # A misbehaving callback must never break the training loop.
                pass
        return alert

    @property
    def overall(self) -> Status:
        """Worst current status across detectors, for a single headline state."""
        worst = Status.OK
        worst_rank = _SEVERITY_RANK[Status.OK]
        for result in self.latest.values():
            rank = _SEVERITY_RANK.get(result.status, 0)
            if rank > worst_rank:
                worst_rank = rank
                worst = result.status
        return worst
