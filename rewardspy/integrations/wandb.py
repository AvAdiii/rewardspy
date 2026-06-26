"""Weights & Biases integration.

Log rewardspy's rolling metrics and detector health alongside your usual
training curves, so reward-hacking signals sit right next to the reward curve in
your W&B dashboard.

    import wandb
    from rewardspy.integrations import wandb as rspy_wandb

    wandb.init(...)
    # in your loop, after some reward calls:
    rspy_wandb.log_metrics(reward.store)

``run`` defaults to the active ``wandb.run``; pass one explicitly if you prefer.
"""

from __future__ import annotations

from typing import Any

from ..records import Alert
from ..store import MetricStore

# Numeric health so it charts cleanly in W&B.
_HEALTH = {"OK": 0, "WARNING": 1, "ALERT": 2}


def _resolve_run(run: Any) -> Any:
    if run is not None:
        return run
    import wandb

    if wandb.run is None:
        raise RuntimeError("no active wandb run; call wandb.init() or pass run=")
    return wandb.run


def log_metrics(
    store: MetricStore,
    run: Any = None,
    step: int | None = None,
    prefix: str = "rewardspy",
) -> dict[str, Any]:
    """Log current rolling metrics and detector health. Returns the logged dict."""
    target = _resolve_run(run)
    ceiling = store.ceiling_rate(store.observed_max) if store.observed_max > 0 else 0.0
    data: dict[str, Any] = {
        f"{prefix}/reward_mean": store.rolling_mean,
        f"{prefix}/reward_std": store.rolling_std,
        f"{prefix}/at_ceiling": ceiling,
        f"{prefix}/alert_count": len(store.alerts),
    }
    engine = store.engine
    if engine is not None:
        for name, result in engine.latest.items():
            data[f"{prefix}/health/{name}"] = _HEALTH.get(result.status.value, 0)
        data[f"{prefix}/health/overall"] = _HEALTH.get(engine.overall.value, 0)

    target.log(data, step=step)
    return data


def alert_callback(run: Any = None, prefix: str = "rewardspy"):
    """Return an alert callback that logs each new alert to W&B."""

    def callback(alert: Alert) -> None:
        target = _resolve_run(run)
        target.log(
            {
                f"{prefix}/alert": _HEALTH.get(alert.status, 1),
                f"{prefix}/alert_detector": alert.detector,
                f"{prefix}/alert_message": alert.message,
            }
        )

    return callback
