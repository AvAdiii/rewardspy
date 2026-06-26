"""TRL integration.

TRL's trainers call reward functions in batch form: given lists of prompts and
completions, return a list of float rewards. ``watch_trl`` wraps such a function
so each completion's reward is recorded individually, while returning the exact
same list TRL expects. Your trainer code does not change.

    from rewardspy.integrations import watch_trl

    reward_func = watch_trl(my_reward_func, name="math")
    trainer = GRPOTrainer(..., reward_funcs=[reward_func])
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from ..detectors import DetectionEngine
from ..exporters import JSONLExporter
from ..records import RolloutRecord
from ..store import MetricStore
from ..wrapper import _estimate_length, _register


def watch_trl(
    reward_func: Callable[..., Sequence[float]] | None = None,
    *,
    name: str | None = None,
    window_size: int = 100,
    sensitivity: str = "medium",
    export_path: str | None = None,
    max_reward: float | None = None,
    detect: bool = True,
) -> Callable[..., Sequence[float]]:
    """Wrap a TRL-style batch reward function ``(prompts, completions, **kw) -> list``."""

    def decorate(fn: Callable[..., Sequence[float]]) -> Callable[..., Sequence[float]]:
        fn_name = name or getattr(fn, "__name__", "trl_reward")
        store = MetricStore(fn_name, window_size=window_size)
        _register(fn_name, store)
        exporter = JSONLExporter(export_path) if export_path is not None else None
        engine = (
            DetectionEngine(store, sensitivity=sensitivity, max_reward=max_reward)
            if detect
            else None
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Sequence[float]:
            rewards = fn(*args, **kwargs)
            completions = _find_completions(args, kwargs)
            _record_batch(rewards, completions, store, exporter, engine)
            return rewards

        wrapper._rewardspy_store = store  # type: ignore[attr-defined]
        wrapper.store = store  # type: ignore[attr-defined]
        wrapper.engine = engine  # type: ignore[attr-defined]
        return wrapper

    if reward_func is None:
        return decorate
    return decorate(reward_func)


def _find_completions(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Sequence[Any] | None:
    if "completions" in kwargs:
        return kwargs["completions"]
    if len(args) >= 2:
        return args[1]
    if len(args) == 1:
        return args[0]
    return None


def _completion_length(completion: Any) -> int:
    if isinstance(completion, str):
        return len(completion)
    if isinstance(completion, list):  # chat-format messages
        total = 0
        for message in completion:
            if isinstance(message, dict):
                total += len(str(message.get("content", "")))
            else:
                total += len(str(message))
        return total
    return _estimate_length(completion)


def _record_batch(
    rewards: Sequence[float],
    completions: Sequence[Any] | None,
    store: MetricStore,
    exporter: JSONLExporter | None,
    engine: DetectionEngine | None,
) -> None:
    import time

    for index, raw in enumerate(rewards):
        try:
            scalar = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(scalar):
            continue
        completion = None
        if completions is not None and index < len(completions):
            completion = completions[index]
        record = RolloutRecord(
            call_id=uuid4().hex,
            timestamp=time.time(),
            step=store.step_counter,
            scalar_reward=scalar,
            output_length=_completion_length(completion) if completion is not None else 0,
        )
        store.append(record)
        if exporter is not None:
            exporter.write(record)
        if engine is not None:
            engine.process(record)
