"""The interceptor layer.

``watch`` wraps a reward function and records every call into a ``MetricStore``.
The wrapped function returns exactly what the original returned; rewardspy is a
pure observer and never changes the reward your training loop sees.

Three call styles are supported, in increasing order of control:

    reward_fn = rewardspy.watch(my_reward_fn)          # one-line

    @rewardspy.watch(name="math", components=["correctness", "format"])
    def reward_fn(response, answer): ...

    spy = rewardspy.Session(name="grpo_run_001", window_size=200)
    @spy.watch(components=["correctness", "format"])
    def reward_fn(response, ground_truth): ...
"""

from __future__ import annotations

import functools
import math
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from .detectors import DetectionEngine
from .exporters import JSONLExporter
from .records import Alert, EvalPoint, RolloutRecord
from .store import MetricStore

# Process-wide registry so the CLI and TUI can discover live stores by name.
_REGISTRY: dict[str, MetricStore] = {}
_WARNED: set[str] = set()


def registered_stores() -> dict[str, MetricStore]:
    """Return a snapshot of every store created in this process, keyed by name."""
    return dict(_REGISTRY)


def get_store(name: str) -> MetricStore | None:
    return _REGISTRY.get(name)


def _resolve_store(target: Any) -> MetricStore | None:
    """Find the store behind a watched function, a store, or a name."""
    if isinstance(target, MetricStore):
        return target
    store = getattr(target, "store", None) or getattr(target, "_rewardspy_store", None)
    if isinstance(store, MetricStore):
        return store
    if isinstance(target, str):
        return get_store(target)
    return None


def log_eval(target: Any, value: float, step: int | None = None) -> EvalPoint:
    """Log a held-out evaluation score against a watched reward function.

    ``target`` is the wrapped reward function, its store, or its name. The score
    should come from a periodic eval on the true objective that the reward never
    optimizes; the divergence detector then flags the reward climbing while this
    score stays flat or falls. See ``MetricStore.log_eval``.
    """
    store = _resolve_store(target)
    if store is None:
        raise ValueError(
            "log_eval() could not find a store for the given target; pass a "
            "watched reward function, a MetricStore, or a registered name."
        )
    return store.log_eval(value, step=step)


def _register(name: str, store: MetricStore) -> str:
    """Register a store under a unique name, suffixing duplicates."""
    key = name
    index = 1
    while key in _REGISTRY:
        index += 1
        key = f"{name}#{index}"
    store.name = key
    _REGISTRY[key] = store
    return key


def watch(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    components: list[str] | None = None,
    window_size: int = 100,
    export_path: str | Path | None = None,
    sensitivity: str = "medium",
    max_reward: float | None = None,
    on_alert: Callable[[Alert], None] | None = None,
    detect: bool = True,
) -> Callable[..., Any]:
    """Wrap a reward function so every call is recorded.

    Usable directly (``watch(fn)``), as a bare decorator (``@watch``), or as a
    parameterized decorator (``@watch(name=..., components=[...])``). When
    ``export_path`` is set, every call is also streamed to a JSONL file. By
    default the hack detectors run after each call; pass ``detect=False`` to
    record without detecting.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        fn_name = name or getattr(target, "__name__", "reward_fn")
        store = MetricStore(fn_name, window_size=window_size)
        _register(fn_name, store)
        exporter = JSONLExporter(export_path) if export_path is not None else None
        engine = (
            DetectionEngine(
                store,
                sensitivity=sensitivity,
                max_reward=max_reward,
                callbacks=[on_alert] if on_alert else None,
            )
            if detect
            else None
        )
        return _build_wrapper(target, components, store, exporter, engine)

    if fn is None:
        return decorate
    if not callable(fn):
        raise TypeError("watch() expects a callable or keyword arguments")
    return decorate(fn)


class Session:
    """A named group of watched reward functions sharing configuration.

    A session is useful when one run has several reward functions, or when you
    want to set the rolling window once and reuse it. Each watched function gets
    its own ``MetricStore``, available in ``session.stores``.
    """

    def __init__(
        self,
        name: str = "session",
        window_size: int = 100,
        export_path: str | Path | None = None,
        hack_sensitivity: str = "medium",
        max_reward: float | None = None,
        alert_callbacks: list[Callable[[Alert], None]] | None = None,
        detect: bool = True,
    ) -> None:
        self.name = name
        self.window_size = window_size
        self.export_path = export_path
        self.hack_sensitivity = hack_sensitivity
        self.max_reward = max_reward
        self.alert_callbacks = list(alert_callbacks or [])
        self.detect = detect
        self.stores: dict[str, MetricStore] = {}
        self.engines: dict[str, DetectionEngine] = {}
        self._used_paths: set[Path] = set()

    def watch(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        components: list[str] | None = None,
    ) -> Callable[..., Any]:
        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            fn_name = name or getattr(target, "__name__", "reward_fn")
            store = MetricStore(fn_name, window_size=self.window_size)
            key = _register(fn_name, store)
            self.stores[key] = store
            exporter = self._make_exporter(key)
            engine = None
            if self.detect:
                engine = DetectionEngine(
                    store,
                    sensitivity=self.hack_sensitivity,
                    max_reward=self.max_reward,
                    callbacks=self.alert_callbacks,
                )
                self.engines[key] = engine
            return _build_wrapper(target, components, store, exporter, engine)

        if fn is None:
            return decorate
        if not callable(fn):
            raise TypeError("Session.watch() expects a callable or keyword arguments")
        return decorate(fn)

    def _make_exporter(self, store_name: str) -> JSONLExporter | None:
        """Build a JSONL exporter for one watched function.

        The first function uses ``export_path`` as given. If more than one
        function shares the session, later ones get the store name inserted into
        the filename so they do not write to the same file.
        """
        if self.export_path is None:
            return None
        base = Path(self.export_path)
        path = base
        if path in self._used_paths:
            path = base.with_name(f"{base.stem}.{store_name}{base.suffix}")
        self._used_paths.add(path)
        return JSONLExporter(path)


def _build_wrapper(
    fn: Callable[..., Any],
    components: list[str] | None,
    store: MetricStore,
    exporter: JSONLExporter | None = None,
    engine: DetectionEngine | None = None,
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)  # call the original; exceptions propagate
        duration_ms = (time.perf_counter() - start) * 1000.0

        scalar, comps = _parse_result(result, components)
        if scalar is None or not math.isfinite(scalar):
            _warn_once(store.name)
            return result

        record = RolloutRecord(
            call_id=uuid4().hex,
            timestamp=time.time(),
            step=store.step_counter,
            scalar_reward=scalar,
            components=comps,
            call_duration_ms=duration_ms,
            input_length=_input_length(args, kwargs),
            output_length=_response_length(args, kwargs),
        )
        store.append(record)
        if exporter is not None:
            exporter.write(record)
        if engine is not None:
            engine.process(record)
        return result

    wrapper._rewardspy_store = store  # type: ignore[attr-defined]
    wrapper.store = store  # type: ignore[attr-defined]
    wrapper.engine = engine  # type: ignore[attr-defined]
    return wrapper


def _parse_result(
    result: Any, components: list[str] | None
) -> tuple[float | None, dict[str, float]]:
    """Reduce a reward function's return to a scalar plus tracked components.

    A dict return is treated as named components. Its ``"total"`` entry is the
    scalar reward; without one, the declared components (or all entries) are
    summed. Anything else is coerced to a float.
    """
    if isinstance(result, Mapping):
        comps: dict[str, float] = {}
        for key, value in result.items():
            try:
                comps[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if "total" in comps:
            scalar: float | None = comps["total"]
        else:
            keys = components if components else list(comps)
            scalar = sum(comps.get(k, 0.0) for k in keys)
        return scalar, comps

    try:
        return float(result), {}
    except (TypeError, ValueError):
        return None, {}


def _estimate_length(obj: Any) -> int:
    """Best-effort length of a value without holding on to large payloads."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return int(math.prod(shape))
        except TypeError:
            pass
    if isinstance(obj, (bool, int, float)):
        return 0
    try:
        return len(obj)
    except TypeError:
        return 0


def _response_length(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Length of the response being scored, taken as the first argument."""
    if args:
        return _estimate_length(args[0])
    if kwargs:
        return _estimate_length(next(iter(kwargs.values())))
    return 0


def _input_length(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    total = sum(_estimate_length(a) for a in args)
    total += sum(_estimate_length(v) for v in kwargs.values())
    return total


def _warn_once(name: str) -> None:
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"rewardspy could not read a numeric reward from '{name}'. "
        "Return a number, or a dict with a 'total' key. These calls are passed "
        "through unrecorded.",
        RuntimeWarning,
        stacklevel=3,
    )
