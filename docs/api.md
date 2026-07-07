# API reference

## `rewardspy.watch(fn=None, *, name=None, components=None, window_size=100, export_path=None, sensitivity="medium", max_reward=None, on_alert=None, detect=True)`

Wrap a reward function so every call is recorded. Usable directly
(`watch(fn)`), as a bare decorator (`@watch`), or parameterized
(`@watch(name=..., components=[...])`). The wrapped function returns exactly what
the original returned.

The returned wrapper exposes `.store` (a `MetricStore`) and `.engine` (a
`DetectionEngine`, or `None` when `detect=False`).

- `components`: names of reward components to track when the function returns a
  dict. The `"total"` key is the scalar reward; without one, the components are
  summed.
- `export_path`: stream every call to a JSONL file.
- `sensitivity`: `"low"`, `"medium"`, or `"high"`.
- `max_reward`: known reward ceiling, used by the ceiling detector.
- `on_alert`: callback invoked with each new `Alert`.

## `rewardspy.Session(name="session", window_size=100, export_path=None, hack_sensitivity="medium", max_reward=None, alert_callbacks=None, detect=True)`

A named group of watched reward functions sharing configuration. Use
`session.watch(...)` exactly like `watch`. Each watched function gets its own
store in `session.stores` and engine in `session.engines`.

## `rewardspy.log_eval(target, value, step=None)`

Log a held-out evaluation score against a watched reward function, for the
proxy-vs-held-out divergence detector. `target` may be the watched function, its
`MetricStore`, or its registered name. `value` is a score from a periodic eval on
the true objective the reward never optimizes. Returns the recorded `EvalPoint`.
Equivalent to `store.log_eval(value, step=None)` on the store directly.

```python
reward_fn = rewardspy.watch(my_reward_fn)
# ... every so often, evaluate on held-out data the reward never sees ...
rewardspy.log_eval(reward_fn, heldout_accuracy)
```

## `rewardspy.show(target=None, *, interval=0.5)`

Launch the live dashboard. `target` may be a `MetricStore`, a watched function,
an object exposing `.store`, or a registered name. With a single watched
function in the process, it is found automatically.

## `rewardspy.MetricStore(name, window_size=100, max_records=10_000)`

In-memory time-series with O(1) rolling statistics over a sliding window.
Key members: `rolling_mean`, `rolling_std`, `rolling_variance`, `percentile(q)`,
`ceiling_rate(ceiling)`, `observed_min`, `observed_max`, `recent(n)`,
`baseline_std(fraction)`, `records`, `alerts`, `component_stats`,
`log_eval(value, step=None)`, `eval_history`.

## Exporters

- `rewardspy.JSONLExporter(path, mode="a")`: streaming sink with `write`,
  `write_many`, `close`, and context-manager support.
- `rewardspy.read_jsonl(path) -> list[RolloutRecord]`
- `rewardspy.write_csv(records, path) -> Path`

## Detection

- `rewardspy.DetectionEngine(store, sensitivity="medium", max_reward=None, callbacks=None)`:
  runs all detectors via `process(record)`, exposes `latest` (per-detector
  `DetectionResult`) and `overall` (worst `Status`).
- `rewardspy.DetectionResult`, `rewardspy.Status`.

## Records

- `rewardspy.RolloutRecord`: `call_id`, `timestamp`, `step`, `scalar_reward`,
  `components`, `call_duration_ms`, `input_length`, `output_length`.
- `rewardspy.Alert`: `step`, `timestamp`, `detector`, `status`, `message`,
  `detail`, `severity`.
- `EvalPoint`: `step`, `timestamp`, `value`, `reward_mean`. One held-out eval
  score with the proxy reward snapshotted at log time.

## Integrations

```python
from rewardspy.integrations import GRPOSpy, watch_trl
from rewardspy.integrations import wandb as rspy_wandb
```

- `GRPOSpy(reward_fn, group_size=None, ...)`: GRPO-native wrapper. Score a group
  inside a `with spy.step(step=...):` block via `spy.reward(...)`. Exposes
  `groups`, `collapse_rate`, `mean_group_variance`, `store`, `engine`.
- `watch_trl(reward_func, ...)`: wrap a TRL-style batch reward function
  `(prompts, completions, **kw) -> list[float]`.
- `rspy_wandb.log_metrics(store, run=None, step=None)` and
  `rspy_wandb.alert_callback(run=None)`.

## CLI

```
rewardspy show    PATH [--follow] [--window N] [--sensitivity L] [--max-reward R]
rewardspy summary PATH [--last N] [--window N] [--sensitivity L]
rewardspy audit   PATH [--window N] [--sensitivity L]
rewardspy export  PATH -o OUT [--format csv|jsonl] [--last N]
rewardspy probe   module:function -p cases.json [--window N] [--sensitivity L]
```

`probe` runs a reward function over a JSON array of test cases offline (each item
is passed as keyword args, or `args`/`kwargs`, or positional args for a list) and
prints the same report as `summary`.
