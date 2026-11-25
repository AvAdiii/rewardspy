# Quickstart

## Install

```bash
pip install rewardspy
```

Optional extras:

```bash
pip install "rewardspy[trl]"     # TRL integration
pip install "rewardspy[wandb]"   # Weights & Biases integration
```

## Wrap a reward function

The whole point is that integration is one line and your reward function is
never modified. rewardspy only observes.

```python
import rewardspy

reward_fn = rewardspy.watch(my_reward_fn)
r = reward_fn(response, ground_truth)   # identical to the original
```

## Track components

Return a dict to track parts of the reward separately. The `"total"` key is the
scalar reward; without one, the declared components are summed.

```python
@rewardspy.watch(name="math", components=["correctness", "format", "length"])
def reward(response, answer):
    correctness = check_answer(response, answer)
    format_ok = has_think_tags(response)
    length_pen = -max(0, len(response) - 2000) / 1000
    return {
        "correctness": correctness,
        "format": format_ok * 0.1,
        "length": length_pen * 0.05,
        "total": correctness + format_ok * 0.1 + length_pen * 0.05,
    }
```

## Watch it live

Stream to a log and open the dashboard from another terminal:

```python
reward = rewardspy.watch(my_reward_fn, export_path="logs/run.jsonl")
```

```bash
rewardspy show logs/run.jsonl --follow
```

Or, in the same process, launch the dashboard directly:

```python
rewardspy.show(reward)
```

## From the command line

```bash
rewardspy summary logs/run.jsonl --last 500   # text summary + verdict
rewardspy audit   logs/run.jsonl              # verdict, exits non-zero if flagged
rewardspy export  logs/run.jsonl -o out.csv   # convert to CSV
```

`audit` is handy in CI: it exits with a non-zero status when a hacking signature
is detected, so a training job can stop or warn automatically.

## Sessions

A `Session` shares configuration across reward functions and adds export and
alert callbacks:

```python
spy = rewardspy.Session(
    name="grpo_run_001",
    window_size=200,
    hack_sensitivity="medium",          # low / medium / high
    export_path="logs/rewards.jsonl",
    alert_callbacks=[my_slack_notifier],
)

@spy.watch(components=["correctness", "format"])
def reward(response, ground_truth):
    ...
```

See [detectors.md](detectors.md) for how each check works and
[api.md](api.md) for the full surface.
