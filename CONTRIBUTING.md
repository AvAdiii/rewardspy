# Contributing to rewardspy

Contributions are welcome. Bug reports, new detectors, integrations, and
documentation improvements are all useful.

## Development setup

```bash
git clone https://github.com/AvAdiii/rewardspy
cd rewardspy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks

```bash
pytest            # tests
ruff check .      # lint
mypy rewardspy    # type check
```

Please make sure tests and lint pass before opening a pull request, and add
tests for any new behavior.

## Project layout

```
rewardspy/
  wrapper.py        the watch decorator and Session
  records.py        RolloutRecord and Alert dataclasses
  store.py          MetricStore with rolling statistics
  stats.py          Welford online statistics
  detectors/        the six detectors plus the engine
  exporters/        JSONL, CSV, and Parquet
  tui/              the Textual dashboard
  integrations/     GRPO, TRL, W&B
  cli.py            the command line interface
examples/           runnable demos
docs/               guides and reference
tests/              the test suite
```

## Adding a detector

1. Create a module in `rewardspy/detectors/` with a class that subclasses
   `BaseDetector` and implements `check(store, record) -> DetectionResult`.
2. Give it a stable `name` and a short `label` (shown in the dashboard).
3. Register it in `DetectionEngine.__init__` and export it from
   `rewardspy/detectors/__init__.py`.
4. Add a test in `tests/test_detectors.py` that covers a healthy sequence (no
   alert) and a hacking sequence (alert fires), and document it in
   `docs/detectors.md`.

## Style

- Keep the wrapper a pure observer: never change a reward function's return
  value, and never let rewardspy break a training loop.
- Match the surrounding code's style. Lint with ruff.
