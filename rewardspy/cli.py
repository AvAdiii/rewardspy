"""Command line interface.

The CLI works on the JSONL logs written by the exporter, so it runs in a
separate process from training. That is how you attach to a live run: training
streams to a ``.jsonl`` file, and ``rewardspy show run.jsonl --follow`` tails it.

    rewardspy show    run.jsonl --follow      launch the live dashboard
    rewardspy summary run.jsonl --last 500    print a text summary
    rewardspy audit   run.jsonl               verdict + exit code for CI
    rewardspy export  run.jsonl -o out.csv    convert a log to CSV
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .detectors import DetectionEngine
from .exporters import JSONLExporter, read_jsonl, write_csv, write_parquet
from .records import RolloutRecord
from .store import MetricStore
from .tui.render import diagnosis
from .wrapper import watch

console = Console()


def _load_store(
    path: str | Path,
    window: int,
    sensitivity: str,
    last: int | None = None,
    max_reward: float | None = None,
) -> tuple[MetricStore, DetectionEngine]:
    """Replay a JSONL log into a fresh store with detection applied."""
    records = read_jsonl(path)
    if last is not None:
        records = records[-last:]
    store = MetricStore(Path(path).stem, window_size=window)
    engine = DetectionEngine(store, sensitivity=sensitivity, max_reward=max_reward)
    for record in records:
        store.append(record)
        engine.process(record)
    return store, engine


def _tail(path: Path, store: MetricStore, engine: DetectionEngine, stop: threading.Event) -> None:
    """Stream a growing JSONL file into the store until told to stop."""
    with open(path, encoding="utf-8") as fh:
        while not stop.is_set():
            position = fh.tell()
            line = fh.readline()
            if not line or not line.endswith("\n"):
                # No data yet, or a partial line mid-write: rewind and wait.
                fh.seek(position)
                time.sleep(0.15)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = RolloutRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            store.append(record)
            engine.process(record)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="rewardspy")
def main() -> None:
    """rewardspy: debug and visualize RL reward functions."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-f", "--follow", is_flag=True, help="Tail the log for a live run.")
@click.option("--window", default=100, show_default=True, help="Rolling window size.")
@click.option("--sensitivity", default="medium", type=click.Choice(["low", "medium", "high"]))
@click.option("--max-reward", type=float, default=None, help="Known reward ceiling.")
@click.option("--interval", default=0.5, show_default=True, help="Refresh seconds.")
def show(
    path: str,
    follow: bool,
    window: int,
    sensitivity: str,
    max_reward: float | None,
    interval: float,
) -> None:
    """Launch the dashboard for a JSONL log."""
    from .tui.app import RewardSpyApp

    if follow:
        store = MetricStore(Path(path).stem, window_size=window)
        engine = DetectionEngine(store, sensitivity=sensitivity, max_reward=max_reward)
        stop = threading.Event()
        thread = threading.Thread(target=_tail, args=(Path(path), store, engine, stop), daemon=True)
        thread.start()
        try:
            RewardSpyApp(store, interval=interval).run()
        finally:
            stop.set()
    else:
        store, _ = _load_store(path, window, sensitivity, max_reward=max_reward)
        RewardSpyApp(store, interval=interval).run()


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--last", default=None, type=int, help="Only the last N records.")
@click.option("--window", default=100, show_default=True)
@click.option("--sensitivity", default="medium", type=click.Choice(["low", "medium", "high"]))
@click.option("--max-reward", type=float, default=None)
def summary(
    path: str, last: int | None, window: int, sensitivity: str, max_reward: float | None
) -> None:
    """Print a text summary of a JSONL log."""
    store, engine = _load_store(path, window, sensitivity, last=last, max_reward=max_reward)
    if store.count == 0:
        console.print("[yellow]No records found.[/yellow]")
        return
    _print_report(store, engine)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--window", default=100, show_default=True)
@click.option("--sensitivity", default="medium", type=click.Choice(["low", "medium", "high"]))
@click.option("--max-reward", type=float, default=None)
def audit(path: str, window: int, sensitivity: str, max_reward: float | None) -> None:
    """Audit a log for reward hacking. Exits non-zero if anything is flagged."""
    store, engine = _load_store(path, window, sensitivity, max_reward=max_reward)
    if store.count == 0:
        console.print("[yellow]No records found.[/yellow]")
        raise SystemExit(0)

    console.print(diagnosis(store))
    if store.alerts:
        console.print()
        for alert in store.alerts:
            color = {"ALERT": "red", "WARNING": "yellow"}.get(alert.status, "white")
            console.print(f"[{color}]{alert.status}[/{color}] step {alert.step}: {alert.message}")

    overall = engine.overall.value
    raise SystemExit(0 if overall == "OK" else 1)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "jsonl", "parquet"]),
    default="csv",
    show_default=True,
)
@click.option("--last", default=None, type=int, help="Only the last N records.")
def export(path: str, output: str, fmt: str, last: int | None) -> None:
    """Convert a JSONL log to CSV, Parquet, or a trimmed JSONL file."""
    records = read_jsonl(path)
    if last is not None:
        records = records[-last:]

    if fmt == "csv":
        write_csv(records, output)
    elif fmt == "parquet":
        write_parquet(records, output)
    else:
        with JSONLExporter(output, mode="w") as exporter:
            exporter.write_many(records)

    console.print(f"Wrote {len(records):,} records to [bold]{output}[/bold]")


@main.command()
@click.argument("target")
@click.option("-p", "--prompts", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--window", default=100, show_default=True)
@click.option("--sensitivity", default="medium", type=click.Choice(["low", "medium", "high"]))
@click.option("--max-reward", type=float, default=None)
def probe(
    target: str, prompts: str, window: int, sensitivity: str, max_reward: float | None
) -> None:
    """Run a reward function over test cases offline.

    TARGET is `module:function`. PROMPTS is a JSON array of test cases; each item
    is passed to the function as keyword args, or as `args`/`kwargs` if present,
    or as positional args if it is a list.
    """
    module_name, _, fn_name = target.partition(":")
    if not fn_name:
        raise click.BadParameter("expected module:function, e.g. my_module:reward_fn")

    sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise click.ClickException(f"could not import '{module_name}': {exc}") from exc
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise click.ClickException(f"'{module_name}' has no attribute '{fn_name}'")

    cases = json.loads(Path(prompts).read_text(encoding="utf-8"))
    watched = watch(
        fn, name=fn_name, window_size=window, sensitivity=sensitivity, max_reward=max_reward
    )
    for case in cases:
        if isinstance(case, dict) and ("args" in case or "kwargs" in case):
            watched(*case.get("args", []), **case.get("kwargs", {}))
        elif isinstance(case, dict):
            watched(**case)
        elif isinstance(case, list):
            watched(*case)
        else:
            watched(case)

    store = watched.store
    if store.count == 0:
        console.print("[yellow]No cases scored.[/yellow]")
        return
    _print_report(store, watched.engine)


def _print_report(store: MetricStore, engine: DetectionEngine) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("records", f"{store.count:,}")
    table.add_row("mean", f"{store.rolling_mean:.4f}")
    table.add_row("std", f"{store.rolling_std:.4f}")
    table.add_row("min / max", f"{store.percentile(0):.4f} / {store.percentile(100):.4f}")
    table.add_row("p50 / p95", f"{store.percentile(50):.4f} / {store.percentile(95):.4f}")

    console.print(f"[bold]{store.name}[/bold]")
    console.print(table)
    console.print()
    _print_detectors(engine)
    console.print()
    console.print(diagnosis(store))


def _print_detectors(engine: DetectionEngine) -> None:
    colors = {"OK": "green", "WARNING": "yellow", "ALERT": "red"}
    labels = {"OK": "OK", "WARNING": "WARN", "ALERT": "ALERT"}

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=6)
    table.add_column()
    for detector in engine.detectors:
        result = engine.latest[detector.name]
        status = result.status.value
        color = colors.get(status)
        badge = f"[{color}]{labels[status]}[/{color}]" if color else "[dim]-[/dim]"
        message = f"  [dim]{result.message}[/dim]" if result.message else ""
        table.add_row(badge, f"{detector.label}{message}")
    console.print(table)


if __name__ == "__main__":
    main()
