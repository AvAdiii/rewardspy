"""Pure render functions for the dashboard.

Each function turns a ``MetricStore`` into a Rich renderable. They hold no state
and never touch the terminal, so they can be unit tested directly and reused by
the widgets in ``app.py``. The headline piece is ``diagnosis``, which reads the
detectors and explains, in plain language, what is happening and what to do.
"""

from __future__ import annotations

import statistics

from rich.table import Table
from rich.text import Text

from ..store import MetricStore
from . import theme

_DETECTOR_ORDER = ["variance", "slope", "component", "ceiling", "length"]
_DETECTOR_LABELS = {
    "variance": "Variance",
    "slope": "Slope",
    "component": "Component",
    "ceiling": "Ceiling",
    "length": "Length",
}


# -- shared helpers ------------------------------------------------------------

def _resample(values: list[float], width: int) -> list[float]:
    n = len(values)
    if n <= width:
        return list(values)
    out: list[float] = []
    for i in range(width):
        start = i * n // width
        end = max((i + 1) * n // width, start + 1)
        chunk = values[start:end]
        out.append(sum(chunk) / len(chunk))
    return out


def _format(value: float) -> str:
    if abs(value) < 5e-4:  # avoid "-0.000"
        value = 0.0
    return f"{value:.3f}"


def _dominant_component(store: MetricStore) -> tuple[str, float] | None:
    means = {
        name: stat.rolling_mean
        for name, stat in store.component_stats.items()
        if name != "total"
    }
    if len(means) < 2:
        return None
    total = sum(abs(v) for v in means.values())
    if total < 1e-6:
        return None
    top = max(means, key=lambda k: abs(means[k]))
    return top, abs(means[top]) / total


def _variance_ratio(store: MetricStore) -> float | None:
    baseline = store.baseline_std()
    if baseline < 1e-6:
        return None
    return store.rolling_std / baseline


def _trend(store: MetricStore) -> tuple[str, str]:
    """Return ``(arrow_word, color)`` describing the recent reward direction."""
    window = store.recent(store.window_size)
    if len(window) < 20:
        return "collecting", theme.MUTED
    half = len(window) // 2
    early = statistics.fmean(r.scalar_reward for r in window[:half])
    late = statistics.fmean(r.scalar_reward for r in window[half:])
    eps = 0.01 * abs(early) + 1e-6
    if late - early > eps:
        return "▲ rising", theme.TEXT
    if late - early < -eps:
        return "▼ falling", theme.TEXT
    return "▬ flat", theme.MUTED


def _ceiling_rate(store: MetricStore) -> float:
    ceiling = store.observed_max
    return store.ceiling_rate(ceiling) if ceiling > 0 else 0.0


def sparkline(values: list[float], width: int = 40, ascii: bool = False) -> str:
    if not values:
        return ""
    sampled = _resample(values, width)
    low, high = min(sampled), max(sampled)
    span = high - low or 1.0
    ramp = theme.ASCII_BLOCKS if ascii else theme.BLOCKS
    return "".join(
        ramp[min(len(ramp) - 1, int((v - low) / span * (len(ramp) - 1)))] for v in sampled
    )


# -- diagnosis (the headline) --------------------------------------------------

def diagnosis(store: MetricStore) -> Text:
    """A plain-language verdict synthesized from the detectors."""
    engine = store.engine
    latest = getattr(engine, "latest", {}) if engine is not None else {}
    actionable = {name: res for name, res in latest.items() if res.actionable}

    out = Text()
    if engine is None or store.count < store.window_size:
        out.append("Collecting data... ", style=theme.MUTED)
        out.append(f"({store.count} calls so far)", style=theme.DIM)
        return out

    if not actionable:
        out.append("✓ Healthy training\n", style=f"bold {theme.OK}")
        out.append(
            "Reward is moving with no reward-hacking signatures. Keep going.",
            style=theme.TEXT,
        )
        return out

    has_alert = any(res.status.value == "ALERT" for res in actionable.values())
    glyph, color = theme.status_style("ALERT" if has_alert else "WARNING")
    headline = "Likely reward hacking" if has_alert else "Early warning signs"
    out.append(f"{glyph} {headline}\n", style=f"bold {color}")

    reasons: list[str] = []
    if "ceiling" in actionable:
        reasons.append(f"~{_ceiling_rate(store):.0%} of rollouts are at the reward ceiling")
    if "variance" in actionable:
        ratio = _variance_ratio(store)
        if ratio is not None:
            reasons.append(f"reward variance collapsed to {ratio:.0%} of baseline")
        else:
            reasons.append("reward variance collapsed")
    if "component" in actionable:
        dom = _dominant_component(store)
        if dom is not None:
            reasons.append(f"'{dom[0]}' now drives {dom[1]:.0%} of the reward")
    if "length" in actionable:
        reasons.append("response length is drifting")
    if "slope" in actionable:
        reasons.append("the reward distribution shifted suddenly")

    out.append("The reward curve looks fine ", style=theme.TEXT)
    out.append(f"(mean {store.rolling_mean:.2f}, {_trend(store)[0]})", style=theme.MUTED)
    out.append(", but ", style=theme.TEXT)
    out.append(_join(reasons) + ".\n", style=theme.TEXT)

    if store.alerts:
        first = min(a.step for a in store.alerts)
        out.append(
            f"First flagged near step {first}. Inspect rollouts from there for shortcut behavior.",
            style=theme.MUTED,
        )
    return out


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return "; ".join(parts[:-1]) + "; and " + parts[-1]


# -- panels --------------------------------------------------------------------

def _smooth(values: list[float], k: int) -> list[float]:
    """Centered moving average so the trend line reads cleanly."""
    if k <= 1 or len(values) <= 2:
        return values
    half = k // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _rolling_series(values: list[float], window: int) -> list[float]:
    """Trailing rolling mean at each step. This is what the curve plots: the
    reward trend, which rises and plateaus rather than the raw per-step noise."""
    if window <= 1:
        return list(values)
    out: list[float] = []
    total = 0.0
    queue: list[float] = []
    for v in values:
        queue.append(v)
        total += v
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def reward_curve(store: MetricStore, width: int = 60, height: int = 8, ascii: bool = False) -> Text:
    """A smoothed block area chart of the reward trend with a value gutter.

    Block characters render cleanly in any terminal font, unlike braille. The
    plotted series is the rolling mean, so the curve shows the trend (rising,
    plateauing, falling) instead of raw per-step noise.
    """
    values = [r.scalar_reward for r in store.records]
    if not values:
        return Text("Waiting for the first reward call...", style=theme.MUTED)

    cols = max(width - 8, 8)  # 6 label cols + axis + one col of slack
    rows = max(height - 2, 3)  # leave room for the axis line and footer
    return _area_chart(values, cols, rows, store.step_counter, ascii=ascii)


def _area_chart(
    values: list[float],
    cols: int,
    rows: int,
    last_step: int,
    ascii: bool = False,
) -> Text:
    window = max(20, len(values) // 20)
    series = _smooth(_resample(_rolling_series(values, window), cols), 3)
    low, high = min(series), max(series)
    span = high - low or 1.0

    units = rows * 8  # eighth-block vertical resolution
    levels = [(v - low) / span * units for v in series]
    ramp = theme.ASCII_BLOCKS if ascii else theme.BLOCKS
    axis = "|" if ascii else "│"
    corner = "+" if ascii else "└"
    dash = "-" if ascii else "─"
    fill = "#" if ascii else "█"

    out = Text()
    for r in range(rows - 1, -1, -1):
        if r == rows - 1:
            label = f"{high:>5.2f} "
        elif r == 0:
            label = f"{low:>5.2f} "
        else:
            label = " " * 6
        out.append(label, style=theme.DIM)
        out.append(axis, style=theme.DIM)
        for level in levels:
            cell = level - r * 8
            if cell >= 8:
                out.append(fill, style=theme.CURVE_FILL)
            elif cell > 0:
                out.append(ramp[max(1, min(len(ramp) - 1, int(cell)))], style=theme.CURVE)
            else:
                out.append(" ")
        out.append("\n")
    out.append(" " * 6 + corner + dash * cols + "\n", style=theme.DIM)
    out.append(f"{'step 0':>6}{'step ' + format(last_step, ','):>{cols + 1}}", style=theme.MUTED)
    return out


def reward_overview(store: MetricStore) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style=theme.MUTED)
    table.add_column(justify="right")

    arrow, arrow_color = _trend(store)
    ceiling = _ceiling_rate(store)
    var_ratio = _variance_ratio(store)

    def colored(text: str, color: str) -> Text:
        return Text(text, style=color)

    ceiling_color = theme.ALERT if ceiling > 0.8 else theme.WARN if ceiling > 0.5 else theme.TEXT
    if var_ratio is None:
        var_text, var_color = "–", theme.MUTED
    else:
        var_text = f"{var_ratio:.0%}"
        if var_ratio < 0.2:
            var_color = theme.ALERT
        elif var_ratio < 0.5:
            var_color = theme.WARN
        else:
            var_color = theme.TEXT

    minmax = f"{_format(store.percentile(0))} / {_format(store.percentile(100))}"
    table.add_row("mean", colored(_format(store.rolling_mean), theme.ACCENT_SOFT))
    table.add_row("std", colored(_format(store.rolling_std), theme.TEXT))
    table.add_row("trend", colored(arrow, arrow_color))
    table.add_row("at ceiling", colored(f"{ceiling:.0%}", ceiling_color))
    table.add_row("var vs base", colored(var_text, var_color))
    table.add_row("min / max", colored(minmax, theme.TEXT))
    table.add_row("p50", colored(_format(store.percentile(50)), theme.TEXT))
    return table


def hack_status(store: MetricStore) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column(justify="left")

    engine = store.engine
    latest = getattr(engine, "latest", {}) if engine is not None else {}

    for name in _DETECTOR_ORDER:
        result = latest.get(name)
        status = result.status.value if result is not None else "INSUFFICIENT_DATA"
        glyph, color = theme.status_style(status)
        table.add_row(Text(glyph, style=color), Text(_DETECTOR_LABELS[name], style=theme.TEXT))

    overall = getattr(engine, "overall", None)
    overall_value = overall.value if overall is not None else "OK"
    glyph, color = theme.status_style(overall_value)
    table.add_row("", "")
    table.add_row(
        Text(glyph, style=color),
        Text(f"Overall: {overall_value}", style=f"bold {color}"),
    )
    return table


def component_bars(store: MetricStore, width: int = 30) -> Table:
    components = {
        name: stat.rolling_mean
        for name, stat in store.component_stats.items()
        if name != "total"
    }
    if not components:
        return _muted("No components reported.")

    scale = max((abs(v) for v in components.values()), default=1.0) or 1.0
    bar_width = max(width - 23, 6)

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style=theme.MUTED, width=12)
    table.add_column(justify="left", width=bar_width)
    table.add_column(justify="right", style=theme.TEXT, width=7)

    for name, value in components.items():
        filled = int(abs(value) / scale * bar_width)
        color = theme.ALERT if value < 0 else theme.ACCENT
        bar = Text("█" * filled, style=color)
        bar.append("░" * (bar_width - filled), style=theme.DIM)
        table.add_row(_truncate(name, 12), bar, _format(value))
    return table


def alerts_log(store: MetricStore, limit: int = 6) -> Text:
    if not store.alerts:
        return Text("No alerts. Training looks healthy.", style=theme.OK)

    out = Text()
    for alert in reversed(store.alerts[-limit:]):
        _, color = theme.status_style(alert.status)
        out.append(f"{alert.status} ", style=f"bold {color}")
        out.append(f"step {alert.step}\n", style=theme.MUTED)
        out.append(f"  {alert.message}\n", style=theme.TEXT)
    return out


def rollouts_table(store: MetricStore, limit: int = 8) -> Table:
    records = store.recent(limit)
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style=theme.MUTED)
    table.add_column(justify="right", style=theme.TEXT)
    table.add_column(justify="left")

    table.add_row(
        Text("step", style=f"bold {theme.MUTED}"),
        Text("reward", style=f"bold {theme.MUTED}"),
        Text("note", style=f"bold {theme.MUTED}"),
    )
    if not records:
        table.add_row("–", "–", "")
        return table

    ceiling = store.observed_max
    for record in reversed(records):
        note = _rollout_note(record, ceiling)
        table.add_row(str(record.step), _format(record.scalar_reward), note)
    return table


def _rollout_note(record, ceiling: float) -> Text:
    comps = {k: v for k, v in record.components.items() if k != "total"}
    correctness = comps.get("correctness")
    others = {k: v for k, v in comps.items() if k != "correctness"}
    if correctness is not None and correctness <= 1e-6 and any(v > 0 for v in others.values()):
        return Text("shortcut: no correctness", style=theme.ALERT)
    if ceiling > 0 and record.scalar_reward >= ceiling * 0.99:
        return Text("at ceiling", style=theme.WARN)
    return Text("", style=theme.DIM)


def _muted(message: str) -> Table:
    table = Table.grid()
    table.add_column()
    table.add_row(Text(message, style=theme.MUTED))
    return table


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
