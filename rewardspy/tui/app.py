"""The live Textual dashboard.

Layout is six panels plus a custom header and a key-binding footer. The widgets
are deliberately thin: each one owns a store and a render function, and on every
tick it asks the render layer for a fresh Rich renderable. All the visual logic
lives in ``render.py`` so it stays testable.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Static

from ..exporters import write_csv
from ..store import MetricStore
from . import render, theme


class HeaderBar(Static):
    """Top status line: brand, run name, step, runtime, and health."""

    def __init__(self, store: MetricStore, start_time: float) -> None:
        super().__init__()
        self.store = store
        self.start_time = start_time

    def refresh_bar(self) -> None:
        try:
            self._render_bar()
        except RuntimeError:
            pass

    def _render_bar(self) -> None:
        engine = self.store.engine
        overall = getattr(engine, "overall", None)
        status = overall.value if overall is not None else "OK"
        _, color = theme.status_style(status)

        elapsed = int(time.time() - self.start_time)
        clock = f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"

        bar = Text()
        bar.append("rewardspy", style=f"bold {theme.ACCENT}")
        bar.append("  ●  ", style=color)
        bar.append(self.store.name, style=theme.TEXT)
        bar.append("   │   ", style=theme.DIM)
        bar.append(f"step {self.store.step_counter:,}", style=theme.MUTED)
        bar.append("   │   ", style=theme.DIM)
        bar.append(f"runtime {clock}", style=theme.MUTED)
        bar.append("   │   ", style=theme.DIM)
        bar.append(f"{status}", style=f"bold {color}")
        self.update(bar)


class Panel(Static):
    """A bordered panel that re-renders from the store each tick.

    ``mode`` controls what the render function is given: nothing, the content
    width, or the width and height (for the chart).
    """

    def __init__(
        self,
        store: MetricStore,
        render_fn,
        title: str,
        mode: str = "plain",
        ascii_charts: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self.store = store
        self.render_fn = render_fn
        self.mode = mode
        self.ascii_charts = ascii_charts
        self.border_title = title

    def refresh_panel(self) -> None:
        width = max(self.content_size.width, 12)
        height = max(self.content_size.height, 4)
        try:
            if self.mode == "chart":
                self.update(self.render_fn(self.store, width, height, ascii=self.ascii_charts))
            elif self.mode == "width":
                self.update(self.render_fn(self.store, width))
            else:
                self.update(self.render_fn(self.store))
        except RuntimeError:
            # The store can be mutated by a training thread while we read it
            # (deque/dict changed size during iteration). Skip this frame; the
            # next tick will pick up a consistent snapshot.
            pass


class RewardSpyApp(App):
    CSS_PATH = "dashboard.tcss"
    TITLE = "rewardspy"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "export", "Export CSV"),
        ("a", "clear_alerts", "Clear alerts"),
    ]

    def __init__(
        self, store: MetricStore, interval: float = 0.5, ascii_charts: bool = False
    ) -> None:
        super().__init__()
        self.store = store
        self.interval = interval
        self.ascii_charts = ascii_charts
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        self._header = HeaderBar(self.store, self._start_time)
        yield self._header
        yield Panel(self.store, render.diagnosis, "Diagnosis", id="diagnosis", classes="panel")
        with Horizontal(id="row-top"):
            yield Panel(self.store, render.reward_overview, "Reward overview",
                        id="overview", classes="panel")
            yield Panel(self.store, render.hack_status, "Hack status",
                        id="hack", classes="panel")
            yield Panel(self.store, render.reward_curve, "Reward curve",
                        mode="chart", ascii_charts=self.ascii_charts,
                        id="curve", classes="panel")
        with Horizontal(id="row-mid"):
            yield Panel(self.store, render.component_bars, "Components",
                        mode="width", id="components", classes="panel")
            yield Panel(self.store, render.alerts_log, "Alerts",
                        id="alerts", classes="panel")
        yield Panel(self.store, render.rollouts_table, "Recent rollouts",
                    id="rollouts", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        # Render once layout has assigned real sizes, then poll on an interval.
        self.call_after_refresh(self.refresh_all)
        self.set_interval(self.interval, self.refresh_all)

    def on_resize(self) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        self._header.refresh_bar()
        for panel in self.query(Panel):
            panel.refresh_panel()

    def action_export(self) -> None:
        path = Path(f"rewardspy_{self.store.name}.csv")
        write_csv(list(self.store.records), path)
        self.notify(f"Exported {self.store.count} rows to {path}", title="Export")

    def action_clear_alerts(self) -> None:
        count = len(self.store.alerts)
        self.store.alerts.clear()
        self.notify(f"Cleared {count} alerts", title="Alerts")
