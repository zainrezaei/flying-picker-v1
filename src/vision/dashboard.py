"""
dashboard.py — Rich terminal dashboard for the Flying Picker vision pipeline.

Replaces print() output with a live-updating, multi-panel terminal UI.
The robot coordinates sent via RTDE are displayed prominently.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule

# ------------------------------------------------------------------ #
# Sparkline helper                                                     #
# ------------------------------------------------------------------ #

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a Unicode sparkline."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    for v in values[-width:]:
        idx = int((v - lo) / rng * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


# ------------------------------------------------------------------ #
# Data container                                                       #
# ------------------------------------------------------------------ #

@dataclass
class DashboardData:
    """All data the dashboard needs to render one frame."""

    # Frame
    frame_num: int = 0

    # Detection (pixel space)
    detected: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    angle_deg: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    confidence: float = 0.0
    part_id: str = ""

    # World coordinates (mm) — from homography
    world_x_mm: Optional[float] = None
    world_y_mm: Optional[float] = None
    world_angle_deg: Optional[float] = None

    # Pick coordinates (belt-compensated)
    pick_x_mm: Optional[float] = None
    pick_y_mm: Optional[float] = None
    pick_angle_deg: Optional[float] = None

    # Robot send — THE COORDINATES ACTUALLY SENT TO THE ROBOT
    robot_sent_this_frame: bool = False
    robot_last_x_mm: Optional[float] = None
    robot_last_y_mm: Optional[float] = None
    robot_last_angle_deg: Optional[float] = None
    robot_last_valid: Optional[float] = None
    robot_last_frame: int = 0
    robot_total_sends: int = 0
    robot_object_present: bool = False

    # Performance
    proc_time_ms: float = 0.0
    actual_fps: float = 0.0

    # System status
    robot_connected: bool = False
    calib_loaded: bool = False
    homog_loaded: bool = False
    belt_enabled: bool = False
    classifier_enabled: bool = False
    tracker_state: str = "IDLE"

    # Config summary
    source_path: str = ""
    resolution: str = ""
    threshold: str = ""
    min_area: int = 0
    roi_info: str = ""

    # Log messages (most recent)
    log_messages: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Dashboard class                                                      #
# ------------------------------------------------------------------ #

class Dashboard:
    """Rich terminal dashboard for the vision pipeline.

    Usage::

        dash = Dashboard(refresh_rate=15, sparkline_length=50)
        dash.start()
        ...
        for frame in frames:
            data = DashboardData(...)
            dash.update(data)
        ...
        dash.stop()
    """

    def __init__(
        self,
        refresh_rate: int = 15,
        sparkline_length: int = 50,
        show_sparkline: bool = True,
    ):
        self._console = Console()
        self._live: Optional[Live] = None
        self._refresh_rate = refresh_rate
        self._sparkline_len = sparkline_length
        self._show_sparkline = show_sparkline

        # Performance history
        self._latency_history: deque[float] = deque(maxlen=sparkline_length)
        self._fps_history: deque[float] = deque(maxlen=sparkline_length)
        self._total_frames: int = 0
        self._total_detections: int = 0
        self._start_time: float = time.time()

        # Log buffer
        self._log_buffer: deque[str] = deque(maxlen=12)

    # ------------------------------------------------------------ #
    # Lifecycle                                                      #
    # ------------------------------------------------------------ #

    def start(self):
        """Start the live dashboard display."""
        self._start_time = time.time()
        self._live = Live(
            self._build_layout(DashboardData()),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=True,
        )
        self._live.start()

    def stop(self):
        """Stop the live dashboard display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------ #
    # Per-frame update                                               #
    # ------------------------------------------------------------ #

    def update(self, data: DashboardData):
        """Update the dashboard with new frame data."""
        # Track history
        self._total_frames += 1
        if data.detected:
            self._total_detections += 1
        self._latency_history.append(data.proc_time_ms)
        if data.actual_fps > 0:
            self._fps_history.append(data.actual_fps)

        # Merge log messages
        for msg in data.log_messages:
            self._log_buffer.append(msg)

        if self._live is not None:
            self._live.update(self._build_layout(data))

    def log(self, message: str):
        """Add a log message to the dashboard log panel."""
        self._log_buffer.append(message)

    # ------------------------------------------------------------ #
    # Layout builder                                                 #
    # ------------------------------------------------------------ #

    def _build_layout(self, d: DashboardData) -> Panel:
        """Build the full dashboard layout from data."""

        # ============================================================
        # 1. ROBOT COORDINATES — Big, prominent, top of dashboard
        # ============================================================
        robot_panel = self._build_robot_panel(d)

        # ============================================================
        # 2. System Status + Config (side by side)
        # ============================================================
        status_table = self._build_status_table(d)
        config_table = self._build_config_table(d)
        top_row = Columns(
            [
                Panel(status_table, title="[bold cyan]System Status", border_style="cyan", expand=True),
                Panel(config_table, title="[bold cyan]Pipeline Config", border_style="cyan", expand=True),
            ],
            equal=True,
            expand=True,
        )

        # ============================================================
        # 3. Live Detection
        # ============================================================
        detection_panel = self._build_detection_panel(d)

        # ============================================================
        # 4. Performance
        # ============================================================
        perf_panel = self._build_performance_panel(d)

        # ============================================================
        # 5. Log
        # ============================================================
        log_panel = self._build_log_panel()

        # Assemble
        from rich.console import Group
        body = Group(
            robot_panel,
            Text(""),
            top_row,
            Text(""),
            Columns(
                [
                    Panel(detection_panel, title="[bold magenta]Live Detection", border_style="magenta", expand=True),
                    Panel(perf_panel, title="[bold blue]Performance", border_style="blue", expand=True),
                ],
                equal=True,
                expand=True,
            ),
            Text(""),
            Panel(log_panel, title="[bold dim]Log", border_style="dim", expand=True),
            Text(""),
            Text.from_markup("  Press [bold cyan]'q'[/] in OpenCV window to quit.", style="dim"),
        )

        return Panel(
            body,
            title="[bold white on blue]  🚀 Flying Picker — Vision Dashboard  [/]",
            border_style="bold blue",
            padding=(1, 2),
        )

    # ------------------------------------------------------------ #
    # Panel builders                                                 #
    # ------------------------------------------------------------ #

    def _build_robot_panel(self, d: DashboardData) -> Panel:
        """Build the ROBOT COORDINATES panel — the most prominent element."""

        if d.robot_last_x_mm is not None and d.robot_total_sends > 0:
            # We have sent coordinates to the robot
            x_val = d.robot_last_x_mm
            y_val = d.robot_last_y_mm
            a_val = d.robot_last_angle_deg

            # Build coordinate lines as centered text for maximum clarity
            coord_lines = Text(justify="center")
            coord_lines.append("\n")
            coord_lines.append("  X  ", style="bold white")
            coord_lines.append(f"  {x_val:+10.2f}  ", style="bold green on grey11")
            coord_lines.append("  mm\n", style="dim white")
            coord_lines.append("  Y  ", style="bold white")
            coord_lines.append(f"  {y_val:+10.2f}  ", style="bold green on grey11")
            coord_lines.append("  mm\n", style="dim white")
            coord_lines.append("  θ  ", style="bold white")
            coord_lines.append(f"  {a_val:+10.2f}  ", style="bold cyan on grey11")
            coord_lines.append("  °\n", style="dim white")

            # Valid flag
            is_valid = d.robot_last_valid and d.robot_last_valid > 0.5
            valid_str = "✅ Valid" if is_valid else "❌ Invalid"
            valid_style = "bold green" if is_valid else "bold red"

            # Status line
            status_parts = []
            if d.robot_sent_this_frame:
                status_parts.append("[bold yellow]⚡ SENT THIS FRAME[/]")
            status_parts.append(f"[dim]Total sends: {d.robot_total_sends}[/]")
            status_parts.append(f"[dim]Last @ frame {d.robot_last_frame}[/]")
            obj_indicator = "[bold green]● Object present[/]" if d.robot_object_present else "[dim]○ No object[/]"
            status_parts.append(obj_indicator)
            status_line = "    ".join(status_parts)

            from rich.console import Group
            content = Group(
                Align.center(coord_lines),
                Align.center(Text.from_markup(f"[{valid_style}]{valid_str}[/]")),
                Text(""),
                Align.center(Text.from_markup(status_line)),
            )

            border_style = "bold yellow" if d.robot_sent_this_frame else "bold green"
            title_flash = " ⚡ " if d.robot_sent_this_frame else " "
            return Panel(
                content,
                title=f"[bold white on green]{title_flash}🤖 ROBOT COORDINATES (RTDE){title_flash}[/]",
                border_style=border_style,
                expand=True,
            )
        else:
            # No coordinates sent yet
            if d.robot_connected:
                msg = Text("Waiting for first detection...", style="dim yellow")
            else:
                msg = Text("Robot not connected — coordinates shown but not sent", style="dim red")

            content = Align.center(msg)
            return Panel(
                content,
                title="[bold white on dim]  🤖 ROBOT COORDINATES (RTDE)  [/]",
                border_style="dim",
                expand=True,
            )

    def _build_status_table(self, d: DashboardData) -> Table:
        """System status indicators."""
        t = Table(show_header=False, show_edge=False, show_lines=False, expand=True, padding=(0, 1))
        t.add_column("item", style="white", ratio=2)
        t.add_column("status", ratio=3)

        def _dot(ok: bool, yes: str = "Connected", no: str = "Disconnected") -> Text:
            if ok:
                return Text(f"● {yes}", style="bold green")
            return Text(f"○ {no}", style="dim red")

        t.add_row("Robot", _dot(d.robot_connected))
        t.add_row("Camera Cal.", _dot(d.calib_loaded, "Loaded", "Not calibrated"))
        t.add_row("Homography", _dot(d.homog_loaded, "Loaded", "Not calibrated"))
        t.add_row("Belt Comp.", _dot(d.belt_enabled, "Enabled", "Disabled"))
        t.add_row("Classifier", _dot(d.classifier_enabled, "Enabled", "Disabled"))

        # Tracker state with color
        state_colors = {
            "IDLE": "dim",
            "CONFIRMING": "yellow",
            "SENT": "bold green",
        }
        ts = d.tracker_state
        t.add_row("Tracker", Text(f"◆ {ts}", style=state_colors.get(ts, "white")))

        return t

    def _build_config_table(self, d: DashboardData) -> Table:
        """Pipeline configuration summary."""
        t = Table(show_header=False, show_edge=False, show_lines=False, expand=True, padding=(0, 1))
        t.add_column("param", style="dim white", ratio=2)
        t.add_column("value", style="white", ratio=3)

        t.add_row("Source", Text(d.source_path or "—", style="cyan"))
        t.add_row("Resolution", Text(d.resolution or "—"))
        t.add_row("Threshold", Text(d.threshold or "—"))
        t.add_row("Min Area", Text(f"{d.min_area:,} px²" if d.min_area else "—"))
        t.add_row("ROI", Text(d.roi_info or "—"))

        return t

    def _build_detection_panel(self, d: DashboardData) -> Table:
        """Current frame detection data."""
        t = Table(show_header=False, show_edge=False, show_lines=False, expand=True, padding=(0, 1))
        t.add_column("param", style="dim white", ratio=2)
        t.add_column("value", ratio=3)

        t.add_row("Frame", Text(f"{d.frame_num:,}", style="bold white"))

        if d.detected:
            # Part ID
            if d.part_id:
                t.add_row("Part ID", Text(d.part_id, style="bold cyan"))

            # Pixel position
            t.add_row("Position (px)", Text(f"x={d.center_x:.1f}  y={d.center_y:.1f}", style="white"))

            # World position
            if d.world_x_mm is not None:
                t.add_row(
                    "Position (mm)",
                    Text(f"X={d.world_x_mm:.1f}  Y={d.world_y_mm:.1f}", style="bold yellow")
                )

            # Pick position (belt compensated)
            if d.pick_x_mm is not None:
                t.add_row(
                    "Pick pos (mm)",
                    Text(f"X={d.pick_x_mm:.1f}  Y={d.pick_y_mm:.1f}", style="bold green")
                )

            t.add_row("Angle", Text(f"{d.angle_deg:.1f}°", style="white"))
            t.add_row("BBox", Text(f"{d.bbox_w:.1f} × {d.bbox_h:.1f} px", style="dim"))

            # Confidence bar
            conf = d.confidence
            bar_len = 20
            filled = int(conf * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            conf_color = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
            conf_text = Text(f"{bar} {conf:.2f}", style=conf_color)
            t.add_row("Confidence", conf_text)

            t.add_row("Proc Time", Text(f"{d.proc_time_ms:.1f} ms", style="dim"))
        else:
            t.add_row("Status", Text("No object detected", style="dim red"))

        return t

    def _build_performance_panel(self, d: DashboardData) -> Table:
        """Performance metrics."""
        t = Table(show_header=False, show_edge=False, show_lines=False, expand=True, padding=(0, 1))
        t.add_column("metric", style="dim white", ratio=2)
        t.add_column("value", ratio=3)

        # FPS
        fps_color = "green" if d.actual_fps >= 25 else "yellow" if d.actual_fps >= 15 else "red"
        t.add_row("FPS (actual)", Text(f"{d.actual_fps:.1f}", style=f"bold {fps_color}"))

        # Average latency
        if self._latency_history:
            avg_lat = sum(self._latency_history) / len(self._latency_history)
            t.add_row("Avg Latency", Text(f"{avg_lat:.1f} ms", style="white"))

            # Sparkline
            if self._show_sparkline and len(self._latency_history) > 2:
                spark = _sparkline(list(self._latency_history), width=self._sparkline_len)
                t.add_row("Latency Graph", Text(spark, style="cyan"))

        # Frame counts
        t.add_row("Total Frames", Text(f"{self._total_frames:,}", style="white"))

        det_pct = (self._total_detections / self._total_frames * 100) if self._total_frames > 0 else 0
        t.add_row(
            "Detections",
            Text(f"{self._total_detections:,} ({det_pct:.1f}%)", style="green" if det_pct > 80 else "yellow")
        )

        # Uptime
        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        t.add_row("Uptime", Text(f"{mins:02d}:{secs:02d}", style="dim"))

        return t

    def _build_log_panel(self) -> Text:
        """Scrolling log messages."""
        if not self._log_buffer:
            return Text("  No messages yet.", style="dim")

        result = Text()
        for i, msg in enumerate(self._log_buffer):
            if i > 0:
                result.append("\n")
            if "[SEND]" in msg:
                result.append(msg, style="green")
            elif "[WARNING]" in msg or "[ERROR]" in msg:
                result.append(msg, style="yellow")
            else:
                result.append(msg, style="dim")

        return result
