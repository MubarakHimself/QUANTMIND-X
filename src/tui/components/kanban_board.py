"""
KanbanBoard Component - 4-Column Strategy Pipeline

Displays strategies in a Kanban board layout with 4 columns:
- Inbox: Pending strategies
- Processing: Being processed by TRD Agent
- Extracting: Ready for extraction
- Done: Primal/complete strategies
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from textual.widget import Widget
from textual.reactive import reactive
from textual import log
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich import box

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class StrategyStatus(str, Enum):
    """Strategy status in the pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    PRIMAL = "primal"
    QUARANTINED = "quarantined"


@dataclass
class StrategyFolder:
    """Represents a strategy folder in the pipeline."""
    id: str
    name: str
    status: StrategyStatus
    created_at: str
    has_nprd: bool = False
    has_trd: bool = False
    has_ea: bool = False
    has_backtest: bool = False


class KanbanColumn:
    """Represents a Kanban board column."""

    def __init__(
        self,
        id: str,
        title: str,
        status_map: List[StrategyStatus],
        color: str = "blue"
    ):
        self.id = id
        self.title = title
        self.status_map = status_map
        self.color = color
        self.strategies: List[StrategyFolder] = []

    def add_strategy(self, strategy: StrategyFolder) -> None:
        """Add a strategy to this column."""
        if strategy.status in self.status_map:
            self.strategies.append(strategy)

    def clear(self) -> None:
        """Clear all strategies from this column."""
        self.strategies = []

    @property
    def count(self) -> int:
        """Get number of strategies in this column."""
        return len(self.strategies)


class KanbanBoard(Widget):
    """Kanban board widget for the YouTube-EA pipeline.

    Displays strategies in a 4-column layout:
    - Inbox: Pending strategies (status='pending')
    - Processing: Being processed (status='processing')
    - Extracting: Ready for extraction (status='ready')
    - Done: Primal/complete (status='primal')
    """

    DEFAULT_CSS = """
    KanbanBoard {
        height: 1fr;
    }

    KanbanBoard .column-title {
        text-style: bold;
    }

    KanbanBoard .strategy-item {
        padding: 0 1;
    }

    KanbanBoard .status-badge {
        text-style: bold;
    }

    KanbanBoard .status-pending {
        color: $primary;
    }

    KanbanBoard .status-processing {
        color: $warning;
    }

    KanbanBoard .status-ready {
        color: $accent;
    }

    KanbanBoard .status-primal {
        color: $success;
    }

    KanbanBoard .status-quarantined {
        color: $error;
    }
    """

    # Reactive state
    strategies: reactive[List[StrategyFolder]] = reactive(list)
    is_loading: reactive[bool] = reactive(False)
    error: reactive[Optional[str]] = reactive(None)

    def __init__(self, id: str | None = None, api_base_url: str = "http://localhost:8000"):
        """Initialize the KanbanBoard widget.

        Args:
            id: Widget ID
            api_base_url: Base URL for the API
        """
        super().__init__(id=id)
        self._api_base_url = api_base_url

        # Define columns
        self.columns: List[KanbanColumn] = [
            KanbanColumn("inbox", "📥 Inbox", [StrategyStatus.PENDING], "blue"),
            KanbanColumn("processing", "⚙️ Processing", [StrategyStatus.PROCESSING], "yellow"),
            KanbanColumn("extracting", "📤 Extracting", [StrategyStatus.READY], "cyan"),
            KanbanColumn("done", "✅ Done", [StrategyStatus.PRIMAL], "green"),
        ]

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.fetch_strategies()
        # Auto-refresh every 15 seconds
        self.set_interval(15.0, self.fetch_strategies)

    async def fetch_strategies(self) -> None:
        """Fetch strategies from the API."""
        self.is_loading = True
        self.error = None

        try:
            if not HTTPX_AVAILABLE:
                self.error = "httpx not installed"
                self.strategies = []
                return

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._api_base_url}/api/strategies")
                response.raise_for_status()
                data = response.json()

                # Parse response into StrategyFolder objects
                strategies = []
                for item in data:
                    strategies.append(StrategyFolder(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        status=StrategyStatus(item.get("status", "pending")),
                        created_at=item.get("created_at", ""),
                        has_nprd=item.get("has_nprd", False),
                        has_trd=item.get("has_trd", False),
                        has_ea=item.get("has_ea", False),
                        has_backtest=item.get("has_backtest", False)
                    ))

                self.strategies = strategies

        except Exception as e:
            log(f"Failed to fetch strategies: {e}")
            self.error = str(e)
            self.strategies = []
        finally:
            self.is_loading = False

    def _use_mock_data(self) -> None:
        """Use mock data for development/testing."""
        mock_strategies = [
            StrategyFolder(
                id="london_breakout",
                name="London Breakout",
                status=StrategyStatus.PRIMAL,
                created_at="2026-02-22T10:00:00Z",
                has_nprd=True,
                has_trd=True,
                has_ea=True,
                has_backtest=True
            ),
            StrategyFolder(
                id="ict_silver_bullet",
                name="ICT Silver Bullet",
                status=StrategyStatus.PROCESSING,
                created_at="2026-02-22T11:00:00Z",
                has_nprd=True,
                has_trd=False,
                has_ea=False,
                has_backtest=False
            ),
            StrategyFolder(
                id="smc_institutional",
                name="SMC Institutional",
                status=StrategyStatus.READY,
                created_at="2026-02-22T12:00:00Z",
                has_nprd=True,
                has_trd=True,
                has_ea=False,
                has_backtest=False
            ),
            StrategyFolder(
                id="range reversal",
                name="Range Reversal",
                status=StrategyStatus.PENDING,
                created_at="2026-02-22T13:00:00Z",
                has_nprd=False,
                has_trd=False,
                has_ea=False,
                has_backtest=False
            ),
        ]
        self.strategies = mock_strategies

    def _group_strategies_by_column(self) -> Dict[str, List[StrategyFolder]]:
        """Group strategies by their column based on status."""
        # Clear all columns
        for column in self.columns:
            column.clear()

        # Assign strategies to columns
        for strategy in self.strategies:
            for column in self.columns:
                if strategy.status in column.status_map:
                    column.add_strategy(strategy)
                    break

        return {col.id: col.strategies for col in self.columns}

    def _get_status_style(self, status: StrategyStatus) -> str:
        """Get text style for a status."""
        styles = {
            StrategyStatus.PENDING: "status-pending",
            StrategyStatus.PROCESSING: "status-processing",
            StrategyStatus.READY: "status-ready",
            StrategyStatus.PRIMAL: "status-primal",
            StrategyStatus.QUARANTINED: "status-quarantined",
        }
        return styles.get(status, "")

    def _render_strategy_item(self, strategy: StrategyFolder) -> Text:
        """Render a single strategy item."""
        text = Text()

        # Name
        text.append(f"• {strategy.name}", style="bold")

        # Status indicators
        badges = []
        if strategy.has_nprd:
            badges.append("NPRD")
        if strategy.has_trd:
            badges.append("TRD")
        if strategy.has_ea:
            badges.append("EA")
        if strategy.has_backtest:
            badges.append("BT")

        if badges:
            text.append(" [")
            text.append(",".join(badges), style="dim")
            text.append("]")

        return text

    def _render_column(self, column: KanbanColumn) -> Panel:
        """Render a single column."""
        # Render strategies in this column
        items = []
        if not column.strategies:
            items.append(Text("Empty", style="dim italic"))
        else:
            for strategy in column.strategies:
                items.append(self._render_strategy_item(strategy))
                items.append(Text(""))

        content = Group(*items) if items else Text("Empty", style="dim italic")

        title = f"{column.title} ({column.count})"

        return Panel(
            content,
            title=title,
            border_style=column.color,
            padding=(0, 1),
            box=box.ROUNDED
        )

    def render(self) -> Panel:
        """Render the Kanban board."""
        # Group strategies by column (side effect updates column.strategies)
        self._group_strategies_by_column()

        # Create table for columns
        board = Table.grid(padding=(0, 2), expand=True)
        for column in self.columns:
            board.add_column()

        max_height = 0

        # Calculate max height for alignment
        for column in self.columns:
            height = len(column.strategies) or 1
            max_height = max(max_height, height)

        # Render columns
        column_panels = [self._render_column(col) for col in self.columns]

        # Create row with all columns
        row_content = Table.grid(padding=(0, 1), expand=True)
        for panel in column_panels:
            row_content.add_column()
        row_content.add_row(*column_panels)

        title = "📊 Strategy Pipeline"
        if self.error:
            title += f" [red]Error: {self.error}[/]"
        elif self.is_loading:
            title += " [dim]Loading...[/]"

        title += f" | Total: {len(self.strategies)}"

        return Panel(
            row_content,
            title=title,
            border_style="bright_blue",
            padding=(0, 1),
            box=box.DOUBLE
        )
