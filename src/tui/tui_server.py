"""
QuantMind TUI Server

A Textual-based terminal UI for monitoring the QuantMind trading system.
Provides real-time health status, bot management, sync status, and trade monitoring.
"""

import os
from typing import Dict, Any, List, Optional

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static
from textual import work

from rich.panel import Panel
from rich.table import Table
from rich.console import Console


# ============== API Client ==============

class QuantMindAPIClient:
    """Async HTTP client for QuantMind API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def get_health(self) -> Dict[str, Any]:
        """Get full health status."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/health/")
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return {"error": str(e), "services": {}}
        except Exception as e:
            return {"error": str(e), "services": {}}
    
    async def get_bots(self) -> List[Dict[str, Any]]:
        """Get list of all bots."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/router/bots")
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.TimeoutException):
            return []
        except Exception:
            return []
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get HMM sync status."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/hmm/status")
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.TimeoutException):
            return {}
        except Exception:
            return {}
    
    async def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades from journal."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(
                f"{self.base_url}/api/journal/trades",
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.TimeoutException):
            return []
        except Exception:
            return []
    
    async def start_bot(self, bot_id: str) -> Dict[str, Any]:
        """Start a bot."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.post(f"{self.base_url}/api/router/bots/{bot_id}/start")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def stop_bot(self, bot_id: str) -> Dict[str, Any]:
        """Stop a bot."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.post(f"{self.base_url}/api/router/bots/{bot_id}/stop")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_session_state(self) -> Dict[str, Any]:
        """Get current trading session state."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/trading/current-session")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_kill_switch_status(self) -> Dict[str, Any]:
        """Get kill switch status."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/kill-switch/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_svss_latest(self) -> Dict[str, Any]:
        """Get latest SVSS values."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/trading/svss/latest")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_trading_session_state(self) -> Dict[str, Any]:
        """Get full trading session state."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/trading/session-state")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_performance_ticker(self) -> Dict[str, Any]:
        """Get performance ticker data (session P&L, daily P&L, open positions)."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/trading/performance")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_node_connectivity(self) -> Dict[str, Any]:
        """Get T1->T2 latency and MT5 terminal status."""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=10.0)
            response = await self.client.get(f"{self.base_url}/api/trading/node-connectivity")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


# ============== TUI Application ==============

class QuantMindTUI(App):
    """Textual TUI for QuantMind monitoring."""
    
    CSS = """
    Screen {
        background: $surface;
    }

    .main-container {
        height: 100%;
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1 1;
        padding: 1;
    }

    .widget {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    .widget-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    Static {
        height: 100%;
    }
    """
    
    def __init__(self, vps_name: str = None, api_base_url: str = None):
        super().__init__()
        self.vps_name = vps_name or os.getenv("QUANTMIND_VPS_NAME", "Trading VPS")
        self.api_base_url = api_base_url or os.getenv("QUANTMIND_API_URL", "http://localhost:8000")
        self.api_client: Optional[QuantMindAPIClient] = None
        
        # Cached data
        self._health_data: Dict[str, Any] = {}
        self._bots_data: List[Dict[str, Any]] = []
        self._sync_data: Dict[str, Any] = {}
        self._trades_data: List[Dict[str, Any]] = []
        self._session_data: Dict[str, Any] = {}
        self._killswitch_data: Dict[str, Any] = {}
        self._svss_data: Dict[str, Any] = {}
        self._performance_data: Dict[str, Any] = {}
        self._node_connectivity_data: Dict[str, Any] = {}
        
        # Current view
        self._current_view = "dashboard"

        # Bot roster pagination
        self._bot_page = 0
        self._bot_page_size = 20
    
    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical():
                yield Static(id="system-status", classes="widget")
                yield Static(id="bot-roster", classes="widget")
            with Vertical():
                yield Static(id="session-state", classes="widget")
                yield Static(id="node-connectivity", classes="widget")
            with Vertical():
                yield Static(id="killswitch-status", classes="widget")
                yield Static(id="performance-ticker", classes="widget")

        yield Footer()

    def on_key(self, event) -> None:
        """Handle keyboard navigation for bot roster pagination."""
        if event.key == "n" or event.key == "N":
            # Next page of bots
            total_pages = max(1, (len(self._bots_data) - 1) // self._bot_page_size + 1)
            self._bot_page = (self._bot_page + 1) % total_pages
            self.update_widgets()
        elif event.key == "p" or event.key == "P":
            # Previous page of bots
            total_pages = max(1, (len(self._bots_data) - 1) // self._bot_page_size + 1)
            self._bot_page = (self._bot_page - 1) % total_pages
            self.update_widgets()
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.api_client = QuantMindAPIClient(base_url=self.api_base_url)
        self.set_interval(1.0, self.refresh_data)
        self.refresh_data()
    
    @work(exclusive=True)
    async def refresh_data(self) -> None:
        """Refresh all data from API."""
        if not self.api_client:
            return

        async with self.api_client:
            # Get all data concurrently
            health = await self.api_client.get_health()
            bots = await self.api_client.get_bots()
            sync = await self.api_client.get_sync_status()
            trades = await self.api_client.get_recent_trades(limit=20)
            session = await self.api_client.get_trading_session_state()
            killswitch = await self.api_client.get_kill_switch_status()
            svss = await self.api_client.get_svss_latest()
            performance = await self.api_client.get_performance_ticker()
            node_conn = await self.api_client.get_node_connectivity()

            self._health_data = health
            self._bots_data = bots
            self._sync_data = sync
            self._trades_data = trades
            self._session_data = session
            self._killswitch_data = killswitch
            self._svss_data = svss
            self._performance_data = performance
            self._node_connectivity_data = node_conn

        # Update widgets
        self.update_widgets()
    
    def update_widgets(self) -> None:
        """Update all widget displays."""
        # System Status
        system_widget = self.query_one("#system-status", Static)
        system_widget.update(self._render_system_status())

        # Bot Roster
        bot_widget = self.query_one("#bot-roster", Static)
        bot_widget.update(self._render_bot_roster())

        # Session State
        session_widget = self.query_one("#session-state", Static)
        session_widget.update(self._render_session_state())

        # Node Connectivity
        node_widget = self.query_one("#node-connectivity", Static)
        node_widget.update(self._render_node_connectivity())

        # Kill Switch Status
        killswitch_widget = self.query_one("#killswitch-status", Static)
        killswitch_widget.update(self._render_killswitch_status())

        # Performance Ticker
        perf_widget = self.query_one("#performance-ticker", Static)
        perf_widget.update(self._render_performance_ticker())
    
    def _render_system_status(self) -> str:
        """Render system status widget."""
        console = Console(force_terminal=True, width=60)
        
        table = Table(title="System Status", show_header=True, box=None)
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="magenta")
        
        services = self._health_data.get("services", {})
        
        service_status_map = {
            "healthy": "[green]✓ Healthy[/green]",
            "degraded": "[yellow]⚠ Degraded[/yellow]",
            "unhealthy": "[red]✗ Unhealthy[/red]"
        }
        
        for service_name, service_data in services.items():
            status = service_data.get("status", "unknown")
            status_display = service_status_map.get(status, status)
            table.add_row(service_name.title(), status_display)
        
        # Add system metrics
        system = self._health_data.get("system", {})
        if system:
            table.add_row("CPU", f"{system.get('cpu_usage', 0)}%")
            table.add_row("Memory", f"{system.get('memory_usage', 0)}%")
            table.add_row("Disk", f"{system.get('disk_usage', 0)}%")
        
        return console.render(table)
    
    def _render_bot_roster(self) -> str:
        """Render bot roster widget with SSL counter and status color coding."""
        console = Console(force_terminal=True, width=60)

        bots = self._bots_data
        total = len(bots)
        total_pages = max(1, (total - 1) // self._bot_page_size + 1) if total else 1
        page = min(self._bot_page, total_pages - 1) if total else 0

        # Count SSL
        ssl_count = sum(1 for b in bots if b.get("ssl_counter", 0) > 0)

        title = f"Bot Roster (Page {page + 1}/{total_pages}) [{total} total, SSL: {ssl_count}]"
        table = Table(title=title, show_header=True, box=None)
        table.add_column("Bot", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("SSL", style="yellow")

        status_color_map = {
            "running": "[green]Running[/green]",
            "active": "[green]Active[/green]",
            "stopped": "[red]Stopped[/red]",
            "idle": "[yellow]Idle[/yellow]",
            "error": "[red]Error[/red]",
        }

        if not bots:
            table.add_row("No bots", "N/A", "N/A")
        else:
            start = page * self._bot_page_size
            end = start + self._bot_page_size
            for bot in bots[start:end]:
                name = bot.get("name", bot.get("id", "Unknown"))
                status = bot.get("status", "unknown")
                status_display = status_color_map.get(status.lower(), status)
                ssl_counter = bot.get("ssl_counter", 0)
                table.add_row(name, status_display, str(ssl_counter))

        rendered = console.render(table)
        return rendered + "\n[N]ext / [P]rev page" if total > self._bot_page_size else rendered
    
    def _render_sync_status(self) -> str:
        """Render sync status widget."""
        console = Console(force_terminal=True, width=60)
        
        table = Table(title="HMM Sync Status", show_header=True, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        sync = self._sync_data
        if not sync:
            table.add_row("Status", "Not connected")
            table.add_row("Last Sync", "N/A")
        else:
            table.add_row("Status", sync.get("status", "unknown"))
            table.add_row("Model", sync.get("model_name", "N/A"))
            table.add_row("Last Update", sync.get("last_trained", "N/A"))
        
        return console.render(table)
    
    def _render_trade_monitor(self) -> str:
        """Render trade monitor widget."""
        console = Console(force_terminal=True, width=60)
        
        table = Table(title="Recent Trades", show_header=True, box=None)
        table.add_column("Symbol", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Profit", style="green")
        
        trades = self._trades_data
        if not trades:
            table.add_row("No trades", "N/A", "N/A")
        else:
            for trade in trades[:10]:  # Limit to 10
                symbol = trade.get("symbol", "N/A")
                trade_type = trade.get("type", "N/A")
                profit = trade.get("profit", 0)
                profit_str = f"${profit:.2f}" if profit else "$0.00"
                profit_style = "green" if profit and profit > 0 else "red" if profit and profit < 0 else "white"
                table.add_row(symbol, trade_type, f"[{profit_style}]{profit_str}[/{profit_style}]")
        
        return console.render(table)

    def _render_session_state(self) -> str:
        """Render session state widget with canonical session, Tilt, next boundary, and SVSS."""
        console = Console(force_terminal=True, width=60)

        session = self._session_data
        svss = self._svss_data

        table = Table(title="Session State", show_header=True, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        if session.get("error"):
            table.add_row("Status", f"[red]Error: {session.get('error')}[/red]")
        else:
            table.add_row("Canonical Session", session.get("canonical_session", session.get("current_window", "Unknown")))
            table.add_row("Tilt State", session.get("tilt_state", "N/A"))
            table.add_row("Next Boundary", session.get("time_to_next_boundary", session.get("next_window", "N/A")))

            # SVSS values
            if svss and not svss.get("error"):
                svss_r = svss.get("r", "N/A")
                svss_vol = svss.get("volatility", "N/A")
                table.add_row("SVSS R", str(svss_r))
                table.add_row("SVSS Vol", str(svss_vol))
            else:
                table.add_row("SVSS R", svss.get("error") if svss else "N/A")
                table.add_row("SVSS Vol", "N/A")

        return console.render(table)

    def _render_killswitch_status(self) -> str:
        """Render kill switch status widget with ACTIVE/INACTIVE and last triggered."""
        console = Console(force_terminal=True, width=60)

        table = Table(title="Kill Switch Status", show_header=True, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        ks = self._killswitch_data
        if ks.get("error"):
            table.add_row("Status", f"[red]Error[/red]")
            table.add_row("Message", ks.get("error"))
        else:
            state = ks.get("state", "UNKNOWN")
            style = "[green]ACTIVE[/green]" if state == "ACTIVE" else "[red]INACTIVE[/red]"
            table.add_row("State", style)
            table.add_row("Last Triggered", ks.get("last_triggered", "Never"))
            table.add_row("Trigger Count", str(ks.get("trigger_count", 0)))

        return console.render(table)

    def _render_node_connectivity(self) -> str:
        """Render node connectivity widget with T1->T2 latency and MT5 status."""
        console = Console(force_terminal=True, width=60)

        table = Table(title="Node Connectivity", show_header=True, box=None)
        table.add_column("Node", style="cyan")
        table.add_column("Status", style="magenta")

        node_conn = self._node_connectivity_data
        if node_conn.get("error"):
            table.add_row("T1->T2", f"[red]Error[/red]")
            table.add_row("MT5 Terminal", node_conn.get("error"))
        else:
            # T1->T2 latency
            latency = node_conn.get("t1_t2_latency", node_conn.get("latency", "N/A"))
            latency_str = f"{latency}ms" if isinstance(latency, (int, float)) else str(latency)
            latency_style = "[green]" + latency_str + "[/green]"
            if isinstance(latency, (int, float)):
                if latency > 100:
                    latency_style = "[red]" + latency_str + "[/red]"
                elif latency > 50:
                    latency_style = "[yellow]" + latency_str + "[/yellow]"
            table.add_row("T1->T2 Latency", latency_style)

            # MT5 terminal status
            mt5_status = node_conn.get("mt5_terminal", node_conn.get("mt5_status", "Unknown"))
            mt5_style = "[green]" + mt5_status + "[/green]" if mt5_status == "Connected" else "[yellow]" + mt5_status + "[/yellow]"
            table.add_row("MT5 Terminal", mt5_style)

        return console.render(table)

    def _render_performance_ticker(self) -> str:
        """Render performance ticker with session P&L, daily P&L, and open positions."""
        console = Console(force_terminal=True, width=60)

        table = Table(title="Performance Ticker", show_header=True, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        perf = self._performance_data
        if perf.get("error"):
            table.add_row("Session P&L", f"[red]Error[/red]")
            table.add_row("Daily P&L", perf.get("error"))
            table.add_row("Open Positions", "N/A")
        else:
            # Session P&L
            session_pnl = perf.get("session_pnl", perf.get("sessionPnL", 0))
            session_pnl_str = f"${session_pnl:.2f}" if isinstance(session_pnl, (int, float)) else str(session_pnl)
            session_pnl_style = "[green]" + session_pnl_str + "[/green]" if isinstance(session_pnl, (int, float)) and session_pnl >= 0 else "[red]" + session_pnl_str + "[/red]"
            table.add_row("Session P&L", session_pnl_style)

            # Daily P&L
            daily_pnl = perf.get("daily_pnl", perf.get("dailyPnL", 0))
            daily_pnl_str = f"${daily_pnl:.2f}" if isinstance(daily_pnl, (int, float)) else str(daily_pnl)
            daily_pnl_style = "[green]" + daily_pnl_str + "[/green]" if isinstance(daily_pnl, (int, float)) and daily_pnl >= 0 else "[red]" + daily_pnl_str + "[/red]"
            table.add_row("Daily P&L", daily_pnl_style)

            # Open positions
            open_pos = perf.get("open_positions", perf.get("openPositions", 0))
            table.add_row("Open Positions", str(open_pos))

        return console.render(table)

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
    
    def action_refresh(self) -> None:
        """Force refresh data."""
        self.refresh_data()
    
    def action_bots_view(self) -> None:
        """Switch to bots view."""
        self._current_view = "bots"
        self.refresh_data()
    
    def action_trades_view(self) -> None:
        """Switch to trades view."""
        self._current_view = "trades"
        self.refresh_data()
    
    def action_sync_view(self) -> None:
        """Switch to sync view."""
        self._current_view = "sync"
        self.refresh_data()
    
    def action_health_view(self) -> None:
        """Switch to health view."""
        self._current_view = "health"
        self.refresh_data()
    
    def bind(self) -> list:
        """Define key bindings."""
        return [
            ("q", "quit", "Quit"),
            ("r", "refresh", "Refresh"),
            ("b", "bots_view", "Bots"),
            ("t", "trades_view", "Trades"),
            ("s", "sync_view", "Sync"),
            ("h", "health_view", "Health"),
        ]


# ============== Main Entry Point ==============

if __name__ == "__main__":
    app = QuantMindTUI()
    app.run()
