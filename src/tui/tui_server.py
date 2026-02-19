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
        grid-size: 2 2;
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
        content-type: text;
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
        
        # Current view
        self._current_view = "dashboard"
    
    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield Header(show_clock=True)
        
        with Horizontal():
            with Vertical():
                yield Static(id="system-status", classes="widget")
                yield Static(id="bot-status", classes="widget")
            with Vertical():
                yield Static(id="sync-status", classes="widget")
                yield Static(id="trade-monitor", classes="widget")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.api_client = QuantMindAPIClient(self.base_url=self.api_base_url)
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
            
            self._health_data = health
            self._bots_data = bots
            self._sync_data = sync
            self._trades_data = trades
        
        # Update widgets
        self.update_widgets()
    
    def update_widgets(self) -> None:
        """Update all widget displays."""
        # System Status
        system_widget = self.query_one("#system-status", Static)
        system_widget.update(self._render_system_status())
        
        # Bot Status
        bot_widget = self.query_one("#bot-status", Static)
        bot_widget.update(self._render_bot_status())
        
        # Sync Status
        sync_widget = self.query_one("#sync-status", Static)
        sync_widget.update(self._render_sync_status())
        
        # Trade Monitor
        trade_widget = self.query_one("#trade-monitor", Static)
        trade_widget.update(self._render_trade_monitor())
    
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
    
    def _render_bot_status(self) -> str:
        """Render bot status widget."""
        console = Console(force_terminal=True, width=60)
        
        table = Table(title="Bot Status", show_header=True, box=None)
        table.add_column("Bot", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Signal", style="green")
        
        bots = self._bots_data
        if not bots:
            table.add_row("No bots", "N/A", "N/A")
        else:
            for bot in bots[:10]:  # Limit to 10
                name = bot.get("name", bot.get("id", "Unknown"))
                status = bot.get("status", "unknown")
                signal = str(bot.get("signalStrength", "N/A"))
                table.add_row(name, status, signal)
        
        return console.render(table)
    
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
