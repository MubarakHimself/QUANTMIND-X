"""
QuantMind CLI

Command-line interface for QuantMind trading system management.
Provides commands for status, bot management, sync, trades, and health checks.
"""

import asyncio
import os

import click
from rich.console import Console
from rich.table import Table


# ============== Client Helper ==============

def get_api_client() -> "QuantMindAPIClient":
    """Get API client instance (imported from tui_server to avoid duplication)."""
    from src.tui.tui_server import QuantMindAPIClient
    base_url = os.getenv("QUANTMIND_API_URL", "http://localhost:8000")
    return QuantMindAPIClient(base_url=base_url)


# ============== CLI Groups ==============

@click.group()
@click.option("--vps", type=click.Choice(["trading", "contabo"]), help="VPS name")
@click.pass_context
def cli(ctx, vps):
    """QuantMind CLI - Trading system management."""
    ctx.ensure_object(dict)
    ctx.obj["vps"] = vps or os.getenv("QUANTMIND_VPS_NAME", "Trading VPS")
    ctx.obj["console"] = Console()


@cli.group()
def status():
    """Check overall system status."""
    pass


@status.command("trading")
def status_trading():
    """Check trading VPS status."""
    console = Console()
    console.print("[bold cyan]Trading VPS Status[/bold cyan]")
    
    async def check():
        client = get_api_client()
        async with client:
            health = await client.get_health()
            return health
    
    result = asyncio.run(check())
    
    table = Table(title="Service Health", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Latency", style="green")
    
    services = result.get("services", {})
    for name, data in services.items():
        status = data.get("status", "unknown")
        latency = data.get("latency_ms", "N/A")
        status_str = f"[green]{status}[/green]" if status == "healthy" else f"[yellow]{status}[/yellow]" if status == "degraded" else f"[red]{status}[/red]"
        table.add_row(name.title(), status_str, f"{latency}ms" if latency else "N/A")
    
    console.print(table)


@status.command("contabo")
def status_contabo():
    """Check Contabo VPS status."""
    console = Console()
    console.print("[bold cyan]Contabo VPS Status[/bold cyan]")
    console.print("[yellow]Contabo status check not yet implemented[/yellow]")


@cli.group()
def bots():
    """Bot management commands."""
    pass


@bots.command("list")
def bots_list():
    """List all bots."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.get_bots()
    
    result = asyncio.run(check())
    
    if not result:
        console.print("[yellow]No bots found[/yellow]")
        return
    
    table = Table(title="Bot List", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Signal", style="yellow")
    
    for bot in result:
        table.add_row(
            bot.get("id", "N/A"),
            bot.get("name", "N/A"),
            bot.get("status", "N/A"),
            str(bot.get("signalStrength", "N/A"))
        )
    
    console.print(table)


@bots.command("start")
@click.argument("bot_id")
def bots_start(bot_id):
    """Start a bot by ID."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.start_bot(bot_id)
    
    result = asyncio.run(check())
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
    else:
        console.print(f"[green]Bot {bot_id} started successfully[/green]")


@bots.command("stop")
@click.argument("bot_id")
def bots_stop(bot_id):
    """Stop a bot by ID."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.stop_bot(bot_id)
    
    result = asyncio.run(check())
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
    else:
        console.print(f"[green]Bot {bot_id} stopped successfully[/green]")


@bots.command("lifecycle")
def bots_lifecycle():
    """Show bot lifecycle status."""
    console = Console()
    console.print("[yellow]Bot lifecycle status not yet implemented[/yellow]")


@cli.group()
def sync():
    """Sync status commands."""
    pass


@sync.command("status")
def sync_status():
    """Check HMM sync status."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.get_sync_status()
    
    result = asyncio.run(check())
    
    if not result:
        console.print("[yellow]No sync data available[/yellow]")
        return
    
    table = Table(title="HMM Sync Status", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in result.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)


@sync.command("hmm")
def sync_hmm():
    """Show HMM model status."""
    sync_status()  # Reuse the same command


@sync.command("data")
def sync_data():
    """Show data sync status."""
    console = Console()
    console.print("[yellow]Data sync status not yet implemented[/yellow]")


@cli.group()
def trades():
    """Trade commands."""
    pass


@trades.command("recent")
@click.option("--limit", default=10, help="Number of trades to show")
def trades_recent(limit):
    """Show recent trades."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.get_recent_trades(limit=limit)
    
    result = asyncio.run(check())
    
    if not result:
        console.print("[yellow]No trades found[/yellow]")
        return
    
    table = Table(title="Recent Trades", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Profit", style="green")
    table.add_column("Status", style="yellow")
    
    for trade in result:
        profit = trade.get("profit", 0)
        profit_str = f"[green]+${profit:.2f}[/green]" if profit and profit > 0 else f"[red]-${abs(profit):.2f}[/red]" if profit else "$0.00"
        table.add_row(
            trade.get("symbol", "N/A"),
            trade.get("type", "N/A"),
            profit_str,
            trade.get("status", "N/A")
        )
    
    console.print(table)


@trades.command("today")
def trades_today():
    """Show today's trades."""
    trades_recent(limit=20)


@trades.command("bot")
@click.argument("bot_id")
def trades_bot(bot_id):
    """Show trades for a specific bot."""
    console = Console()
    console.print(f"[yellow]Showing trades for bot {bot_id}...[/yellow]")
    # TODO: Implement bot-specific trades


@cli.command("health")
@click.option("--service", type=click.Choice(["api", "mt5", "database", "redis", "prometheus"]), help="Specific service to check")
def health(service):
    """Check health status of services."""
    console = Console()
    
    async def check():
        client = get_api_client()
        async with client:
            return await client.get_health()
    
    result = asyncio.run(check())
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return
    
    if service:
        # Check specific service
        services = result.get("services", {})
        if service in services:
            data = services[service]
            table = Table(title=f"{service.title()} Health", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Status", data.get("status", "N/A"))
            if data.get("latency_ms"):
                table.add_row("Latency", f"{data['latency_ms']}ms")
            if data.get("message"):
                table.add_row("Message", data.get("message"))
            
            console.print(table)
        else:
            console.print(f"[yellow]Service {service} not found[/yellow]")
    else:
        # Show all services
        services = result.get("services", {})
        
        table = Table(title="Health Status", show_header=True)
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Latency", style="green")
        
        for name, data in services.items():
            status = data.get("status", "unknown")
            latency = data.get("latency_ms", "N/A")
            status_str = f"[green]{status}[/green]" if status == "healthy" else f"[yellow]{status}[/yellow]" if status == "degraded" else f"[red]{status}[/red]"
            table.add_row(name.title(), status_str, f"{latency}ms" if latency else "N/A")
        
        console.print(table)
        
        # Show system metrics
        system = result.get("system", {})
        if system:
            console.print()
            console.print("[bold]System Metrics:[/bold]")
            console.print(f"  CPU: {system.get('cpu_usage', 0)}%")
            console.print(f"  Memory: {system.get('memory_usage', 0)}%")
            console.print(f"  Disk: {system.get('disk_usage', 0)}%")


# ============== Main Entry Point ==============

if __name__ == "__main__":
    cli()
