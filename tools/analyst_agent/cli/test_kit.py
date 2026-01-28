#!/usr/bin/env python3
"""
Quick test script for Analyst Agent CLI.
Tests KB connection and shows available options.
"""

import sys
import os
from pathlib import Path

# Get correct paths regardless of where script is run from
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent

# Add to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root
os.chdir(project_root)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from kb.client import ChromaKBClient
from utils.config import get_config, Config

console = Console()


def test_kb_connection():
    """Test knowledge base connection."""
    console.print("\n[bold cyan]Testing Knowledge Base Connection[/bold cyan]")

    try:
        kb = ChromaKBClient()
        collections = kb.list_collections()

        console.print(f"[green]✓ ChromaDB connected:[/green] {kb.db_path}")

        # Show collections table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Collection", style="cyan")
        table.add_column("Documents", justify="right", style="green")

        for col_name in collections:
            stats = kb.get_collection_stats(col_name)
            count = stats.get("count", 0)
            table.add_row(col_name, str(count))

        console.print(table)

        # Test search
        console.print("\n[bold]Testing semantic search...[/bold]")
        test_queries = ["ORB strategy", "Kelly criterion", "position sizing"]

        for query in test_queries:
            results = kb.search(query, collection="mql5_knowledge", n=1)
            if results:
                console.print(f"  [cyan]{query}[/cyan] → [green]{results[0]['title'][:40]}...[/green] (score: {results[0]['score']:.2f})")
            else:
                console.print(f"  [cyan]{query}[/cyan] → [yellow]No results[/yellow]")

        return True

    except Exception as e:
        console.print(f"[red]✗ KB connection failed:[/red] {e}")
        return False


def show_nprd_files():
    """Show available NPRD files."""
    console.print("\n[bold cyan]Scanning NPRD Files[/bold cyan]")

    nprd_dir = project_root / "outputs" / "videos"

    import json
    nprd_files = list(nprd_dir.rglob("*.json"))

    if not nprd_files:
        console.print("[yellow]No NPRD files found[/yellow]")
        return

    nprd_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", width=3)
    table.add_column("File Name", style="cyan")
    table.add_column("Size", justify="right", width=10)

    for i, f in enumerate(nprd_files[:10], 1):
        size_kb = f"{f.stat().st_size / 1024:.1f} KB"
        table.add_row(str(i), f.name[:45], size_kb)

    console.print(table)
    console.print(f"\n[dim]Found {len(nprd_files)} NPRD files total[/dim]")


def show_commands():
    """Show available commands."""
    console.print("\n[bold cyan]Available Commands[/bold cyan]")
    console.print("""
  [green]python3 tools/analyst_agent/cli/main.py interactive[/green]
      Launch interactive test environment (requires terminal)

  [green]python3 tools/analyst_agent/cli/test_kit.py[/green]
      Run this quick test (KB connection + file scan)

  [green]export OPENROUTER_API_KEY="sk-..."[/green]
      Set API key for TRD generation

  [green]python3 tools/analyst_agent/cli/main.py stats[/green]
      Show knowledge base statistics
    """)


def main():
    """Main test function."""
    console.print(Panel.fit(
        "[bold cyan]QuantMindX Analyst Agent[/bold cyan]\n"
        "[dim]Quick Test Kit[/dim]",
        box=box.DOUBLE,
        padding=(1, 2)
    ))

    # Test KB
    kb_ok = test_kb_connection()

    # Show NPRD files
    if kb_ok:
        show_nprd_files()

    # Show commands
    show_commands()

    console.print("\n[bold green]✓ Test complete![/bold green]")
    console.print("[dim]To use the interactive CLI, run:[/dim]")
    console.print("  [cyan]python3 tools/analyst_agent/cli/main.py interactive[/cyan]")


if __name__ == "__main__":
    main()
