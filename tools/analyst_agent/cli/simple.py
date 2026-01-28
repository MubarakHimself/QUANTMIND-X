"""
Interactive CLI for QuantMindX Analyst Agent

Simple, focused interface for strategy extraction from NPRD files.
MCP is optional and configured separately.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from kb.client import ChromaKBClient
from key_manager import get_key_manager
from agent import create_analyst_agent
from agent.mcp import create_mcp_client, get_quantmindx_mcp_servers

console = Console()


class SimpleAnalystCLI:
    """Simple CLI for Analyst Agent."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.nprd_dir = self.project_root / "outputs" / "videos"
        self.strategy_dir = self.project_root / "docs" / "strategies"
        self.kb_client = None
        self.api_key = None
        self.agent = None
        self.mcp_client = None

    def startup(self) -> bool:
        """Quick startup."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]QuantMindX Analyst Agent[/bold cyan]\n"
            "[dim]Strategy Parameter Extractor[/dim]",
            box=box.DOUBLE,
            padding=(1, 2)
        ))

        # KB
        self._init_kb()

        # API Key
        self._init_api()

        # MCP (optional, silent)
        self._init_mcp()

        # Agent
        self._init_agent()

        # Status
        self._show_status()

        return True

    def _init_kb(self):
        """Initialize KB."""
        try:
            self.kb_client = ChromaKBClient()
            collections = self.kb_client.list_collections()
            console.print(f"[green]✓ KB:[/green] {len(collections)} collections ({sum(self.kb_client.get_collection_stats(c).get('count', 0) for c in collections)} docs)")
        except Exception as e:
            console.print(f"[yellow]⚠ KB:[/yellow] {e}")

    def _init_api(self):
        """Get API key."""
        key_manager = get_key_manager()
        stored_keys = key_manager.list_keys()
        env_key = os.getenv("OPENROUTER_API_KEY")

        if stored_keys or env_key:
            # Use first stored key or env key
            if stored_keys:
                self.api_key = stored_keys[0].key
                console.print(f"[green]✓ API Key:[/green] {stored_keys[0].name}")
            elif env_key:
                self.api_key = env_key
                console.print("[green]✓ API Key:[/green] Environment")
        else:
            console.print("[yellow]⚠ No API key - limited mode[/yellow]")

    def _init_mcp(self):
        """Initialize MCP (optional, silent)."""
        try:
            servers = get_quantmindx_mcp_servers()
            self.mcp_client = create_mcp_client(servers)
            # Try to connect silently
            results = self.mcp_client.connect_all()
            connected = sum(1 for v in results.values() if v)
            if connected > 0:
                console.print(f"[dim]✓ MCP: {connected} server(s)[/dim]")
        except Exception:
            # MCP is optional, fail silently
            self.mcp_client = None

    def _init_agent(self):
        """Initialize agent."""
        if self.api_key:
            try:
                self.agent = create_analyst_agent(
                    api_key=self.api_key,
                    kb_client=self.kb_client,
                    mcp_client=self.mcp_client
                )
                console.print("[green]✓ Agent Ready[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Agent:[/yellow] {e}")
        else:
            console.print("[yellow]⚠ Agent: No API key[/yellow]")

    def _show_status(self):
        """Show quick status."""
        nprd_files = list(self.nprd_dir.rglob("*.json"))
        nprd_files = [f for f in nprd_files if "chunk" not in f.name]
        strategy_files = list(self.strategy_dir.glob("*.json")) if self.strategy_dir.exists() else []

        console.print(Panel(
            f"[cyan]NPRD Files:[/cyan] {len(nprd_files)}\n"
            f"[cyan]Strategies:[/cyan] {len(strategy_files)}\n"
            f"[cyan]API Key:[/cyan] {'✓' if self.api_key else '✗'}\n"
            f"[cyan]Agent:[/cyan] {'✓' if self.agent else '✗'}",
            title="[bold green]Ready[/bold green]",
            border_style="green"
        ))

    def run(self):
        """Run main menu."""
        while True:
            console.print("\n[bold cyan]Menu:[/bold cyan]")
            console.print("  [1] Scan NPRD files")
            console.print("  [2] Search KB")
            console.print("  [3] Extract Strategy")
            console.print("  [4] View Strategies")
            console.print("  [5] Agent Info")
            console.print("  [6] Chat")
            console.print("  [0] Exit")

            choice = Prompt.ask("\nSelect", choices=["0", "1", "2", "3", "4", "5", "6"], default="0")

            if choice == "1":
                self._scan_nprd()
            elif choice == "2":
                self._search_kb()
            elif choice == "3":
                self._extract_strategy()
            elif choice == "4":
                self._view_strategies()
            elif choice == "5":
                self._agent_info()
            elif choice == "6":
                self._chat()
            elif choice == "0":
                console.print("\n[yellow]Goodbye![/yellow]")
                break

    def _scan_nprd(self):
        """Scan NPRD files."""
        nprd_files = []
        for json_file in self.nprd_dir.rglob("*.json"):
            if "chunk" not in json_file.name:
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        has_content = bool(data.get("transcript") or data.get("ocr_text"))
                        nprd_files.append({
                            "name": json_file.name,
                            "path": json_file,
                            "has_content": has_content,
                            "transcript": bool(data.get("transcript")),
                            "ocr": bool(data.get("ocr_text")),
                            "keywords": len(data.get("keywords", []))
                        })
                except:
                    pass

        if not nprd_files:
            console.print("[yellow]No NPRD files[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Content", style="green")
        table.add_column("Details", style="dim")

        for nprd in nprd_files:
            details = []
            if nprd["transcript"]:
                details.append("Transcript")
            if nprd["ocr"]:
                details.append("OCR")
            if nprd["keywords"]:
                details.append(f"{nprd['keywords']} kw")

            content_status = "[green]✓[/green]" if nprd["has_content"] else "[red]✗[/red]"
            table.add_row(nprd["name"][:50], content_status, ", ".join(details))

        console.print(table)
        console.print(f"\n[dim]Total: {len(nprd_files)} files[/dim]")

    def _search_kb(self):
        """Search KB."""
        if not self.kb_client:
            console.print("[red]KB not available[/red]")
            return

        query = Prompt.ask("\nSearch query")
        if not query:
            return

        with console.status("[cyan]Searching...[/cyan]"):
            results = self.kb_client.search(query, collection="analyst_kb", n=5)

        if not results:
            console.print("[yellow]No results[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Title", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Preview", style="dim")

        for r in results:
            table.add_row(
                r.get("title", "Untitled")[:40],
                f"{r.get('score', 0):.2f}",
                r.get("preview", "")[:60]
            )

        console.print(table)

    def _extract_strategy(self):
        """Extract strategy from NPRD."""
        if not self.agent:
            console.print("[red]Agent not available[/red]")
            return

        nprd_files = list(self.nprd_dir.rglob("*.json"))
        nprd_files = [f for f in nprd_files if "chunk" not in f.name]

        if not nprd_files:
            console.print("[yellow]No NPRD files[/yellow]")
            return

        console.print("\n[bold]Select NPRD:[/bold]")
        for i, f in enumerate(nprd_files[:10], 1):
            console.print(f"  [{i}] {f.name}")

        choice = Prompt.ask("\nSelect", choices=[str(i) for i in range(1, min(11, len(nprd_files) + 1))])
        nprd_path = nprd_files[int(choice) - 1]

        console.print(f"\n[cyan]Extracting:[/cyan] {nprd_path.name}")

        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.strategy_dir / f"{nprd_path.stem}_strategy.json"

        try:
            with console.status("[cyan]Extracting entry/exit logic...[/cyan]"):
                strategy = self.agent.extract_from_nprd(nprd_path, output_path)

            console.print(f"[green]✓ Saved to:[/green] {output_path}")

            if "error" in strategy:
                console.print(f"[red]Error: {strategy['error']}[/red]")
            else:
                console.print("\n[bold cyan]Extracted:[/bold cyan]")
                if strategy.get("entry_logic"):
                    console.print("  [green]✓[/green] Entry Logic")
                if strategy.get("exit_logic"):
                    console.print("  [green]✓[/green] Exit Logic")
                if strategy.get("parameters"):
                    console.print("  [green]✓[/green] Parameters")

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

    def _view_strategies(self):
        """View extracted strategies."""
        if not self.strategy_dir.exists():
            console.print("[yellow]No strategies directory[/yellow]")
            return

        strategy_files = list(self.strategy_dir.glob("*.json"))

        if not strategy_files:
            console.print("[yellow]No strategies found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Size", style="dim")

        for f in strategy_files:
            size = f.stat().st_size
            table.add_row(f.name, f"{size} bytes")

        console.print(table)

    def _agent_info(self):
        """Show agent info."""
        if not self.agent:
            console.print("[yellow]Agent not available[/yellow]")
            return

        info = self.agent.get_info()

        console.print("\n[bold cyan]AnalystAgent - Strategy Extractor[/bold cyan]")
        console.print(f"  [dim]Model:[/dim] {info['model']}")
        console.print(f"  [dim]Provider:[/dim] OpenRouter")
        console.print(f"  [dim]Streaming:[/dim] Enabled")
        console.print(f"  [dim]Memory:[/dim] ChromaDB")

        console.print("\n[bold green]✓ Responsible For:[/bold green]")
        for r in info.get('responsibilities', []):
            console.print(f"  • {r}")

        console.print("\n[bold red]✗ NOT Responsible For:[/bold red]")
        for nr in info.get('not_responsible_for', []):
            console.print(f"  • {nr}")

        console.print(f"\n[dim]KB Collection: {info.get('kb_collection', 'N/A')}[/dim]")

    def _chat(self):
        """Chat with agent with streaming support."""
        if not self.agent:
            console.print("[red]Agent not available[/red]")
            return

        console.print("\n[bold cyan]Chat Mode[/bold cyan]")
        console.print("[dim]Ask about strategy extraction, entry/exit logic, etc.[/dim]")
        console.print("[dim]Type 'quit' to exit[/dim]")
        console.print("[dim]Model: qwen/qwen3-vl-30b-a3b-thinking | Streaming: Enabled[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if user_input.lower() in ("quit", "exit", "q"):
                    break

                if not user_input.strip():
                    continue

                console.print(f"\n[green]Agent:[/green]")
                console.print("  ", end="")  # Indent for streaming

                # Stream response
                for chunk in self.agent.stream(user_input, use_kb=True):
                    console.print(chunk, end="")

                console.print("\n")  # New line after response

            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting chat[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()[:500]}[/dim]")


def main():
    """Main entry."""
    cli = SimpleAnalystCLI()

    if not cli.startup():
        return

    cli.run()


if __name__ == "__main__":
    main()
