"""
Interactive CLI for QuantMindX Analyst Agent - Centralized Version

Uses the new unified agent framework with:
- BaseAgent foundation
- Skills and tools
- MCP access
- LangMem integration
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb.client import ChromaKBClient
from key_manager import get_key_manager
from agent import create_analyst_agent, get_analyst_agent
from agent.mcp import create_mcp_client, get_quantmindx_mcp_servers
from chat import print_assistant_message, print_user_message

console = Console()


class InteractiveAnalystCLI:
    """
    Interactive CLI for Analyst Agent using unified agent framework.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.nprd_dir = self.project_root / "outputs" / "videos"
        self.trd_dir = self.project_root / "docs" / "trds"

        # Components
        self.kb_client: Optional[ChromaKBClient] = None
        self.api_key: Optional[str] = None
        self.agent = None
        self.mcp_client = None

    def startup(self) -> bool:
        """Run startup sequence."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]QuantMindX Analyst Agent[/bold cyan]\n"
            "[dim]Interactive CLI - Unified Agent Framework[/dim]",
            box=box.DOUBLE,
            padding=(1, 2)
        ))

        # Step 1: Knowledge Base
        console.print("\n[bold yellow]Step 1: Knowledge Base[/bold yellow]")
        if not self._init_kb():
            return False

        # Step 2: API Key
        console.print("\n[bold yellow]Step 2: API Configuration[/bold yellow]")
        if not self._init_api():
            console.print("[yellow]Continuing without API (limited mode)[/yellow]")

        # Step 3: MCP Servers
        console.print("\n[bold yellow]Step 3: MCP Servers[/bold yellow]")
        self._init_mcp()

        # Step 4: Initialize Agent
        console.print("\n[bold yellow]Step 4: Initialize Agent[/bold yellow]")
        if not self._init_agent():
            console.print("[yellow]Agent not available[/yellow]")

        # Show status
        self._show_status()

        return True

    def _init_kb(self) -> bool:
        """Initialize knowledge base."""
        try:
            self.kb_client = ChromaKBClient()
            collections = self.kb_client.list_collections()

            console.print(f"[green]ChromaDB:[/green] {self.kb_client.db_path}")

            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Collection", style="cyan")
            table.add_column("Documents", justify="right", style="green")

            for col in collections:
                stats = self.kb_client.get_collection_stats(col)
                table.add_row(col, str(stats.get("count", 0)))

            console.print(table)

            # Create analyst_kb if needed
            if "analyst_kb" not in collections and "mql5_knowledge" in collections:
                if Confirm.ask("\nCreate analyst_kb from mql5_knowledge?"):
                    with console.status("[cyan]Creating analyst_kb...[/cyan]"):
                        stats = self.kb_client.create_analyst_kb()
                    console.print(f"[green]Created:[/green] {stats['included']} docs")

            return True

        except Exception as e:
            console.print(f"[red]KB error:[/red] {e}")
            return False

    def _init_api(self) -> bool:
        """Initialize API key."""
        key_manager = get_key_manager()
        stored_keys = key_manager.list_keys()
        env_key = os.getenv("OPENROUTER_API_KEY")

        if stored_keys or env_key:
            console.print("[dim]Found API key sources:[/dim]")
            if stored_keys:
                console.print(f"  [green]✓[/green] {len(stored_keys)} stored")
            if env_key:
                console.print(f"  [green]✓[/green] Environment")

            if Confirm.ask("\nSelect existing key?", default=True):
                selected = key_manager.select_key_interactive()
                if selected:
                    self.api_key = selected.key
                    console.print(f"[green]✓ Using:[/green] {selected.get_masked_key()}")
                    return True

        if Confirm.ask("\nAdd new API key?"):
            new_key = key_manager.add_key_interactive()
            if new_key:
                self.api_key = new_key.key
                console.print(f"[green]✓ Added:[/green] {new_key.get_masked_key()}")
                return True

        return False

    def _init_mcp(self):
        """Initialize MCP client."""
        try:
            servers = get_quantmindx_mcp_servers()
            self.mcp_client = create_mcp_client(servers)

            # Try to connect
            results = self.mcp_client.connect_all()
            connected = sum(1 for v in results.values() if v)

            console.print(f"[green]MCP:[/green] {connected}/{len(servers)} servers connected")

            # Show available tools
            tools = self.mcp_client.list_tools()
            if tools:
                console.print(f"[dim]  Tools: {len(tools)} available[/dim]")

        except Exception as e:
            console.print(f"[dim]MCP not available: {e}[/dim]")

    def _init_agent(self) -> bool:
        """Initialize the analyst agent."""
        if not self.api_key:
            console.print("[yellow]No API key - agent in limited mode[/yellow]")
            return False

        try:
            self.agent = create_analyst_agent(
                api_key=self.api_key,
                kb_client=self.kb_client,
                mcp_client=self.mcp_client
            )

            # Show capabilities
            info = self.agent.get_info()
            console.print(f"[green]Agent:[/green] {info['name']}")
            console.print(f"[dim]  Model: {info['model']}[/dim]")

            capabilities = info['capabilities']
            console.print(f"[dim]  Skills: {len(capabilities['skills'])}[/dim]")
            console.print(f"[dim]  Tools: {len(capabilities['tools'])}[/dim]")
            if capabilities['mcp_tools']:
                console.print(f"[dim]  MCP: {len(capabilities['mcp_tools'])}[/dim]")

            return True

        except Exception as e:
            console.print(f"[red]Agent init error:[/red] {e}")
            return False

    def _show_status(self):
        """Show system status."""
        nprd_files = list(self.nprd_dir.rglob("*.json"))
        nprd_main = [f for f in nprd_files if "chunk" not in f.name]
        trd_files = list(self.trd_dir.glob("*.md")) if self.trd_dir.exists() else []

        console.print("\n")
        console.print(Panel(
            f"[bold cyan]NPRD Files:[/bold cyan] {len(nprd_main)}\n"
            f"[bold cyan]TRD Files:[/bold cyan] {len(trd_files)}\n"
            f"[bold cyan]API Key:[/bold cyan] {'[green]Set[/green]' if self.api_key else '[yellow]No[/yellow]'}\n"
            f"[bold cyan]Agent:[/bold cyan] {'[green]Ready[/green]' if self.agent else '[yellow]No[/yellow]'}",
            title="[bold green]Status[/bold green]",
            border_style="green"
        ))

    def run(self):
        """Run interactive loop."""
        while True:
            console.print("\n[bold cyan]Menu:[/bold cyan]")
            console.print("  [1] Scan NPRD files")
            console.print("  [2] Search KB")
            console.print("  [3] Extract Strategy (Entry/Exit Logic)")
            console.print("  [4] View Strategies")
            console.print("  [5] Test KB")
            console.print("  [6] Agent Info")
            console.print("  [7] 💬 Chat")
            console.print("  [8] 🔑 Keys")
            console.print("  [0] Exit")

            choice = Prompt.ask("\nSelect", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"], default="0")

            try:
                if choice == "1":
                    self._scan_nprd()
                elif choice == "2":
                    self._search_kb()
                elif choice == "3":
                    self._extract_strategy()
                elif choice == "4":
                    self._view_strategies()
                elif choice == "5":
                    self._test_kb()
                elif choice == "6":
                    self._agent_info()
                elif choice == "7":
                    self._chat()
                elif choice == "8":
                    self._manage_keys()
                elif choice == "0":
                    console.print("[yellow]Goodbye![/yellow]")
                    break

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

    def _scan_nprd(self):
        """Scan NPRD files."""
        console.print("\n[bold]Scanning NPRD files...[/bold]")

        nprd_files = []
        for json_file in self.nprd_dir.rglob("*.json"):
            if "chunk" not in json_file.name:
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        nprd_files.append({
                            "name": json_file.name,
                            "path": json_file,
                            "has_transcript": bool(data.get("transcript")),
                            "has_ocr": bool(data.get("ocr_text")),
                            "keywords": len(data.get("keywords", []))
                        })
                except:
                    pass

        if not nprd_files:
            console.print("[yellow]No NPRD files found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Transcript", style="green")
        table.add_column("OCR", style="green")
        table.add_column("Keywords", justify="right", style="yellow")

        for nprd in nprd_files:
            table.add_row(
                nprd["name"][:50],
                "✓" if nprd["has_transcript"] else "✗",
                "✓" if nprd["has_ocr"] else "✗",
                str(nprd["keywords"])
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(nprd_files)} files[/dim]")

    def _search_kb(self):
        """Search knowledge base."""
        if not self.kb_client:
            console.print("[red]KB not available[/red]")
            return

        query = Prompt.ask("\nEnter search query")
        if not query:
            return

        with console.status("[cyan]Searching...[/cyan]"):
            results = self.kb_client.search(query, collection="analyst_kb", n=5)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Title", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Preview", style="dim")

        for r in results:
            table.add_row(
                r.get("title", "Untitled")[:40],
                f"{r.get('score', 0):.2f}",
                r.get("preview", "")[:60]
            )

        console.print(table)

    def _extract_strategy(self):
        """Extract strategy parameters from NPRD."""
        if not self.agent:
            console.print("[red]Agent not available[/red]")
            return

        # List NPRD files
        nprd_files = list(self.nprd_dir.rglob("*.json"))
        nprd_files = [f for f in nprd_files if "chunk" not in f.name]

        if not nprd_files:
            console.print("[yellow]No NPRD files found[/yellow]")
            return

        console.print("\n[bold]Select NPRD file to extract strategy:[/bold]")
        for i, f in enumerate(nprd_files[:10], 1):
            console.print(f"  [{i}] {f.name}")

        choice = Prompt.ask("\nSelect file", choices=[str(i) for i in range(1, min(11, len(nprd_files) + 1))])
        nprd_path = nprd_files[int(choice) - 1]

        console.print(f"\n[cyan]Extracting strategy from:[/cyan] {nprd_path.name}")

        # Create output path
        output_dir = self.project_root / "docs" / "strategies"
        output_path = output_dir / f"{nprd_path.stem}_strategy.json"

        try:
            with console.status("[cyan]Extracting entry/exit logic...[/cyan]"):
                strategy = self.agent.extract_from_nprd(nprd_path, output_path)

            # Show results
            console.print(f"[green]✓ Strategy extracted to:[/green] {output_path}")

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
        strategy_dir = self.project_root / "docs" / "strategies"

        if not strategy_dir.exists():
            console.print("[yellow]No strategies directory[/yellow]")
            return

        strategy_files = list(strategy_dir.glob("*.json"))

        if not strategy_files:
            console.print("[yellow]No strategy files found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right", style="dim")

        for f in strategy_files:
            size = f.stat().st_size
            table.add_row(f.name, f"{size} bytes")

        console.print(table)

    def _test_kb(self):
        """Test KB connection."""
        if not self.kb_client:
            console.print("[red]KB not available[/red]")
            return

        console.print("\n[bold]Testing KB...[/bold]")

        try:
            collections = self.kb_client.list_collections()
            console.print(f"[green]Collections:[/green] {len(collections)}")

            for col in collections:
                stats = self.kb_client.get_collection_stats(col)
                console.print(f"  [cyan]{col}:[/cyan] {stats.get('count', 0)} docs")

            # Test search
            results = self.kb_client.search("Kelly", collection="analyst_kb", n=1)
            console.print(f"[green]Search test:[/green] {len(results)} results")

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

    def _agent_info(self):
        """Show agent information."""
        if not self.agent:
            console.print("[yellow]Agent not available[/yellow]")
            return

        info = self.agent.get_info()

        console.print("\n[bold cyan]AnalystAgent - Strategy Parameter Extractor[/bold cyan]")
        console.print(f"  [dim]Model:[/dim] {info['model']}")

        console.print("\n[bold green]✓ Responsible For:[/bold green]")
        for resp in info.get('responsibilities', []):
            console.print(f"  • {resp}")

        console.print("\n[bold red]✗ NOT Responsible For:[/bold red]")
        for not_resp in info.get('not_responsible_for', []):
            console.print(f"  • {not_resp}")

        capabilities = info['capabilities']

        console.print("\n[bold yellow]Skills:[/bold yellow]")
        for skill in capabilities['skills']:
            console.print(f"  • {skill}")

        console.print("\n[bold yellow]Tools:[/bold yellow]")
        for tool in capabilities['tools']:
            console.print(f"  • {tool}")

    def _chat(self):
        """Chat with agent."""
        if not self.agent:
            console.print("[red]Agent not available[/red]")
            return

        console.print("\n[bold cyan]Chat Mode[/bold cyan]")
        console.print("[dim]Type 'quit' to exit[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if user_input.lower() in ("quit", "exit", "q"):
                    break

                if not user_input.strip():
                    continue

                print_user_message(user_input)

                with console.status("[cyan]Thinking...[/cyan]"):
                    response = self.agent.chat(user_input, use_kb=True)

                print_assistant_message(response)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Exiting chat[/yellow]")
                break

    def _manage_keys(self):
        """Manage API keys."""
        key_manager = get_key_manager()

        console.print("\n[bold cyan]Key Management[/bold cyan]")
        console.print("  [1] List keys")
        console.print("  [2] Add key")
        console.print("  [3] Delete key")
        console.print("  [0] Back")

        choice = Prompt.ask("\nSelect", choices=["0", "1", "2", "3"])

        if choice == "1":
            keys = key_manager.list_keys()
            if not keys:
                console.print("[yellow]No stored keys[/yellow]")
                return

            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Name", style="cyan")
            table.add_column("Key", style="yellow")
            table.add_column("Model", style="dim")

            for k in keys:
                table.add_row(k.name, k.get_masked_key(), k.model[:30])

            console.print(table)

        elif choice == "2":
            key_manager.add_key_interactive()

        elif choice == "3":
            keys = key_manager.list_keys()
            if not keys:
                return

            console.print("\n[bold]Select key to delete:[/bold]")
            for i, k in enumerate(keys, 1):
                console.print(f"  [{i}] {k.name}")

            choice = Prompt.ask("\nSelect", choices=[str(i) for i in range(1, len(keys) + 1)])
            if key_manager.delete_key(keys[int(choice) - 1].name):
                console.print("[green]Key deleted[/green]")


def main():
    """Main entry point."""
    cli = InteractiveAnalystCLI()

    if not cli.startup():
        console.print("[red]Startup failed[/red]")
        return

    cli.run()


if __name__ == "__main__":
    main()
