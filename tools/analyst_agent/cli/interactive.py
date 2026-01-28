"""
Interactive Test CLI for Analyst Agent

Provides a rich interactive interface for testing NPRD to TRD conversion
with real knowledge base integration and live feedback.
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
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb.client import ChromaKBClient
from utils.config import get_config, Config
from generator_langchain import generate_trd_interactive
from chat import ChatAgent
from chat import print_assistant_message, print_user_message
from key_manager import get_key_manager, get_api_key_interactive

console = Console()


class InteractiveAnalystAgent:
    """Interactive CLI for Analyst Agent testing."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize interactive agent."""
        self.config: Optional[Config] = None
        self.kb_client: Optional[ChromaKBClient] = None
        self.api_key: Optional[str] = None
        self.chat_agent: Optional[ChatAgent] = None
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.nprd_dir = self.project_root / "outputs" / "videos"
        self.trd_dir = self.project_root / "docs" / "trds"

    def startup(self) -> bool:
        """
        Run startup sequence: KB check, API key, config.

        Returns True if startup successful, False otherwise.
        """
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]QuantMindX Analyst Agent[/bold cyan]\n"
            "[dim]Interactive Test Environment[/dim]",
            box=box.DOUBLE,
            padding=(1, 2)
        ))

        # Step 1: Check Knowledge Base
        console.print("\n[bold yellow]Step 1: Checking Knowledge Base...[/bold yellow]")
        if not self._check_knowledge_base():
            console.print("[red]Knowledge base check failed[/red]")
            return False

        # Step 2: Configure API
        console.print("\n[bold yellow]Step 2: Configure OpenRouter API[/bold yellow]")
        self._configure_api()

        # Step 3: Load Configuration
        console.print("\n[bold yellow]Step 3: Load Configuration[/bold yellow]")
        self._load_configuration()

        # Show status dashboard
        self._show_status_dashboard()

        return True

    def _check_knowledge_base(self) -> bool:
        """Check if ChromaDB knowledge base is available."""
        try:
            self.kb_client = ChromaKBClient()

            # List collections
            collections = self.kb_client.list_collections()

            console.print(f"[green]ChromaDB connected:[/green] {self.kb_client.db_path}")

            # Show collections table
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Collection", style="cyan")
            table.add_column("Documents", justify="right", style="green")
            table.add_column("Status", style="yellow")

            for col_name in collections:
                try:
                    stats = self.kb_client.get_collection_stats(col_name)
                    count = stats.get("count", 0)
                    status = "[green]Ready[/green]" if count > 0 else "[dim]Empty[/dim]"
                    table.add_row(col_name, str(count), status)
                except Exception as e:
                    table.add_row(col_name, "0", f"[red]Error[/red]")

            console.print(table)

            # Check if analyst_kb exists, if not offer to create it
            if "analyst_kb" not in collections and "mql5_knowledge" in collections:
                console.print("\n[yellow]analyst_kb collection not found[/yellow]")
                if Confirm.ask("Create analyst_kb from mql5_knowledge?"):
                    with console.status("[bold cyan]Creating analyst_kb collection...", spinner="dots"):
                        stats = self.kb_client.create_analyst_kb()
                    console.print(f"[green]Created analyst_kb:[/green] {stats['included']} documents")

            return True

        except Exception as e:
            console.print(f"[red]Knowledge base error:[/red] {e}")
            console.print("[yellow]Tip: Ensure ChromaDB is initialized at data/chromadb[/yellow]")
            return False

    def _configure_api(self):
        """Configure OpenRouter API key using key manager."""
        key_manager = get_key_manager()

        # First check if we have stored keys
        stored_keys = key_manager.list_keys()

        # Also check environment variable
        env_key = os.getenv("OPENROUTER_API_KEY")

        console.print("\n[bold cyan]API Key Selection[/bold cyan]")

        if stored_keys or env_key:
            # We have options - let user choose
            console.print("[dim]Found the following API key sources:[/dim]")

            if stored_keys:
                console.print(f"  [green]✓[/green] {len(stored_keys)} stored key(s)")
            if env_key:
                masked = env_key[:8] + "..." if len(env_key) > 8 else "***"
                console.print(f"  [green]✓[/green] Environment variable: {masked}")

            console.print("")
            if Confirm.ask("Select from existing keys?", default=True):
                selected_key = key_manager.select_key_interactive()
                if selected_key:
                    self.api_key = selected_key.key
                    masked = selected_key.get_masked_key()
                    console.print(f"[green]✓ Using API key:[/green] {masked} ({selected_key.name})")
                    return
                else:
                    console.print("[yellow]Selection cancelled[/yellow]")

        # No stored keys or user wants to add new
        console.print("\n[dim]Options:[/dim]")
        console.print("  [1] Enter new API key")
        console.print("  [2] Continue without API key (limited functionality)")
        console.print("  [0] Cancel")

        choice = Prompt.ask("\nSelect option", choices=["1", "2", "0"], default="1")

        if choice == "1":
            # Add new key interactively
            new_key = key_manager.add_key_interactive()
            if new_key:
                self.api_key = new_key.key
                console.print(f"[green]✓ New API key configured:[/green] {new_key.get_masked_key()}")
            else:
                console.print("[yellow]Continuing without API key[/yellow]")
        elif choice == "0":
            console.print("[yellow]Skipping API key configuration[/yellow]")
        else:
            console.print("[yellow]Continuing without API key (limited functionality)[/yellow]")

    def _save_to_env_file(self, api_key: str):
        """Save API key to .env file."""
        env_path = self.project_root / ".env"
        try:
            existing_lines = []
            if env_path.exists():
                with open(env_path, "r") as f:
                    existing_lines = f.readlines()

            # Update or add OPENROUTER_API_KEY
            updated = False
            for i, line in enumerate(existing_lines):
                if line.startswith("OPENROUTER_API_KEY="):
                    existing_lines[i] = f'OPENROUTER_API_KEY="{api_key}"\n'
                    updated = True
                    break

            if not updated:
                existing_lines.append(f'OPENROUTER_API_KEY="{api_key}"\n')

            with open(env_path, "w") as f:
                f.writelines(existing_lines)

            console.print(f"[green]Saved to:[/green] {env_path}")
        except Exception as e:
            console.print(f"[yellow]Could not save .env file:[/yellow] {e}")

    def _load_configuration(self):
        """Load configuration from file and defaults."""
        try:
            self.config = get_config()
            console.print("[green]Configuration loaded[/green]")

            # Show key settings
            console.print(f"  [dim]Model:[/dim] {self.config.llm.model}")
            console.print(f"  [dim]Provider:[/dim] {self.config.llm.provider}")
            console.print(f"  [dim]KB Collection:[/dim] {self.config.kb.collection_name}")
        except Exception as e:
            console.print(f"[yellow]Config load warning:[/yellow] {e}")
            self.config = Config()

    def _show_status_dashboard(self):
        """Show current status dashboard."""
        # Count NPRD files
        nprd_files = list(self.nprd_dir.rglob("*.json"))
        nprd_chunks = [f for f in nprd_files if "chunk" in f.name]
        nprd_main = [f for f in nprd_files if "chunk" not in f.name]

        # Count existing TRDs
        trd_files = list(self.trd_dir.glob("*.md")) if self.trd_dir.exists() else []

        # Create dashboard
        console.print("\n")
        panel = Panel(
            f"[bold cyan]Knowledge Base:[/bold cyan] {len(self.kb_client.list_collections())} collections\n"
            f"[bold cyan]NPRD Files:[/bold cyan] {len(nprd_main)} main, {len(nprd_chunks)} chunks\n"
            f"[bold cyan]TRD Files:[/bold cyan] {len(trd_files)} generated\n"
            f"[bold cyan]API Key:[/bold cyan] {'[green]Set[/green]' if self.api_key else '[yellow]Not set[/yellow]'}",
            title="[bold green]System Ready[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(panel)

    def run_interactive_loop(self):
        """Run main interactive command loop."""
        while True:
            console.print("\n")

            # Show main menu
            console.print("[bold cyan]Main Menu:[/bold cyan]")
            console.print("  [1] Scan and list NPRD files")
            console.print("  [2] Search knowledge base")
            console.print("  [3] Generate TRD from NPRD")
            console.print("  [4] View existing TRDs")
            console.print("  [5] Test KB connection")
            console.print("  [6] View configuration")
            console.print("  [7] 💬 Chat with Agent")
            console.print("  [8] 🔑 Manage API keys")
            console.print("  [0] Exit")

            choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4", "5", "6", "7", "8", "0"], default="0")

            try:
                if choice == "1":
                    self._action_list_nprd()
                elif choice == "2":
                    self._action_search_kb()
                elif choice == "3":
                    self._action_generate_trd()
                elif choice == "4":
                    self._action_list_trds()
                elif choice == "5":
                    self._action_test_kb()
                elif choice == "6":
                    self._action_show_config()
                elif choice == "7":
                    self._action_chat()
                elif choice == "8":
                    self._action_manage_keys()
                elif choice == "0":
                    console.print("[yellow]Goodbye![/yellow]")
                    break

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

    def _action_list_nprd(self):
        """List available NPRD files."""
        console.print("\n[bold]Scanning for NPRD files...[/bold]")

        nprd_files = []
        for json_file in self.nprd_dir.rglob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    nprd_files.append({
                        "path": json_file,
                        "name": json_file.name,
                        "size": json_file.stat().st_size,
                        "modified": json_file.stat().st_mtime,
                        "has_transcript": bool(data.get("transcript")),
                        "has_ocr": bool(data.get("ocr_text")),
                    })
            except Exception:
                pass

        if not nprd_files:
            console.print("[yellow]No NPRD files found[/yellow]")
            return

        # Sort by modified time
        nprd_files.sort(key=lambda x: x["modified"], reverse=True)

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", width=3)
        table.add_column("File Name", style="cyan")
        table.add_column("Size", justify="right", width=10)
        table.add_column("T", justify="center", width=3)
        table.add_column("O", justify="center", width=3)

        for i, f in enumerate(nprd_files[:20], 1):  # Show first 20
            size_kb = f"{f['size'] / 1024:.1f} KB"
            tx = "[green]✓[/green]" if f["has_transcript"] else "[red]✗[/red]"
            ocr = "[green]✓[/green]" if f["has_ocr"] else "[red]✗[/red]"

            table.add_row(str(i), f["name"][:40], size_kb, tx, ocr)

        console.print(table)
        console.print(f"\n[dim]Showing {min(20, len(nprd_files))} of {len(nprd_files)} files[/dim]")
        console.print("[dim]T=Transcript, O=OCR[/dim]")

    def _action_search_kb(self):
        """Search knowledge base."""
        if not self.kb_client:
            console.print("[red]Knowledge base not connected[/red]")
            return

        query = Prompt.ask("\nEnter search query")
        if not query:
            return

        collection = "analyst_kb" if "analyst_kb" in self.kb_client.list_collections() else "mql5_knowledge"

        with console.status(f"[bold cyan]Searching '{collection}'...", spinner="dots"):
            results = self.kb_client.search(query, collection=collection, n=5)

        if not results:
            console.print(f"[yellow]No results found for:[/yellow] {query}")
            return

        console.print(f"\n[green]Found {len(results)} results:[/green]\n")

        for i, result in enumerate(results, 1):
            console.print(Panel.fit(
                f"[bold cyan]{i}. {result['title']}[/bold cyan]\n"
                f"[dim]Score:[/dim] {result['score']:.2f}  "
                f"[dim]Categories:[/dim] {result.get('categories', 'N/A')}\n"
                f"[dim]{result['preview'][:200]}...[/dim]",
                box=box.ROUNDED
            ))

    def _action_generate_trd(self):
        """Generate TRD from selected NPRD file."""
        # List NPRD files for selection
        nprd_files = list(self.nprd_dir.rglob("*.json"))
        if not nprd_files:
            console.print("[yellow]No NPRD files found[/yellow]")
            return

        # Show main files (not chunks)
        main_files = [f for f in nprd_files if "chunk" not in f.name]
        if not main_files:
            main_files = nprd_files[:10]

        console.print("\n[bold]Available NPRD files:[/bold]")
        for i, f in enumerate(main_files, 1):
            console.print(f"  [{i}] {f.name}")

        choice_idx = Prompt.ask("\nSelect file number", default="1")
        try:
            idx = int(choice_idx) - 1
            if idx < 0 or idx >= len(main_files):
                console.print("[red]Invalid selection[/red]")
                return
            nprd_path = main_files[idx]
        except ValueError:
            console.print("[red]Invalid input[/red]")
            return

        console.print(f"\n[bold cyan]Processing:[/bold cyan] {nprd_path.name}")

        # Load NPRD data for preview
        with open(nprd_path, "r") as f:
            nprd_data = json.load(f)

        console.print(f"  [dim]Transcript:[/dim] {'Yes' if nprd_data.get('transcript') else 'No'}")
        console.print(f"  [dim]OCR Text:[/dim] {'Yes' if nprd_data.get('ocr_text') else 'No'}")
        console.print(f"  [dim]Keywords:[/dim] {len(nprd_data.get('keywords', []))}")

        if not Confirm.ask("\nContinue with TRD generation?"):
            return

        # Generate TRD using the working generator
        console.print("\n[bold]Starting TRD generation...[/bold]")
        console.print(f"  [dim]Model:[/dim] qwen/qwen3-vl-30b-a3b-thinking")
        console.print(f"  [dim]KB Search:[/dim] {'Yes' if self.kb_client else 'No'}")

        try:
            output_path = generate_trd_interactive(
                nprd_path=nprd_path,
                kb_client=self.kb_client
            )
            if output_path:
                console.print(f"\n[green]✓ TRD successfully generated![/green]")
                console.print(f"[dim]Location:[/dim] {output_path}")
        except Exception as e:
            console.print(f"\n[red]✗ Generation failed:[/red] {e}")

    def _action_list_trds(self):
        """List existing TRD files."""
        if not self.trd_dir.exists():
            console.print("[yellow]No TRD directory found[/yellow]")
            return

        trd_files = list(self.trd_dir.glob("*.md"))

        if not trd_files:
            console.print("[yellow]No TRD files found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("TRD File", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Modified", style="dim")

        import datetime
        for f in sorted(trd_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_kb = f"{f.stat().st_size / 1024:.1f} KB"
            modified = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(f.name, size_kb, modified)

        console.print(table)

    def _action_test_kb(self):
        """Test knowledge base connection with sample queries."""
        if not self.kb_client:
            console.print("[red]Knowledge base not connected[/red]")
            return

        console.print("\n[bold]Testing Knowledge Base[/bold]")

        test_queries = [
            "ORB strategy",
            "Kelly criterion",
            "position sizing",
            "risk management",
        ]

        collection = "analyst_kb" if "analyst_kb" in self.kb_client.list_collections() else "mql5_knowledge"

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Query", style="cyan")
        table.add_column("Results", justify="right", style="green")
        table.add_column("Best Match", style="yellow")

        for query in test_queries:
            with console.status(f"[dim]Testing: {query}[/dim]", spinner="dots"):
                results = self.kb_client.search(query, collection=collection, n=1)

            count = len(results)
            best = results[0]["title"][:30] if results else "No results"

            table.add_row(query, str(count), best)

        console.print(table)
        console.print("\n[green]Knowledge base is responding correctly[/green]")

    def _action_show_config(self):
        """Show current configuration."""
        console.print("\n[bold]Current Configuration[/bold]")

        if not self.config:
            console.print("[yellow]Using default configuration[/yellow]")
            self.config = Config()

        table = Table(show_header=False, box=box.ROUNDED)
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("LLM Provider", self.config.llm.provider)
        table.add_row("LLM Model", self.config.llm.model)
        table.add_row("LLM Base URL", self.config.llm.base_url)
        table.add_row("Temperature", str(self.config.llm.temperature))
        table.add_row("Max Tokens", str(self.config.llm.max_tokens))
        table.add_row("", "")  # spacer
        table.add_row("ChromaDB Path", self.config.kb.chroma_path)
        table.add_row("Collection Name", self.config.kb.collection_name)
        table.add_row("Embedding Model", self.config.kb.embedding_model)
        table.add_row("", "")  # spacer
        table.add_row("NPRD Output Dir", self.config.paths.nprd_output_dir)
        table.add_row("TRD Output Dir", self.config.paths.trd_output_dir)
        table.add_row("Log Dir", self.config.paths.log_dir)

        console.print(table)

        console.print("\n[dim]API Key:[/dim] " + (
            f"[green]{self.api_key[:8]}...[/green]" if self.api_key else "[yellow]Not set[/yellow]"
        ))

    def _action_chat(self):
        """Chat with the AI agent."""
        # Check API key
        if not self.api_key:
            console.print("[yellow]API key required for chat mode[/yellow]")
            if not Confirm.ask("Enter API key now?"):
                return
            self.api_key = Prompt.ask("OpenRouter API key", password=True)
            if not self.api_key:
                console.print("[red]API key required[/red]")
                return

        # Initialize chat agent if needed
        if not self.chat_agent:
            self.chat_agent = ChatAgent(
                api_key=self.api_key,
                model="qwen/qwen3-vl-30b-a3b-thinking",
                kb_client=self.kb_client
            )

        console.print("\n")
        console.print(Panel.fit(
            "[bold green]💬 Chat Mode[/bold green]\n"
            "[dim]Ask questions about trading strategies, request TRDs, or get help[/dim]\n"
            "[dim]Type 'back' to return to menu, 'clear' to reset history[/dim]",
            border_style="green"
        ))

        # Show some example questions
        console.print("\n[dim]Example questions:[/dim]")
        console.print("  [dim]• What is the ORB strategy?[/dim]")
        console.print("  [dim]• Explain Kelly criterion vs optimal-f[/dim]")
        console.print("  [dim]• How do I implement position sizing?[/dim]")
        console.print("  [dim]• Generate a TRD for a breakout strategy[/dim]")
        console.print("")

        while True:
            try:
                user_input = Prompt.ask(
                    "\n[bold cyan]You[/bold cyan]",
                    default="",
                    show_default=False
                )

                if not user_input.strip():
                    continue

                # Check for commands
                if user_input.lower() in ["back", "exit", "quit", "0"]:
                    console.print("[dim]Returning to main menu...[/dim]\n")
                    break

                if user_input.lower() == "clear":
                    self.chat_agent.clear_history()
                    console.print("[green]✓ Conversation history cleared[/green]")
                    continue

                if user_input.lower() == "help":
                    console.print("\n[bold]Chat commands:[/bold]")
                    console.print("  [dim]back[/dim] - Return to main menu")
                    console.print("  [dim]clear[/dim] - Clear conversation history")
                    console.print("  [dim]help[/dim] - Show this help message")
                    continue

                # Display user message
                print_user_message(user_input)

                # Get assistant response
                with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                    response = self.chat_agent.chat(user_input)

                # Display assistant response
                print_assistant_message(response, self.chat_agent.session.messages[-1].sources)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Exiting chat...[/dim]\n")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

    def _action_manage_keys(self):
        """Manage API keys - add, list, delete, select."""
        key_manager = get_key_manager()

        while True:
            console.print("\n")
            console.print("[bold cyan]API Key Management[/bold cyan]")

            # Show current keys summary
            keys = key_manager.list_keys()
            console.print(f"[dim]Stored keys: {len(keys)}[/dim]")

            console.print("\n[dim]Options:[/dim]")
            console.print("  [1] List all keys")
            console.print("  [2] Add new key")
            console.print("  [3] Delete a key")
            console.print("  [4] Select active key")
            console.print("  [0] Back to main menu")

            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "0"], default="0")

            if choice == "0":
                break
            elif choice == "1":
                self._key_list_keys(key_manager)
            elif choice == "2":
                self._key_add_new(key_manager)
            elif choice == "3":
                self._key_delete(key_manager)
            elif choice == "4":
                self._key_select_active(key_manager)

    def _key_list_keys(self, key_manager):
        """List all stored API keys."""
        keys = key_manager.list_keys()

        if not keys:
            console.print("\n[yellow]No stored API keys[/yellow]")
            return

        console.print("\n")
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="dim")
        table.add_column("Key", style="yellow")
        table.add_column("Model", style="dim")
        table.add_column("Created", style="dim")

        from datetime import datetime
        for key in keys:
            created = key.created_at[:10] if key.created_at else "N/A"
            table.add_row(
                key.name,
                key.provider,
                key.get_masked_key(),
                key.model[:30] + "..." if len(key.model) > 30 else key.model,
                created
            )

        console.print(table)

        # Also show environment variable
        env_key = os.getenv("OPENROUTER_API_KEY")
        if env_key:
            masked = env_key[:8] + "..." + env_key[-4:] if len(env_key) > 12 else "***"
            console.print(f"\n[green]Environment variable (OPENROUTER_API_KEY):[/green] {masked}")

    def _key_add_new(self, key_manager):
        """Add a new API key."""
        console.print("\n[bold]Add New API Key[/bold]")

        new_key = key_manager.add_key_interactive()
        if new_key:
            console.print(f"\n[green]✓ Key '{new_key.name}' added successfully[/green]")
            console.print(f"[dim]Key: {new_key.get_masked_key()}[/dim]")
            console.print(f"[dim]Model: {new_key.model}[/dim]")

    def _key_delete(self, key_manager):
        """Delete an API key."""
        keys = key_manager.list_keys()

        if not keys:
            console.print("\n[yellow]No stored API keys to delete[/yellow]")
            return

        console.print("\n[bold]Delete API Key[/bold]")

        # Show keys for selection
        console.print("\n[dim]Select key to delete:[/dim]")
        for i, key in enumerate(keys, 1):
            console.print(f"  [{i}] {key.name} - {key.get_masked_key()}")
        console.print("  [0] Cancel")

        choice = Prompt.ask("\nSelect", choices=[str(i) for i in range(len(keys) + 1)])

        if choice == "0":
            console.print("[dim]Cancelled[/dim]")
            return

        idx = int(choice) - 1
        key_to_delete = keys[idx]

        # Confirm deletion
        console.print(f"\n[yellow]About to delete key:[/yellow] {key_to_delete.name}")
        console.print(f"[dim]Provider: {key_to_delete.provider}[/dim]")
        console.print(f"[dim]Key: {key_to_delete.get_masked_key()}[/dim]")

        if Confirm.ask("Are you sure?", default=False):
            key_manager.delete_key(key_to_delete.name)
            console.print(f"[green]✓ Key '{key_to_delete.name}' deleted[/green]")
        else:
            console.print("[dim]Deletion cancelled[/dim]")

    def _key_select_active(self, key_manager):
        """Select and set the active API key."""
        console.print("\n[bold]Select Active API Key[/bold]")

        selected_key = key_manager.select_key_interactive("Choose API key to use")

        if selected_key:
            self.api_key = selected_key.key
            console.print(f"\n[green]✓ Active API key set to:[/green] {selected_key.name}")
            console.print(f"[dim]Key: {selected_key.get_masked_key()}[/dim]")
            console.print(f"[dim]Model: {selected_key.model}[/dim]")
        else:
            console.print("[yellow]No key selected[/yellow]")


def main():
    """Main entry point for interactive CLI."""
    agent = InteractiveAnalystAgent()

    if not agent.startup():
        console.print("[red]Startup failed. Please fix the issues above and try again.[/red]")
        return 1

    agent.run_interactive_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
