"""
API Key Manager for Analyst Agent.
Store and manage multiple API keys with labels.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box
import os

console = Console()


@dataclass
class StoredKey:
    """A stored API key."""
    name: str
    key: str
    provider: str = "openrouter"
    model: str = "qwen/qwen3-vl-30b-a3b-thinking"
    created_at: str = None

    def __post_init__(self):
        from datetime import datetime
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def get_masked_key(self) -> str:
        """Get masked version of the key for display."""
        if len(self.key) > 8:
            return self.key[:8] + "..." + self.key[-4:]
        return "***"


class APIKeyManager:
    """Manages API keys storage and retrieval."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize key manager.

        Args:
            storage_path: Path to store keys. Defaults to ~/.quantmindx/keys.json
        """
        if storage_path is None:
            home = Path.home()
            storage_path = home / ".quantmindx" / "keys.json"

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, StoredKey] = {}
        self._load_keys()

    def _load_keys(self):
        """Load keys from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for name, key_data in data.items():
                        self.keys[name] = StoredKey(**key_data)
            except Exception as e:
                console.print(f"[dim]Could not load keys: {e}[/dim]")
                self.keys = {}

    def _save_keys(self):
        """Save keys to storage."""
        try:
            data = {name: asdict(key) for name, key in self.keys.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Could not save keys: {e}[/yellow]")

    def list_keys(self) -> List[StoredKey]:
        """List all stored keys."""
        return list(self.keys.values())

    def add_key(self, name: str, key: str, provider: str = "openrouter", model: str = None) -> bool:
        """Add a new API key.

        Args:
            name: Label for the key
            key: The API key
            provider: Provider name (default: openrouter)
            model: Default model for this key

        Returns:
            True if added successfully
        """
        if name in self.keys:
            console.print(f"[yellow]Key '{name}' already exists. Overwriting...[/yellow]")
            if not Confirm.ask("Continue?", default=False):
                return False

        self.keys[name] = StoredKey(
            name=name,
            key=key,
            provider=provider,
            model=model or "qwen/qwen3-vl-30b-a3b-thinking"
        )
        self._save_keys()
        return True

    def get_key(self, name: str) -> Optional[StoredKey]:
        """Get a stored key by name."""
        return self.keys.get(name)

    def delete_key(self, name: str) -> bool:
        """Delete a stored key."""
        if name in self.keys:
            del self.keys[name]
            self._save_keys()
            return True
        return False

    def select_key_interactive(self, prompt: str = "Select API key") -> Optional[StoredKey]:
        """Interactive key selection.

        Args:
            prompt: Prompt message to display

        Returns:
            Selected StoredKey or None
        """
        try:
            keys = self.list_keys()

            if not keys:
                console.print("[yellow]No saved API keys found[/yellow]")
                if Confirm.ask("Add a new API key now?"):
                    return self.add_key_interactive()
                return None

            console.print(f"\n[bold cyan]{prompt}[/bold cyan]")

            # Show keys table
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", width=3)
            table.add_column("Name", style="cyan")
            table.add_column("Provider", style="dim")
            table.add_column("Key", style="yellow")
            table.add_column("Model", style="dim")

            for i, key in enumerate(keys, 1):
                table.add_row(
                    str(i),
                    key.name,
                    key.provider,
                    key.get_masked_key(),
                    key.model[:30] + "..." if len(key.model) > 30 else key.model
                )

            console.print(table)

            # Also check environment variable
            env_key = os.getenv("OPENROUTER_API_KEY")
            env_option = len(keys) + 1
            if env_key:
                masked = env_key[:8] + "..." + env_key[-4:] if len(env_key) > 12 else "***"
                console.print(f"  [{env_option}] [dim]Environment variable (OPENROUTER_API_KEY)[/dim] [green]{masked}[/green]")

            console.print(f"  [{env_option + 1}] [dim]Add new API key[/dim]")
            console.print(f"  [0] [dim]Cancel[/dim]")

            choice = Prompt.ask("\nSelect option", default="0")

            if choice == "0":
                return None
            elif choice == str(env_option + 1):
                return self.add_key_interactive()
            elif choice == str(env_option) and env_key:
                # Return environment key as temporary stored key
                return StoredKey(
                    name="Environment",
                    key=env_key,
                    provider="openrouter",
                    model="qwen/qwen3-vl-30b-a3b-thinking"
                )
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(keys):
                        return keys[idx]
                except ValueError:
                    pass

            console.print("[red]Invalid selection[/red]")
            return None

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Key selection cancelled[/yellow]")
            return None

    def add_key_interactive(self) -> Optional[StoredKey]:
        """Interactively add a new key.

        Returns:
            The newly created StoredKey or None
        """
        try:
            console.print("\n[bold cyan]Add New API Key[/bold cyan]")

            # Get existing key names
            existing_names = set(self.keys.keys())

            # Prompt for name
            while True:
                name = Prompt.ask("Enter a label for this key", default="default")
                if not name:
                    console.print("[red]Name cannot be empty[/red]")
                    continue
                if name in existing_names:
                    if not Confirm.ask(f"Key '{name}' already exists. Overwrite?"):
                        continue
                break

            # Prompt for key
            key = Prompt.ask("Enter API key", password=True)
            if not key:
                console.print("[red]API key cannot be empty[/red]")
                return None

            # Prompt for provider (with default)
            provider = Prompt.ask("Provider", default="openrouter")

            # Prompt for model (with default)
            model = Prompt.ask("Model", default="qwen/qwen3-vl-30b-a3b-thinking")

            # Add the key
            stored_key = StoredKey(
                name=name,
                key=key,
                provider=provider,
                model=model
            )

            self.keys[name] = stored_key
            self._save_keys()

            console.print(f"[green]✓ API key '{name}' saved[/green]")

            return stored_key

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Key addition cancelled[/yellow]")
            return None


# Singleton instance
_key_manager_instance: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get the singleton key manager instance."""
    global _key_manager_instance
    if _key_manager_instance is None:
        _key_manager_instance = APIKeyManager()
    return _key_manager_instance


def get_api_key_interactive(prompt: str = "Select API key") -> Optional[str]:
    """
    Interactively get an API key from stored keys or environment.

    Args:
        prompt: Prompt message to display

    Returns:
        API key string or None
    """
    manager = get_key_manager()
    selected = manager.select_key_interactive(prompt)

    if selected:
        return selected.key

    return None
