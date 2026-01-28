"""
Interactive interface for Analyst Agent CLI
"""

import typer
from typing import Optional, Dict, Any
from pathlib import Path
import json
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.console import Console
from rich.panel import Panel

console = Console()

def interactive_prompt(trd_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interactive prompt to complete missing TRD fields.

    Args:
        trd_data: Current TRD data with potential missing fields

    Returns:
        Updated TRD data with user-provided values
    """
    console.print(Panel.fit(
        "[bold]Interactive TRD Completion[/bold]\n"
        "Please provide missing information to complete the TRD.",
        title="Interactive Mode"
    ))

    # Example interactive prompts - these would be customized based on actual TRD structure
    if "project_name" not in trd_data or not trd_data["project_name"]:
        trd_data["project_name"] = Prompt.ask("Project Name", default="Unknown Project")

    if "version" not in trd_data or not trd_data["version"]:
        trd_data["version"] = Prompt.ask("Version", default="1.0.0")

    if "author" not in trd_data or not trd_data["author"]:
        trd_data["author"] = Prompt.ask("Author", default="Unknown Author")

    if "date_created" not in trd_data or not trd_data["date_created"]:
        trd_data["date_created"] = Prompt.ask("Date Created", default="2024-01-01")

    if "requirements" not in trd_data or not trd_data["requirements"]:
        console.print("[yellow]Requirements section is empty[/yellow]")
        add_requirements = Confirm.ask("Would you like to add requirements?")
        if add_requirements:
            requirements = []
            while True:
                req = Prompt.ask("Enter requirement (leave empty to finish)")
                if not req:
                    break
                requirements.append(req)
            trd_data["requirements"] = requirements

    if "technical_details" not in trd_data or not trd_data["technical_details"]:
        console.print("[yellow]Technical details section is empty[/yellow]")
        add_technical = Confirm.ask("Would you like to add technical details?")
        if add_technical:
            technical_details = {}
            technical_details["architecture"] = Prompt.ask("Architecture", default="Not specified")
            technical_details["technologies"] = Prompt.ask("Technologies", default="Not specified")
            technical_details["dependencies"] = Prompt.ask("Dependencies", default="Not specified")
            trd_data["technical_details"] = technical_details

    if "acceptance_criteria" not in trd_data or not trd_data["acceptance_criteria"]:
        console.print("[yellow]Acceptance criteria section is empty[/yellow]")
        add_criteria = Confirm.ask("Would you like to add acceptance criteria?")
        if add_criteria:
            criteria = []
            while True:
                criterion = Prompt.ask("Enter acceptance criterion (leave empty to finish)")
                if not criterion:
                    break
                criteria.append(criterion)
            trd_data["acceptance_criteria"] = criteria

    return trd_data

def confirm_action(message: str, default: bool = True) -> bool:
    """
    Confirm an action with the user.

    Args:
        message: Message to display
        default: Default answer if user just presses Enter

    Returns:
        User's confirmation
    """
    return Confirm.ask(message, default=default)

def select_option(options: List[str], message: str = "Select an option") -> str:
    """
    Let user select from multiple options.

    Args:
        options: List of available options
        message: Prompt message

    Returns:
        Selected option
    """
    if not options:
        raise ValueError("No options provided")

    if len(options) == 1:
        console.print(f"[blue]{message}[/blue]: {options[0]}")
        return options[0]

    for i, option in enumerate(options, 1):
        console.print(f"[{i}] {option}")

    choice = IntPrompt.ask(message, default=1, show_default=True)
    while choice < 1 or choice > len(options):
        console.print("[red]Invalid choice. Please try again.[/red]")
        choice = IntPrompt.ask(message, default=1, show_default=True)

    return options[choice - 1]

def prompt_for_missing_fields(trd_file: Path) -> Dict[str, Any]:
    """
    Load TRD file and prompt for missing fields.

    Args:
        trd_file: Path to TRD file

    Returns:
        Updated TRD data
    """
    try:
        with open(trd_file, 'r') as f:
            trd_data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: TRD file not found at {trd_file}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON in TRD file[/red]")
        raise typer.Exit(1)

    return interactive_prompt(trd_data)

def save_trd_file(trd_data: Dict[str, Any], output_file: Path):
    """
    Save TRD data to file.

    Args:
        trd_data: TRD data to save
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(trd_data, f, indent=2)
        console.print(f"[green]✓ TRD saved to {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving TRD file: {e}[/red]")
        raise typer.Exit(1)