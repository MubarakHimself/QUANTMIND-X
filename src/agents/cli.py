import click
import asyncio
from rich.console import Console
from rich.markdown import Markdown

from src.agents.implementations.quant_code import create_quant_code_agent
from src.agents.implementations.analyst import create_analyst_agent

console = Console()

AGENTS = {
    "quantcode": create_quant_code_agent,
    "analyst": create_analyst_agent
}

@click.group()
def cli():
    pass

@cli.command()
@click.option("--agent", default="quantcode", help="Name of the agent to launch")
def chat(agent):
    """Start an interactive chat session."""
    if agent not in AGENTS:
        console.print(f"[red]Unknown agent: {agent}[/red]")
        return

    # Initialize
    console.print(f"[green]Initializing {agent}...[/green]")
    bot = AGENTS[agent]()
    console.print(f"[bold blue]System:[/bold blue] {bot.name} is ready. Type 'exit' to quit.")

    # Loop
    while True:
        user_input = console.input("[bold yellow]You:[/bold yellow] ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            # Simple synchronous invoke for V1
            response = bot.invoke(user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    cli()
