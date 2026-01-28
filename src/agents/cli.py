import click
import asyncio
from rich.console import Console
from rich.markdown import Markdown

from src.agents.implementations.quant_code import create_quant_code_agent
from src.agents.implementations.analyst import create_analyst_agent
from src.agents.implementations.copilot import create_copilot_agent

console = Console()

AGENTS = {
    "quantcode": create_quant_code_agent,
    "analyst": create_analyst_agent,
    "copilot": create_copilot_agent
}

@click.group()
def cli():
    pass

async def async_chat(agent_name):
    """Start an interactive async chat session."""
    if agent_name not in AGENTS:
        console.print(f"[red]Unknown agent: {agent_name}[/red]")
        return

    # Initialize
    console.print(f"[green]Initializing {agent_name}...[/green]")
    bot = AGENTS[agent_name]()
    console.print(f"[bold blue]System:[/bold blue] {bot.name} is ready. Type 'exit' to quit.")

    # Loop
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: console.input("[bold yellow]You:[/bold yellow] "))
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            # Use ainvoke for all agents now
            response = await bot.ainvoke(user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

@cli.command()
@click.option("--agent", default="quantcode", help="Name of the agent to launch")
def chat(agent):
    """Start an interactive chat session."""
    asyncio.run(async_chat(agent))

if __name__ == "__main__":
    cli()
