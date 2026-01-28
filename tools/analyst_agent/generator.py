"""
Working TRD Generator using OpenRouter API.
Simple implementation that generates TRDs from NPRD content.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def create_openrouter_llm(api_key: str, model: str = "qwen/qwen3-vl-30b-a3b-thinking"):
    """
    Create a simple OpenRouter LLM wrapper.

    Args:
        api_key: OpenRouter API key
        model: Model name to use

    Returns:
        dict with invoke method for LangChain compatibility
    """
    class OpenRouterLLM:
        def __init__(self, api_key: str, model: str):
            self.api_key = api_key
            self.model = model
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        def invoke(self, prompt: str) -> str:
            """Invoke the model synchronously."""
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert trading strategy analyst who generates detailed Technical Requirements Documents (TRDs) from trading content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 4096,
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        async def ainvoke(self, prompt: str) -> str:
            """Async invoke for LangChain compatibility."""
            import asyncio
            return await asyncio.to_thread(self.invoke, prompt)

    return OpenRouterLLM(api_key, model)


def generate_trd_from_nprd(
    nprd_path: Path,
    api_key: str,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    kb_client=None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Generate a TRD from an NPRD file using OpenRouter.

    Args:
        nprd_path: Path to NPRD JSON file
        api_key: OpenRouter API key
        model: Model to use
        kb_client: Optional KB client for searching related articles
        output_dir: Optional output directory (defaults to docs/trds)

    Returns:
        Path to generated TRD file
    """
    # Load NPRD data
    with open(nprd_path, "r") as f:
        nprd_data = json.load(f)

    console.print(f"[cyan]Processing:[/cyan] {nprd_path.name}")

    # Extract content
    transcript = nprd_data.get("transcript", "")
    ocr_text = nprd_data.get("ocr_text", "")
    keywords = nprd_data.get("keywords", [])

    # Combine content
    content = transcript or ocr_text or ""
    if not content:
        console.print("[yellow]No content found in NPRD file[/yellow]")
        # Use filename as fallback
        content = f"Trading strategy video: {nprd_path.stem}"

    # Search KB if available
    kb_articles_text = ""
    if kb_client:
        with console.status("[dim]Searching knowledge base...[/dim]", spinner="dots"):
            # Generate search queries from content/keywords
            search_queries = []

            # Use keywords if available
            if keywords:
                search_queries.extend(keywords[:3])
            else:
                # Fallback queries based on filename
                search_queries = ["trading strategy", "entry exit", "risk management"]

            # Search and collect articles
            kb_results = []
            for query in search_queries[:3]:
                try:
                    results = kb_client.search(query, collection="analyst_kb", n=2)
                    kb_results.extend(results)
                except Exception as e:
                    console.print(f"[dim]KB search error for '{query}': {e}[/dim]")

            # Format KB articles
            if kb_results:
                kb_articles_text = "\n\n### Relevant Knowledge Base Articles:\n\n"
                for i, article in enumerate(kb_results[:5], 1):
                    kb_articles_text += f"{i}. **{article['title']}**\n"
                    kb_articles_text += f"   - Relevance: {article['score']:.2f}\n"
                    kb_articles_text += f"   - Preview: {article['preview'][:200]}...\n\n"
            else:
                kb_articles_text = "\n\nNo relevant KB articles found.\n"

    # Build the prompt
    prompt = f"""Generate a complete Technical Requirements Document (TRD) in markdown format for the following trading strategy.

## Source Content

**File:** {nprd_path.name}
**Keywords:** {', '.join(keywords) if keywords else 'N/A'}

**Content:**
{content[:8000]}

{kb_articles_text}

## Required TRD Structure

Generate a markdown document with these sections:

1. **YAML Frontmatter** (at the top)
   ```yaml
   ---
   strategy_name: "Extracted Strategy Name"
   source: "{nprd_path.name}"
   generated_at: "{nprd_data.get('created_at', '2024-01-01')}"
   status: "draft"
   version: "1.0"
   analyst_version: "1.0"
   ---
   ```

2. **Overview** - 2-3 paragraphs describing the strategy

3. **Entry Logic**
   - Primary Entry Trigger
   - Entry Confirmation
   - Entry Example (pseudo-code)

4. **Exit Logic**
   - Take Profit
   - Stop Loss
   - Trailing Stop (if mentioned)
   - Time Exit (if mentioned)

5. **Filters**
   - Time Filters (session, time of day)
   - Market Condition Filters (volatility, spread, trend)

6. **Indicators & Settings** - Table format

7. **Position Sizing & Risk Management**
   - Risk Per Trade
   - Position Sizing Method
   - Max Drawdown Limit

8. **Knowledge Base References** - Reference relevant KB articles found above

9. **Missing Information** - List any critical info not found

10. **Next Steps** - Implementation steps

Be thorough and specific. Extract all details from the content. Use proper markdown formatting."""

    # Call OpenRouter
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Generating TRD using {model}...[/cyan]",
            total=None,
        )

        llm = create_openrouter_llm(api_key, model)

        try:
            trd_content = llm.invoke(prompt)
        except requests.exceptions.RequestException as e:
            console.print(f"[red]API Error:[/red] {e}")
            console.print("[yellow]Check your API key and try again[/yellow]")
            raise
        except Exception as e:
            console.print(f"[red]Generation error:[/red] {e}")
            raise

    # Save TRD
    output_dir = output_dir or Path.cwd() / "docs" / "trds"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = nprd_path.stem + "_trd.md"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        f.write(trd_content)

    console.print(f"[green]✓ TRD saved to:[/green] {output_path}")

    return output_path


def generate_trd_interactive(
    nprd_path: Path,
    kb_client=None
) -> Optional[Path]:
    """
    Generate TRD with interactive API key prompt.

    Args:
        nprd_path: Path to NPRD file
        kb_client: Optional KB client

    Returns:
        Path to generated TRD or None if failed
    """
    from rich.prompt import Prompt, Confirm

    # Check for API key in env
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        console.print("[yellow]OPENROUTER_API_KEY not found in environment[/yellow]")
        if not Confirm.ask("Enter OpenRouter API key now?", default=False):
            console.print("[yellow]Cannot generate TRD without API key[/yellow]")
            return None

        api_key = Prompt.ask("OpenRouter API key", password=True)
        if not api_key:
            console.print("[red]API key required[/red]")
            return None

    # Use user's preferred model
    model = "qwen/qwen3-vl-30b-a3b-thinking"

    # Generate
    try:
        return generate_trd_from_nprd(
            nprd_path=nprd_path,
            api_key=api_key,
            model=model,
            kb_client=kb_client
        )
    except Exception as e:
        console.print(f"[red]Failed to generate TRD:[/red] {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python generator.py <nprd_file.json>[/yellow]")
        sys.exit(1)

    nprd_file = Path(sys.argv[1])
    if not nprd_file.exists():
        console.print(f"[red]File not found:[/red] {nprd_file}")
        sys.exit(1)

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = console.input("[yellow]Enter OpenRouter API key:[/yellow] ", password=True)

    if not api_key:
        console.print("[red]API key required[/red]")
        sys.exit(1)

    # Generate
    generate_trd_from_nprd(nprd_file, api_key)
