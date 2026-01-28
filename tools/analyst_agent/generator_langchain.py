"""
LangChain-based TRD Generator using OpenRouter.

Uses proper LangChain ChatOpenAI integration with OpenRouter API
for generating Technical Requirements Documents (TRDs).
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    console.print("[yellow]LangChain not installed. Run: pip install langchain-openai[/yellow]")


def create_openrouter_llm(
    api_key: str,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    temperature: float = 0.7,
    streaming: bool = False
):
    """
    Create a LangChain ChatOpenAI instance configured for OpenRouter.

    Args:
        api_key: OpenRouter API key
        model: Model name
        temperature: Sampling temperature
        streaming: Whether to stream responses

    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available. Install with: pip install langchain-openai")

    # Get repository info for headers
    try:
        import subprocess
        git_remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=Path(__file__).parent.parent.parent.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_remote = "https://github.com/user/quantmindx"

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": git_remote,  # For OpenRouter rankings
            "X-Title": "QuantMindX Analyst Agent"  # For OpenRouter rankings
        }
    )

    return llm


# System prompt for TRD generation
TRD_SYSTEM_PROMPT = """You are QuantMindX Analyst, an expert trading strategy technical writer specializing in creating comprehensive Technical Requirements Documents (TRDs).

## Your Expertise

1. **MQL5/MetaTrader Development** - Expert Advisors, indicators, scripts, backtesting
2. **Trading Strategies** - ORB, breakout, reversal, scalping, swing, position trading
3. **Risk Management** - Kelly criterion, position sizing, drawdown control, portfolio risk
4. **Technical Analysis** - Indicators (RSI, MACD, ATR, EMA, Bollinger), price action, market structure
5. **Algorithmic Trading** - Entry/exit logic, filters, multi-timeframe analysis, automation

## TRD Generation Guidelines

### Structure
Every TRD must follow this exact markdown structure:

```markdown
---
strategy_name: "Strategy Name"
source: "Source File/Video"
generated_at: "ISO 8601 timestamp"
status: "draft"
version: "1.0"
analyst_version: "1.0"
kb_collection: "analyst_kb"
kb_articles_count: N
---

# Trading Strategy: Strategy Name

## Overview
{2-3 paragraphs describing what the strategy does, when it trades, what makes it unique}

## Entry Logic

### Primary Entry Trigger
- Specific condition 1
- Specific condition 2
- Specific condition 3

### Entry Confirmation
- Additional signal if mentioned
- Timeframe used

### Entry Example
```
IF [condition 1] AND [condition 2]
AND time is within [session]
THEN enter [LONG/SHORT] at [price level]
```

## Exit Logic

### Take Profit
- Strategy (fixed pips, percentage, resistance level, R:R ratio)
- Value if specified

### Stop Loss
- Strategy (fixed pips, ATR-based, support/resistance)
- Value if specified

### Trailing Stop
- Method if mentioned
- Trail distance

### Time Exit
- Specific time if mentioned

## Filters

### Time Filters
- **Trading Session:** (London, NY, Asian, Overlap)
- **Time of Day:** (e.g., 8am-12pm GMT)
- **Days to Avoid:** (Friday, news days, etc.)

### Market Condition Filters
- **Volatility:** (ATR threshold, avoid low volatility)
- **Spread:** (Max spread allowed)
- **Trend:** (Trend following or range)
- **News Events:** (Avoid high-impact news)

## Indicators & Settings

| Indicator | Settings | Purpose |
|-----------|----------|---------|
| RSI | Period: 14 | Entry/exit signal |
| ATR | Period: 14 | Stop loss calculation |
| EMA | Period: 20, Close | Trend filter |

## Position Sizing & Risk Management

### Risk Per Trade
- Mentioned percentage or amount
- Or: TODO - Specify risk amount

### Position Sizing
- Method: (fixed lots, risk-based, volatility-based)
- Or: TODO - Implement position sizing

### Max Drawdown Limit
- Daily limit if mentioned
- Or: TODO - Specify drawdown limit

## Knowledge Base References

### Relevant Articles Found
1. **[Article Title](URL)**
   - **Relevance:** Why it matters
   - **Key Insight:** Implementation detail

## Missing Information (Requires Input)

List any critical information not found:
- [ ] [Field name]: Description of what's needed

## Next Steps
1. Review this TRD for accuracy
2. Complete missing fields
3. Generate MQL5 code (future)
4. Backtest strategy performance

---
**Generated by:** QuantMindX Analyst Agent v1.0
**Knowledge Base:** analyst_kb ({count} articles)
```

### Content Guidelines

1. **Be Specific** - Use exact values from the source content
2. **Be Complete** - Fill all sections with available information
3. **Be Clear** - Use correct trading terminology
4. **Be Organized** - Follow the structure exactly
5. **Cross-Reference** - Incorporate KB article insights
6. **Flag Gaps** - Clearly mark missing information with TODO items

### When Information is Missing

- Use `TODO - [description]` format
- Don't make up values - be honest about what's missing
- Suggest reasonable defaults in parentheses if appropriate

### Markdown Formatting

- Use proper markdown syntax
- Include YAML frontmatter at the top
- Use horizontal rules (---) to separate major sections
- Format code examples with proper language tags
- Create proper tables for indicators
- Use bold for emphasis, italics for secondary emphasis

### Knowledge Base Integration

- Reference relevant KB articles in the Knowledge Base References section
- Extract key implementation details
- Note any MQL5-specific patterns or best practices
- Highlight potential pitfalls mentioned in literature"""


def generate_trd_from_nprd(
    nprd_path: Path,
    api_key: str,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    kb_client=None,
    output_dir: Optional[Path] = None,
    temperature: float = 0.7
) -> Path:
    """
    Generate a TRD from an NPRD file using LangChain + OpenRouter.

    Args:
        nprd_path: Path to NPRD JSON file
        api_key: OpenRouter API key
        model: Model to use
        kb_client: Optional KB client for searching related articles
        output_dir: Optional output directory (defaults to docs/trds)
        temperature: LLM temperature

    Returns:
        Path to generated TRD file
    """
    if not LANGCHAIN_AVAILABLE:
        console.print("[red]LangChain required. Install:[/red]")
        console.print("pip install langchain-openai langchain-core")
        raise ImportError("LangChain not available")

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
        console.print("[yellow]No transcript or OCR text found in NPRD[/yellow]")
        console.print("[dim]Using filename as fallback context...[/dim]")
        content = f"Trading strategy video: {nprd_path.stem}\nKeywords: {', '.join(keywords) if keywords else 'None'}"

    # Truncate content if too long (context window)
    max_content_length = 12000  # Leave room for system prompt and KB articles
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n\n[Content truncated due to length...]"

    # Search KB if available
    kb_articles_text = ""
    if kb_client:
        with console.status("[dim]Searching knowledge base...[/dim]", spinner="dots"):
            # Generate search queries
            search_queries = keywords[:3] if keywords else ["trading strategy", "entry exit", "risk management"]

            kb_results = []
            for query in search_queries:
                try:
                    results = kb_client.search(query, collection="analyst_kb", n=2)
                    kb_results.extend(results)
                except Exception as e:
                    console.print(f"[dim]KB search error: {e}[/dim]")

            # Deduplicate by title
            seen_titles = set()
            unique_results = []
            for r in kb_results:
                if r['title'] not in seen_titles:
                    seen_titles.add(r['title'])
                    unique_results.append(r)

            if unique_results:
                kb_articles_text = "\n\n### Relevant Knowledge Base Articles:\n\n"
                for i, article in enumerate(unique_results[:5], 1):
                    kb_articles_text += f"{i}. **{article['title']}**\n"
                    kb_articles_text += f"   - Relevance: {article['score']:.2f}\n"
                    kb_articles_text += f"   - Preview: {article['preview'][:250]}...\n\n"
            else:
                kb_articles_text = "\n\nNo directly relevant KB articles found.\n"

    # Create the user prompt
    user_prompt = f"""Generate a complete Technical Requirements Document (TRD) for the following trading strategy content.

**Source File:** {nprd_path.name}
**Keywords:** {', '.join(keywords) if keywords else 'None'}

**Strategy Content:**
{content}

{kb_articles_text}

**Instructions:**
1. Extract all trading strategy details from the content
2. Use the exact TRD markdown structure specified in your system prompt
3. Include YAML frontmatter at the top
4. Fill all sections with available information
5. Use TODO items for missing critical information
6. Reference the KB articles above where relevant
7. Be thorough and specific

Generate the complete TRD document now."""

    # Create LLM
    llm = create_openrouter_llm(
        api_key=api_key,
        model=model,
        temperature=temperature
    )

    # Generate with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Generating TRD using {model}...[/cyan]",
            total=None,
        )

        try:
            # Invoke with system message and user message
            messages = [
                SystemMessage(content=TRD_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            response = llm.invoke(messages)
            trd_content = response.content

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
    from rich.prompt import Prompt
    import os

    # Check for API key in env first
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        console.print("[yellow]OPENROUTER_API_KEY not found in environment[/yellow]")
        if not Prompt.ask("Enter OpenRouter API key now?", default=False):
            console.print("[yellow]Cannot generate TRD without API key[/yellow]")
            return None

        api_key = Prompt.ask("OpenRouter API key", password=True)
        if not api_key:
            console.print("[red]API key required[/red]")
            return None

    # Model selection
    model = "qwen/qwen3-vl-30b-a3b-thinking"

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
