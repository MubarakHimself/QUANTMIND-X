"""
MCP Tools Module - Modular Structure.

This package provides tool wrappers for all MCP servers:
- Context7: MQL5 documentation retrieval
- Sequential Thinking: Task decomposition
- PageIndex: PDF indexing and search
- Backtest: Strategy backtesting
- MT5 Compiler: MQL5 code compilation

For backward compatibility, imports are also available from src.agents.tools.mcp_tools.
"""

# Re-export from submodules for package-level access
from src.agents.tools.mcp.manager import MCPClientManager
from src.agents.tools.mcp.context7 import (
    get_mql5_documentation,
    get_mql5_examples,
)
from src.agents.tools.mcp.sequential_thinking import (
    sequential_thinking,
    analyze_errors,
)
from src.agents.tools.mcp.page_index import (
    index_pdf,
    search_pdf,
    get_indexed_documents,
)
from src.agents.tools.mcp.backtest import (
    run_backtest,
    get_backtest_status,
    get_backtest_results,
    compare_backtests,
)
from src.agents.tools.mcp.mt5_compiler import (
    compile_mql5_code,
    validate_mql5_syntax,
    get_compilation_errors,
)

__all__ = [
    # Manager
    "MCPClientManager",
    # Context7
    "get_mql5_documentation",
    "get_mql5_examples",
    # Sequential Thinking
    "sequential_thinking",
    "analyze_errors",
    # PageIndex
    "index_pdf",
    "search_pdf",
    "get_indexed_documents",
    # Backtest
    "run_backtest",
    "get_backtest_status",
    "get_backtest_results",
    "compare_backtests",
    # MT5 Compiler
    "compile_mql5_code",
    "validate_mql5_syntax",
    "get_compilation_errors",
]
