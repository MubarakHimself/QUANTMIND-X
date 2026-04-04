"""
Tool Registry for department-based tool access.

The old registry tried to infer classes from filenames. That drifted badly from
the actual production tool surface and caused the department runtime to expose
missing or schema-less tools. This registry now uses explicit SDK-style tool
adapters so every loaded tool has:

- a stable `name`
- a human-readable `description`
- a real JSON `input_schema`
- an `execute()` entrypoint for the department head tool loop
"""

from __future__ import annotations

import importlib
import inspect
import json
from logging import getLogger
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .types import Department
from .tool_access import ToolAccessController, ToolPermission

logger = getLogger(__name__)


def _json_default(value: Any) -> Any:
    """Serialize common dataclass/enum/path-style objects for tool responses."""
    if hasattr(value, "__dict__"):
        return value.__dict__
    if hasattr(value, "value"):
        return value.value
    return str(value)


class SDKToolAdapter:
    """Minimal Anthropic/Claude-compatible tool wrapper."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.requires_approval = requires_approval
        # Keep the public field name aligned with the Claude Agent SDK contract.
        self.handler = handler

    def as_sdk_definition(self) -> Dict[str, Any]:
        """Expose an SDK-style tool definition for future direct SDK wiring."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "handler": self.handler,
        }

    async def execute(self, tool_input: Dict[str, Any]) -> str:
        result = self.handler(tool_input)
        if inspect.isawaitable(result):
            result = await result
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> str:
        """
        Normalize handler output for the current Anthropic messages loop.

        The official Agent SDK expects tool handlers to return structured dicts
        like {"content": [{"type": "text", "text": "..."}]}. Our department
        loop still feeds tool results back through `tool_result.content`, which
        is safest today as a string. This method accepts both contracts.
        """
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                text_chunks: List[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            text_chunks.append(text)
                if text_chunks:
                    return "\n".join(text_chunks)
            return json.dumps(result, default=_json_default, ensure_ascii=False)
        return json.dumps(result, default=_json_default, ensure_ascii=False)


class ToolRegistry:
    """
    Central registry for managing tool access by department.

    Exposes only tools that are actually loadable in the current runtime.
    """

    _tool_instances: Dict[str, Any] = {}

    @classmethod
    def get_tools_for_department(
        cls,
        department: Department,
    ) -> Dict[str, Any]:
        access_controller = ToolAccessController(department)
        tools: Dict[str, Any] = {}
        for tool_name in access_controller.get_available_tools():
            if not access_controller.has_read_access(tool_name):
                continue
            tool_instance = cls._get_tool_instance(tool_name, department)
            if tool_instance is not None:
                tools[tool_name] = tool_instance
        logger.info("Provided %s tools to %s", len(tools), department.value)
        return tools

    @classmethod
    def get_tool(
        cls,
        tool_name: str,
        department: Department,
    ) -> Optional[Any]:
        access_controller = ToolAccessController(department)
        if not access_controller.has_read_access(tool_name):
            logger.warning("%s denied access to %s", department.value, tool_name)
            return None
        return cls._get_tool_instance(tool_name, department)

    @classmethod
    def _get_cache_key(cls, tool_name: str, department: Optional[Department]) -> str:
        if department is None:
            return tool_name
        return f"{tool_name}:{department.value}"

    @classmethod
    def _get_tool_instance(
        cls,
        tool_name: str,
        department: Optional[Department] = None,
    ) -> Optional[Any]:
        cache_key = cls._get_cache_key(tool_name, department)
        if cache_key in cls._tool_instances:
            return cls._tool_instances[cache_key]

        try:
            instance = cls._build_tool_instance(tool_name, department)
        except Exception as exc:
            logger.error("Failed to load tool %s: %s", tool_name, exc)
            return None

        if instance is None:
            logger.info("Tool disabled or unavailable in current runtime: %s", tool_name)
            return None

        cls._tool_instances[cache_key] = instance
        logger.info("Initialized tool: %s", cache_key)
        return instance

    @classmethod
    def _build_tool_instance(
        cls,
        tool_name: str,
        department: Optional[Department],
    ) -> Optional[Any]:
        builders: Dict[str, Callable[[Optional[Department]], Optional[Any]]] = {
            "memory_tools": cls._build_memory_tools,
            "memory_all_depts": cls._build_memory_all_depts_tool,
            "knowledge_tools": cls._build_knowledge_tools,
            "knowledge_uploads": cls._build_knowledge_uploads_tool,
            "backtest_tools": cls._build_backtest_tools,
            "risk_tools": cls._build_risk_tools,
            "strategy_router": cls._build_strategy_router_tool,
            "strategy_extraction": cls._build_strategy_extraction_tool,
            "pinescript_tools": cls._build_pinescript_tools,
            "mql5_tools": cls._build_mql5_tools,
            "ea_lifecycle": cls._build_ea_lifecycle_tool,
            "gemini_cli": cls._build_gemini_cli_tool,
            "prop_firm_research": cls._build_prop_firm_research_tool,
            "task_list": cls._build_task_list_tool,
            "shared_assets": cls._build_shared_assets_tool,
            "video_ingest": cls._build_video_ingest_tool,
            "mail": cls._build_mail_tool,
        }

        builder = builders.get(tool_name)
        if builder is not None:
            return builder(department)

        # Keep explicit class-based fallbacks only for tools that still match.
        fallback_specs = {
            "broker_tools": ("src.agents.tools.broker_tools", "BrokerTools"),
        }
        spec = fallback_specs.get(tool_name)
        if spec is None:
            return None
        module = importlib.import_module(spec[0])
        tool_class = getattr(module, spec[1], None)
        if tool_class is None:
            return None
        # BrokerTools returns hardcoded account/position data today, so do not
        # expose it until a live bridge is configured.
        if tool_name == "broker_tools":
            return None
        return tool_class()

    @staticmethod
    def _build_memory_tools(department: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.memory_tools import (
            add_memory,
            delete_memory,
            get_all_memories,
            get_memory,
            search_memories,
        )

        default_department = department.value if department else "research"

        async def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "search")).strip().lower()
            target_department = str(args.get("department") or default_department)
            if operation == "add":
                return await add_memory(
                    content=str(args["content"]),
                    department=target_department,
                    importance=float(args.get("importance", 0.5)),
                    tags=args.get("tags"),
                    metadata=args.get("metadata"),
                )
            if operation == "get":
                return await get_memory(str(args["memory_id"]))
            if operation == "delete":
                return await delete_memory(str(args["memory_id"]))
            if operation == "list":
                return await get_all_memories(
                    department=target_department,
                    limit=int(args.get("limit", 100)),
                )
            return await search_memories(
                query=str(args.get("query", "")),
                department=target_department,
                limit=int(args.get("limit", 10)),
            )

        return SDKToolAdapter(
            name="memory_tools",
            description="Search, list, add, fetch, or delete department memory entries.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["search", "list", "add", "get", "delete"],
                        "default": "search",
                    },
                    "query": {"type": "string"},
                    "department": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "content": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "memory_id": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_memory_all_depts_tool(_: Optional[Department]) -> SDKToolAdapter:
        adapter = ToolRegistry._build_memory_tools(None)
        adapter.description = "Search or inspect memory across departments."
        return adapter

    @staticmethod
    def _build_knowledge_tools(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.knowledge_tools import (
            get_article_content,
            get_indexing_status,
            get_indicator_template,
            get_mql5_book_section,
            index_pdf_document,
            list_indexed_documents,
            list_knowledge_namespaces,
            remove_indexed_document,
            search_knowledge_hub,
            search_mql5_book,
            search_strategy_patterns,
        )

        async def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "search_knowledge_hub")).strip().lower()
            if operation == "search_mql5_book":
                return await search_mql5_book(str(args.get("query", "")))
            if operation == "get_mql5_book_section":
                return await get_mql5_book_section(str(args.get("section", "")))
            if operation == "get_article_content":
                return await get_article_content(str(args.get("article_id", "")))
            if operation == "list_namespaces":
                return await list_knowledge_namespaces()
            if operation == "index_pdf_document":
                return await index_pdf_document(
                    pdf_path=str(args.get("pdf_path", "")),
                    namespace=str(args.get("namespace", "custom")),
                    metadata=args.get("metadata"),
                )
            if operation == "get_indexing_status":
                return await get_indexing_status(str(args.get("job_id", "")))
            if operation == "list_indexed_documents":
                return await list_indexed_documents(str(args.get("namespace", "custom")))
            if operation == "remove_indexed_document":
                return await remove_indexed_document(
                    namespace=str(args.get("namespace", "custom")),
                    document_id=str(args.get("document_id", "")),
                )
            if operation == "search_strategy_patterns":
                return await search_strategy_patterns(str(args.get("query", "")))
            if operation == "get_indicator_template":
                return await get_indicator_template(str(args.get("indicator", "")))
            return await search_knowledge_hub(
                query=str(args.get("query", "")),
                namespace=args.get("namespace"),
                limit=int(args.get("limit", 10)),
            )

        return SDKToolAdapter(
            name="knowledge_tools",
            description="Search PageIndex knowledge, MQL5 books, indexed PDFs, and article content.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "search_knowledge_hub",
                            "get_article_content",
                            "search_mql5_book",
                            "get_mql5_book_section",
                            "list_namespaces",
                            "index_pdf_document",
                            "get_indexing_status",
                            "list_indexed_documents",
                            "remove_indexed_document",
                            "search_strategy_patterns",
                            "get_indicator_template",
                        ],
                        "default": "search_knowledge_hub",
                    },
                    "query": {"type": "string"},
                    "namespace": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "article_id": {"type": "string"},
                    "section": {"type": "string"},
                    "pdf_path": {"type": "string"},
                    "metadata": {"type": "object"},
                    "job_id": {"type": "string"},
                    "document_id": {"type": "string"},
                    "indicator": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_knowledge_uploads_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.knowledge_tools import (
            get_indexing_status,
            index_pdf_document,
            list_indexed_documents,
            remove_indexed_document,
        )

        async def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "index")).strip().lower()
            if operation == "status":
                return await get_indexing_status(str(args.get("job_id", "")))
            if operation == "list":
                return await list_indexed_documents(str(args.get("namespace", "custom")))
            if operation == "remove":
                return await remove_indexed_document(
                    namespace=str(args.get("namespace", "custom")),
                    document_id=str(args.get("document_id", "")),
                )
            return await index_pdf_document(
                pdf_path=str(args.get("pdf_path", "")),
                namespace=str(args.get("namespace", "custom")),
                metadata=args.get("metadata"),
            )

        return SDKToolAdapter(
            name="knowledge_uploads",
            description="Index, inspect, or remove uploaded knowledge documents.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["index", "status", "list", "remove"],
                        "default": "index",
                    },
                    "pdf_path": {"type": "string"},
                    "namespace": {"type": "string"},
                    "metadata": {"type": "object"},
                    "job_id": {"type": "string"},
                    "document_id": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_backtest_tools(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.backtest_tools import BacktestConfig, BacktestTools

        tool = BacktestTools()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "run_backtest")).strip().lower()
            if operation != "run_backtest":
                return {"success": False, "error": f"Unsupported operation: {operation}"}
            config = BacktestConfig(
                strategy_name=str(args["strategy_name"]),
                symbol=str(args["symbol"]),
                timeframe=str(args["timeframe"]),
                start_date=str(args["start_date"]),
                end_date=str(args["end_date"]),
                initial_deposit=float(args.get("initial_deposit", 10000.0)),
                lot_size=float(args.get("lot_size", 0.01)),
                spread=int(args.get("spread", 0)),
                stop_loss=args.get("stop_loss"),
                take_profit=args.get("take_profit"),
            )
            result = tool.run_backtest(
                config=config,
                strategy_code=str(args["strategy_code"]),
                variant=str(args.get("variant", "vanilla")),
            )
            return result

        return SDKToolAdapter(
            name="backtest_tools",
            description="Run production backtests on real market data via the QuantMind backtesting stack.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["run_backtest"], "default": "run_backtest"},
                    "strategy_name": {"type": "string"},
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "initial_deposit": {"type": "number"},
                    "lot_size": {"type": "number"},
                    "spread": {"type": "integer"},
                    "stop_loss": {"type": "integer"},
                    "take_profit": {"type": "integer"},
                    "strategy_code": {"type": "string"},
                    "variant": {"type": "string"},
                },
                "required": [
                    "strategy_name",
                    "symbol",
                    "timeframe",
                    "start_date",
                    "end_date",
                    "strategy_code",
                ],
            },
            handler=handler,
        )

    @staticmethod
    def _build_risk_tools(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.risk_tools import RiskTools

        tool = RiskTools()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "check_risk_limits")).strip().lower()
            if operation == "calculate_position_size":
                return tool.calculate_position_size(
                    account_balance=float(args["account_balance"]),
                    risk_percent=float(args["risk_percent"]),
                    entry_price=float(args["entry_price"]),
                    stop_loss_price=float(args["stop_loss_price"]),
                )
            if operation == "calculate_var":
                return {
                    "var": tool.calculate_var(
                        returns=list(args.get("returns", [])),
                        confidence_level=float(args.get("confidence_level", 0.95)),
                    )
                }
            if operation == "calculate_max_drawdown":
                return {"max_drawdown": tool.calculate_max_drawdown(list(args.get("equity_curve", [])))}
            if operation == "get_current_drawdown":
                return tool.get_current_drawdown(
                    current_equity=float(args["current_equity"]),
                    peak_equity=float(args["peak_equity"]),
                )
            if operation == "calculate_risk_reward_ratio":
                return {
                    "risk_reward_ratio": tool.calculate_risk_reward_ratio(
                        entry_price=float(args["entry_price"]),
                        take_profit=float(args["take_profit"]),
                        stop_loss=float(args["stop_loss"]),
                    )
                }
            return tool.check_risk_limits(
                position_size=float(args["position_size"]),
                account_balance=float(args["account_balance"]),
                max_position_percent=float(args.get("max_position_percent", 10.0)),
            )

        return SDKToolAdapter(
            name="risk_tools",
            description="Calculate position sizing, drawdown, VaR, and read-only risk checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "calculate_position_size",
                            "calculate_var",
                            "calculate_max_drawdown",
                            "get_current_drawdown",
                            "calculate_risk_reward_ratio",
                            "check_risk_limits",
                        ],
                        "default": "check_risk_limits",
                    },
                    "account_balance": {"type": "number"},
                    "risk_percent": {"type": "number"},
                    "entry_price": {"type": "number"},
                    "stop_loss_price": {"type": "number"},
                    "returns": {"type": "array", "items": {"type": "number"}},
                    "confidence_level": {"type": "number"},
                    "equity_curve": {"type": "array", "items": {"type": "number"}},
                    "current_equity": {"type": "number"},
                    "peak_equity": {"type": "number"},
                    "take_profit": {"type": "number"},
                    "stop_loss": {"type": "number"},
                    "position_size": {"type": "number"},
                    "max_position_percent": {"type": "number"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_strategy_router_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.strategy_router import StrategyRouter, StrategyStatus

        router = StrategyRouter()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "list_strategies")).strip().lower()
            if operation == "search_strategies":
                return router.search_strategies(
                    query=str(args.get("query", "")),
                    limit=int(args.get("limit", 10)),
                )
            if operation == "get_strategy":
                return router.get_strategy(str(args.get("strategy_id", "")))
            if operation == "get_strategy_status":
                return router.get_strategy_status(str(args.get("strategy_id", "")))
            if operation == "get_strategy_performance":
                return router.get_strategy_performance(str(args.get("strategy_id", "")))
            if operation == "get_strategy_count_by_status":
                return router.get_strategy_count_by_status()
            status = args.get("status")
            return router.list_strategies(
                status=StrategyStatus(status) if status else None,
                department=args.get("department"),
            )

        return SDKToolAdapter(
            name="strategy_router",
            description="Inspect strategy registry state and status without mutating live routing.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "list_strategies",
                            "search_strategies",
                            "get_strategy",
                            "get_strategy_status",
                            "get_strategy_performance",
                            "get_strategy_count_by_status",
                        ],
                        "default": "list_strategies",
                    },
                    "strategy_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "status": {"type": "string"},
                    "department": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_strategy_extraction_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.strategy_extraction import StrategyExtraction

        tool = StrategyExtraction()

        def extract(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "extract_from_text")).strip().lower()
            if operation == "extract_from_video":
                return tool.extract_from_video(
                    video_url=str(args["video_url"]),
                    use_transcript=bool(args.get("use_transcript", True)),
                )
            if operation == "extract_from_pdf":
                return tool.extract_from_pdf(pdf_path=str(args["pdf_path"]))
            return tool.extract_from_text(text=str(args["text"]))

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "extract_from_text")).strip().lower()
            strategy = extract(args)
            if operation == "validate":
                return tool.validate_strategy(strategy)
            if operation == "generate_trd":
                return tool.generate_trd(strategy)
            return strategy

        return SDKToolAdapter(
            name="strategy_extraction",
            description="Extract, validate, and convert trading strategies from text, PDF, or video sources.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "extract_from_text",
                            "extract_from_pdf",
                            "extract_from_video",
                            "validate",
                            "generate_trd",
                        ],
                        "default": "extract_from_text",
                    },
                    "text": {"type": "string"},
                    "pdf_path": {"type": "string"},
                    "video_url": {"type": "string"},
                    "use_transcript": {"type": "boolean"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_pinescript_tools(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.pinescript_tools import (
            validate_pine_script_strategy,
            validate_pine_script_syntax,
        )

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "validate_syntax")).strip().lower()
            pine_code = str(args.get("pine_code", ""))
            if operation == "validate_strategy":
                return validate_pine_script_strategy(pine_code)
            return validate_pine_script_syntax(pine_code)

        return SDKToolAdapter(
            name="pinescript_tools",
            description="Validate Pine Script syntax and strategy completeness.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["validate_syntax", "validate_strategy"],
                        "default": "validate_syntax",
                    },
                    "pine_code": {"type": "string"},
                },
                "required": ["pine_code"],
            },
            handler=handler,
        )

    @staticmethod
    def _build_mql5_tools(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.mql5_tools import MQL5Tools

        tool = MQL5Tools()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "validate_mql5_syntax")).strip().lower()
            if operation == "generate_ea_from_strategy":
                return tool.generate_ea_from_strategy(
                    strategy_name=str(args["strategy_name"]),
                    strategy_description=str(args.get("strategy_description", "")),
                    entry_conditions=list(args.get("entry_conditions", [])),
                    exit_conditions=list(args.get("exit_conditions", [])),
                    risk_params=dict(args.get("risk_params", {})),
                )
            if operation == "pinescript_to_mql5":
                return tool.pinescript_to_mql5(str(args.get("pinescript_code", "")))
            return tool.validate_mql5_syntax(str(args.get("code", "")))

        return SDKToolAdapter(
            name="mql5_tools",
            description="Generate MQL5 EAs, validate syntax, and convert Pine Script to MQL5.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "generate_ea_from_strategy",
                            "validate_mql5_syntax",
                            "pinescript_to_mql5",
                        ],
                        "default": "validate_mql5_syntax",
                    },
                    "strategy_name": {"type": "string"},
                    "strategy_description": {"type": "string"},
                    "entry_conditions": {"type": "array", "items": {"type": "string"}},
                    "exit_conditions": {"type": "array", "items": {"type": "string"}},
                    "risk_params": {"type": "object"},
                    "code": {"type": "string"},
                    "pinescript_code": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_ea_lifecycle_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.ea_lifecycle import EALifecycleManager

        manager = EALifecycleManager()

        def handler(args: Dict[str, Any]) -> Any:
            return manager.create_ea_from_trd(
                trd_variants=dict(args.get("trd_variants", {})),
                create_variants=str(args.get("create_variants", "both")),
            )

        return SDKToolAdapter(
            name="ea_lifecycle",
            description="Create vanilla and spiced EA variants from TRD strategy definitions.",
            input_schema={
                "type": "object",
                "properties": {
                    "trd_variants": {"type": "object"},
                    "create_variants": {
                        "type": "string",
                        "enum": ["vanilla", "spiced", "both"],
                        "default": "both",
                    },
                },
                "required": ["trd_variants"],
            },
            handler=handler,
        )

    @staticmethod
    def _build_gemini_cli_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.gemini_cli_tool import GeminiCLITool

        tool = GeminiCLITool()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "research")).strip().lower()
            if operation == "is_available":
                return {"available": tool.is_available()}
            if operation == "get_version":
                return tool.get_version()
            return tool.research(
                query=str(args.get("query", "")),
                context=args.get("context"),
            )

        return SDKToolAdapter(
            name="gemini_cli",
            description="Run external Gemini CLI research when configured on the host.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["research", "is_available", "get_version"],
                        "default": "research",
                    },
                    "query": {"type": "string"},
                    "context": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_prop_firm_research_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.prop_firm_research_tool import PropFirmResearch

        tool = PropFirmResearch()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "firm_analysis")).strip().lower()
            if operation == "get_firms":
                return {"firms": tool.get_firms()}
            if operation == "get_tiers":
                return {"tiers": tool.get_tiers(str(args.get("firm", "")))}
            if operation == "get_rules":
                return tool.get_rules(
                    firm=str(args.get("firm", "")),
                    tier=str(args.get("tier", "standard")),
                )
            if operation == "compare_firms":
                return tool.compare_firms(
                    firms=list(args.get("firms", [])),
                    tier=str(args.get("tier", "standard")),
                )
            return tool.firm_analysis(
                firm=str(args.get("firm", "")),
                tier=str(args.get("tier", "standard")),
            )

        return SDKToolAdapter(
            name="prop_firm_research",
            description="Inspect prop firm rules, scoring, and comparisons.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["get_firms", "get_tiers", "get_rules", "firm_analysis", "compare_firms"],
                        "default": "firm_analysis",
                    },
                    "firm": {"type": "string"},
                    "tier": {"type": "string"},
                    "firms": {"type": "array", "items": {"type": "string"}},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_task_list_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.task_list_tool import TaskListTool, TaskStatus, get_task_list_tool

        tool: TaskListTool = get_task_list_tool()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "list_tasks")).strip().lower()
            if operation == "create_task":
                return tool.create_task(
                    task_id=str(args["task_id"]),
                    title=str(args["title"]),
                    description=str(args.get("description", "")),
                    priority=int(args.get("priority", 0)),
                    assignee=args.get("assignee"),
                    tags=args.get("tags"),
                    metadata=args.get("metadata"),
                )
            if operation == "get_task":
                return tool.get_task(str(args["task_id"]))
            if operation == "update_task":
                status = args.get("status")
                return tool.update_task(
                    task_id=str(args["task_id"]),
                    title=args.get("title"),
                    description=args.get("description"),
                    status=TaskStatus(status) if status else None,
                    priority=args.get("priority"),
                    assignee=args.get("assignee"),
                    tags=args.get("tags"),
                    metadata=args.get("metadata"),
                )
            if operation == "delete_task":
                return tool.delete_task(str(args["task_id"]))
            if operation == "update_task_status":
                return tool.update_task_status(
                    task_id=str(args["task_id"]),
                    status=TaskStatus(str(args["status"])),
                )
            status = args.get("status")
            return tool.list_tasks(
                status=TaskStatus(status) if status else None,
                assignee=args.get("assignee"),
                tags=args.get("tags"),
                limit=int(args.get("limit", 100)),
            )

        return SDKToolAdapter(
            name="task_list",
            description="Create, inspect, update, delete, and list persistent department task records.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "create_task",
                            "get_task",
                            "update_task",
                            "delete_task",
                            "list_tasks",
                            "update_task_status",
                        ],
                        "default": "list_tasks",
                    },
                    "task_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "integer"},
                    "assignee": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_shared_assets_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.agents.tools.shared_assets_tool import get_default_instance

        tool = get_default_instance()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "list")).strip().lower()
            if operation == "upload":
                return tool.upload(
                    file_path=str(args["file_path"]),
                    name=str(args["name"]),
                    category=str(args["category"]),
                    description=str(args.get("description", "")),
                    tags=args.get("tags"),
                )
            if operation == "download":
                return tool.download(
                    asset_id=str(args["asset_id"]),
                    destination=str(args["destination"]),
                )
            if operation == "get_asset":
                return tool.get_asset(str(args["asset_id"]))
            if operation == "delete":
                return tool.delete(str(args["asset_id"]))
            if operation == "update":
                return tool.update(
                    asset_id=str(args["asset_id"]),
                    name=args.get("name"),
                    description=args.get("description"),
                    tags=args.get("tags"),
                )
            if operation == "get_categories":
                return {"categories": tool.get_categories()}
            if operation == "get_stats":
                return tool.get_stats()
            return tool.list(
                category=args.get("category"),
                tags=args.get("tags"),
                search=args.get("search"),
                page=int(args.get("page", 1)),
                limit=int(args.get("limit", 50)),
                sort=str(args.get("sort", "created_at")),
                order=str(args.get("order", "desc")),
            )

        return SDKToolAdapter(
            name="shared_assets",
            description="List, upload, update, and retrieve shared cross-department assets.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "list",
                            "upload",
                            "download",
                            "get_asset",
                            "delete",
                            "update",
                            "get_categories",
                            "get_stats",
                        ],
                        "default": "list",
                    },
                    "asset_id": {"type": "string"},
                    "file_path": {"type": "string"},
                    "destination": {"type": "string"},
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "search": {"type": "string"},
                    "page": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "sort": {"type": "string"},
                    "order": {"type": "string"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_video_ingest_tool(_: Optional[Department]) -> SDKToolAdapter:
        from src.video_ingest.tool import VideoIngestTool

        tool = VideoIngestTool()

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "get_video_info")).strip().lower()
            if operation == "calculate_chunks":
                return {
                    "chunks": tool.calculate_chunks(
                        duration_seconds=int(args["duration_seconds"]),
                    )
                }
            return tool.get_video_info(str(args["url"]))

        return SDKToolAdapter(
            name="video_ingest",
            description="Inspect video metadata and chunking for ingest workflows.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["get_video_info", "calculate_chunks"],
                        "default": "get_video_info",
                    },
                    "url": {"type": "string"},
                    "duration_seconds": {"type": "integer"},
                },
            },
            handler=handler,
        )

    @staticmethod
    def _build_mail_tool(department: Optional[Department]) -> SDKToolAdapter:
        from src.agents.departments.department_mail import (
            MessageType,
            Priority,
            create_mail_service,
        )

        dept_name = department.value if department else "research"
        mail_service = create_mail_service(
            db_path=".quantmind/department_mail.db",
            use_redis=True,
            consumer_name=f"tool-registry-{dept_name}",
        )
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }

        def handler(args: Dict[str, Any]) -> Any:
            operation = str(args.get("operation", "check_inbox")).strip().lower()
            if operation == "send":
                message = mail_service.send(
                    from_dept=dept_name,
                    to_dept=str(args["to_dept"]),
                    type=MessageType.RESULT,
                    subject=str(args.get("subject", "(no subject)")),
                    body=str(args.get("body", "")),
                    priority=priority_map.get(
                        str(args.get("priority", "normal")).lower(),
                        Priority.NORMAL,
                    ),
                )
                return {"status": "sent", "message_id": message.id}
            inbox = mail_service.check_inbox(
                dept=str(args.get("department") or dept_name),
                unread_only=bool(args.get("unread_only", True)),
            )
            return {
                "department": str(args.get("department") or dept_name),
                "messages": [message.to_dict() for message in inbox],
                "count": len(inbox),
            }

        return SDKToolAdapter(
            name="mail",
            description="Check department inboxes or send cross-department mail.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["check_inbox", "send"],
                        "default": "check_inbox",
                    },
                    "department": {"type": "string"},
                    "unread_only": {"type": "boolean"},
                    "to_dept": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {"type": "string"},
                },
            },
            handler=handler,
        )

    @classmethod
    def can_use_tool(
        cls,
        tool_name: str,
        department: Department,
        permission: ToolPermission = ToolPermission.READ,
    ) -> bool:
        access_controller = ToolAccessController(department)
        return access_controller.can_access(tool_name, permission)

    @classmethod
    def get_tool_info(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        return {"name": tool_name, "available": True}

    @classmethod
    def list_all_tools(cls) -> List[str]:
        from .tool_access import TOOL_ACCESS

        tool_names = set()
        for tool_map in TOOL_ACCESS.values():
            tool_names.update(tool_map.keys())
        return sorted(tool_names)

    @classmethod
    def clear_cache(cls) -> None:
        cls._tool_instances.clear()
        logger.info("Tool cache cleared")


def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()
