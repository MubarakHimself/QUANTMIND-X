import asyncio
import json

import pytest

from src.agents.departments.tool_registry import ToolRegistry
from src.agents.departments.types import Department


def test_active_department_tools_expose_real_input_schemas():
    expected = {
        Department.RESEARCH: {"memory_tools", "knowledge_tools", "backtest_tools", "task_list", "mail", "strategy_extraction"},
        Department.DEVELOPMENT: {"memory_tools", "knowledge_tools", "backtest_tools", "mql5_tools", "pinescript_tools", "ea_lifecycle", "strategy_extraction"},
        Department.RISK: {"memory_tools", "knowledge_tools", "risk_tools", "task_list", "shared_assets"},
        Department.TRADING: {"memory_tools", "knowledge_tools", "risk_tools", "task_list", "shared_assets"},
        Department.PORTFOLIO: {"memory_tools", "knowledge_tools", "risk_tools", "task_list", "shared_assets", "strategy_extraction"},
    }

    ToolRegistry.clear_cache()

    for department, required_tools in expected.items():
        tools = ToolRegistry.get_tools_for_department(department)
        assert required_tools.issubset(set(tools.keys()))
        for name, tool in tools.items():
            assert getattr(tool, "input_schema", None), f"{department.value}:{name} missing input_schema"


def test_task_list_tool_adapter_executes_create_and_list():
    ToolRegistry.clear_cache()
    tool = ToolRegistry.get_tool("task_list", Department.RESEARCH)
    assert tool is not None

    task_id = "registry-runtime-test-task"
    existing = json.loads(asyncio.run(tool.execute({"operation": "get_task", "task_id": task_id})))
    if existing.get("success"):
        asyncio.run(tool.execute({"operation": "delete_task", "task_id": task_id}))

    create_result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "operation": "create_task",
                    "task_id": task_id,
                    "title": "Registry Runtime Test Task",
                    "description": "created through adapter",
                }
            )
        )
    )
    assert create_result["success"] is True
    assert create_result["task"]["id"] == task_id

    list_result = json.loads(asyncio.run(tool.execute({"operation": "list_tasks", "limit": 20})))
    assert list_result["success"] is True
    assert any(task["id"] == task_id for task in list_result["tasks"])


def test_sdk_tool_adapter_exposes_sdk_shape_and_normalizes_content_blocks():
    from src.agents.departments.tool_registry import SDKToolAdapter

    async def handler(args):
        return {
            "content": [
                {"type": "text", "text": f"Hello {args['name']}"},
                {"type": "text", "text": "Tool output normalized"},
            ]
        }

    adapter = SDKToolAdapter(
        name="greet",
        description="Greet a user",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        handler=handler,
    )

    sdk_shape = adapter.as_sdk_definition()
    assert sdk_shape["name"] == "greet"
    assert sdk_shape["description"] == "Greet a user"
    assert sdk_shape["input_schema"]["type"] == "object"
    assert sdk_shape["handler"] is handler

    result = asyncio.run(adapter.execute({"name": "QuantMind"}))
    assert result == "Hello QuantMind\nTool output normalized"


def test_mql5_tool_adapter_executes_validation():
    ToolRegistry.clear_cache()
    tool = ToolRegistry.get_tool("mql5_tools", Department.DEVELOPMENT)
    assert tool is not None

    result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "operation": "validate_mql5_syntax",
                    "code": (
                        "#property strict\n"
                        "int OnInit(){return(INIT_SUCCEEDED);}\n"
                        "void OnDeinit(){}\n"
                        "void OnTick(){}\n"
                    ),
                }
            )
        )
    )
    assert result["valid"] is True


def test_strategy_extraction_tool_adapter_executes_extract_and_validate():
    ToolRegistry.clear_cache()
    tool = ToolRegistry.get_tool("strategy_extraction", Department.RESEARCH)
    assert tool is not None

    extract_result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "operation": "extract_from_text",
                    "text": (
                        "Entry Rules\n"
                        "Buy when price closes above EMA 50\n"
                        "Exit Rules\n"
                        "Close when RSI exceeds 70\n"
                    ),
                }
            )
        )
    )
    assert extract_result["name"] == "Strategy from Text"
    assert "Buy when price closes above EMA 50" in extract_result["entry_conditions"]

    validate_result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "operation": "validate",
                    "text": (
                        "Entry Rules\n"
                        "Buy when price closes above EMA 50\n"
                        "Exit Rules\n"
                        "Close when RSI exceeds 70\n"
                    ),
                }
            )
        )
    )
    assert validate_result["valid"] is True
