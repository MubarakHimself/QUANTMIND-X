"""
NPRD to EA Workflow.

Implements the automatic pipeline from Natural Product Requirements
Document to compiled Expert Advisor.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import (
    WorkflowState,
    WorkflowStep,
    WorkflowStatus,
    WorkflowResult,
    StepStatus,
    AgentType,
    get_workflow_store,
)
from ..agents.tools.registry import tool_registry, AgentType as ToolAgentType
from ..agents.tools.nprd_trd import ParseNPRDTool, ValidateNPRDTool, GenerateTRDTool
from ..agents.tools.quantcode import GenerateMQL5Tool, CompileMQL5Tool


logger = logging.getLogger(__name__)


# Define the steps for NPRD → EA workflow
NPRD_TO_EA_STEPS = [
    {
        "name": "parse_nprd",
        "description": "Parse the Natural Product Requirements Document",
        "agent_type": AgentType.ANALYST,
    },
    {
        "name": "validate_nprd",
        "description": "Validate NPRD structure and completeness",
        "agent_type": AgentType.ANALYST,
    },
    {
        "name": "generate_trd",
        "description": "Generate Technical Requirements Document",
        "agent_type": AgentType.ANALYST,
    },
    {
        "name": "generate_mql5",
        "description": "Generate MQL5 code from TRD",
        "agent_type": AgentType.QUANTCODE,
    },
    {
        "name": "validate_syntax",
        "description": "Validate MQL5 code syntax",
        "agent_type": AgentType.QUANTCODE,
    },
    {
        "name": "compile_ea",
        "description": "Compile the Expert Advisor",
        "agent_type": AgentType.QUANTCODE,
    },
]


class NPRDToEATWorkflow:
    """
    Workflow for converting NPRD to EA.

    Steps:
    1. Parse NPRD
    2. Validate NPRD
    3. Generate TRD
    4. Generate MQL5 code
    5. Validate syntax
    6. Compile EA
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        auto_continue: bool = True,
    ):
        self.workspace_path = workspace_path
        self.auto_continue = auto_continue
        self._store = get_workflow_store()

        # Initialize tools
        self._parse_nprd = ParseNPRDTool()
        self._validate_nprd = ValidateNPRDTool()
        self._generate_trd = GenerateTRDTool()
        self._generate_mql5 = GenerateMQL5Tool()
        self._compile_ea = CompileMQL5Tool()

        if workspace_path:
            self._parse_nprd.set_workspace(workspace_path)
            self._validate_nprd.set_workspace(workspace_path)
            self._generate_trd.set_workspace(workspace_path)
            self._generate_mql5.set_workspace(workspace_path)
            self._compile_ea.set_workspace(workspace_path)

    def create_state(
        self,
        nprd_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Create initial workflow state.

        Args:
            nprd_content: The NPRD content
            metadata: Additional metadata

        Returns:
            Initialized workflow state
        """
        state = WorkflowState(
            workflow_type="nprd_to_ea",
            auto_continue=self.auto_continue,
            input_data={
                "nprd_content": nprd_content,
                **(metadata or {}),
            },
        )

        # Add steps
        for i, step_def in enumerate(NPRD_TO_EA_STEPS):
            step = WorkflowStep(
                step_id=f"step_{i+1}_{step_def['name']}",
                name=step_def["name"],
                description=step_def["description"],
                agent_type=step_def["agent_type"],
            )
            state.add_step(step)

        self._store.save(state)
        return state

    async def run(
        self,
        nprd_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Run the complete workflow.

        Args:
            nprd_content: The NPRD content
            metadata: Additional metadata

        Returns:
            Workflow result
        """
        state = self.create_state(nprd_content, metadata)

        try:
            # Start workflow
            state.start()
            self._store.save(state)

            # Execute steps
            for step in state.steps:
                result = await self._execute_step(state, step)

                if not result["success"]:
                    if step.can_retry():
                        step.retry()
                        result = await self._execute_step(state, step)
                        if not result["success"]:
                            state.fail(f"Step '{step.name}' failed after retries")
                            break
                    else:
                        state.fail(f"Step '{step.name}' failed: {result.get('error')}")
                        break

                if not self.auto_continue:
                    break

                state.advance()

            # Complete if all steps done
            if state.status == WorkflowStatus.RUNNING:
                state.complete({
                    "ea_path": state.get_intermediate("ea_path"),
                    "mql5_code": state.get_intermediate("mql5_code"),
                })

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state.fail(str(e))

        self._store.save(state)

        return WorkflowResult(
            workflow_id=state.workflow_id,
            success=state.status == WorkflowStatus.COMPLETED,
            status=state.status,
            final_output=state.final_result,
            intermediate_outputs=state.intermediate_results,
            duration_seconds=state.duration_seconds or 0,
            steps_completed=len(state.get_completed_steps()),
            steps_total=len(state.steps),
            error=state.error,
        )

    async def _execute_step(
        self,
        state: WorkflowState,
        step: WorkflowStep,
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step.start()
        logger.info(f"Executing step: {step.name}")

        try:
            if step.name == "parse_nprd":
                result = self._parse_nprd.execute(
                    nprd_content=state.input_data.get("nprd_content")
                )
                step.complete(result.data)
                state.set_intermediate("parsed_nprd", result.data)

            elif step.name == "validate_nprd":
                result = self._validate_nprd.execute(
                    nprd_content=state.input_data.get("nprd_content")
                )
                step.complete(result.data)
                state.set_intermediate("nprd_validation", result.data)

                if not result.data.get("is_valid"):
                    return {"success": False, "error": "NPRD validation failed"}

            elif step.name == "generate_trd":
                result = self._generate_trd.execute(
                    nprd_content=state.input_data.get("nprd_content")
                )
                step.complete(result.data)
                state.set_intermediate("trd", result.data)

            elif step.name == "generate_mql5":
                trd_content = state.get_intermediate("trd", {}).get("content", "")
                result = self._generate_mql5.execute(
                    trd_content=trd_content
                )
                step.complete(result.data)
                state.set_intermediate("mql5_code", result.data.get("code"))

            elif step.name == "validate_syntax":
                mql5_code = state.get_intermediate("mql5_code", "")
                from ..agents.tools.quantcode import ValidateSyntaxTool
                validate_tool = ValidateSyntaxTool()
                result = validate_tool.execute(code=mql5_code)
                step.complete(result.data)

                if not result.data.get("is_valid"):
                    return {"success": False, "error": "Syntax validation failed"}

            elif step.name == "compile_ea":
                mql5_code = state.get_intermediate("mql5_code", "")
                result = self._compile_ea.execute(code=mql5_code)
                step.complete(result.data)

                if not result.data.get("success"):
                    return {"success": False, "error": "Compilation failed"}

                state.set_intermediate("ea_path", result.data.get("output_path"))

            else:
                return {"success": False, "error": f"Unknown step: {step.name}"}

            return {"success": True, "data": step.output_data}

        except Exception as e:
            step.fail(str(e))
            return {"success": False, "error": str(e)}

    def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow state by ID."""
        return self._store.get(workflow_id)

    def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        state = self._store.get(workflow_id)
        if state and state.status == WorkflowStatus.RUNNING:
            state.cancel()
            self._store.save(state)
            return True
        return False

    def pause(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        state = self._store.get(workflow_id)
        if state and state.status == WorkflowStatus.RUNNING:
            state.pause()
            self._store.save(state)
            return True
        return False

    def resume(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        state = self._store.get(workflow_id)
        if state and state.status == WorkflowStatus.PAUSED:
            state.resume()
            self._store.save(state)
            return True
        return False

    def build_langgraph(self) -> StateGraph:
        """
        Build a LangGraph representation of the workflow.

        This enables integration with LangGraph's execution engine.
        """
        from typing import TypedDict

        class GraphState(TypedDict):
            nprd_content: str
            parsed_nprd: Dict[str, Any]
            nprd_validation: Dict[str, Any]
            trd: Dict[str, Any]
            mql5_code: str
            syntax_validation: Dict[str, Any]
            compilation_result: Dict[str, Any]
            ea_path: str
            error: Optional[str]

        # Create graph
        workflow = StateGraph(GraphState)

        # Add nodes
        async def parse_node(state: GraphState) -> GraphState:
            result = self._parse_nprd.execute(nprd_content=state["nprd_content"])
            return {**state, "parsed_nprd": result.data}

        async def validate_node(state: GraphState) -> GraphState:
            result = self._validate_nprd.execute(nprd_content=state["nprd_content"])
            return {**state, "nprd_validation": result.data}

        async def trd_node(state: GraphState) -> GraphState:
            result = self._generate_trd.execute(nprd_content=state["nprd_content"])
            return {**state, "trd": result.data}

        async def mql5_node(state: GraphState) -> GraphState:
            trd_content = state["trd"].get("content", "")
            result = self._generate_mql5.execute(trd_content=trd_content)
            return {**state, "mql5_code": result.data.get("code", "")}

        async def compile_node(state: GraphState) -> GraphState:
            result = self._compile_ea.execute(code=state["mql5_code"])
            return {
                **state,
                "compilation_result": result.data,
                "ea_path": result.data.get("output_path", ""),
            }

        workflow.add_node("parse_nprd", parse_node)
        workflow.add_node("validate_nprd", validate_node)
        workflow.add_node("generate_trd", trd_node)
        workflow.add_node("generate_mql5", mql5_node)
        workflow.add_node("compile_ea", compile_node)

        # Add edges
        workflow.set_entry_point("parse_nprd")
        workflow.add_edge("parse_nprd", "validate_nprd")
        workflow.add_edge("validate_nprd", "generate_trd")
        workflow.add_edge("generate_trd", "generate_mql5")
        workflow.add_edge("generate_mql5", "compile_ea")
        workflow.add_edge("compile_ea", END)

        return workflow.compile()


def create_nprd_to_ea_workflow(
    workspace_path: Optional[str] = None,
) -> NPRDToEATWorkflow:
    """Create an NPRD to EA workflow."""
    return NPRDToEATWorkflow(workspace_path=workspace_path)


async def run_nprd_to_ea_workflow(
    nprd_content: str,
    workspace_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowResult:
    """
    Run the NPRD to EA workflow.

    Convenience function for one-shot execution.
    """
    workflow = create_nprd_to_ea_workflow(workspace_path)
    return await workflow.run(nprd_content, metadata)
