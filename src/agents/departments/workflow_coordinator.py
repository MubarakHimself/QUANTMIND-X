"""
Department Workflow Coordinator — All 4 Canonical Workflows

Orchestrates the flow between Trading Floor departments using the department
mail system, Prefect durable execution, graph memory, and Kanban board updates.

Supports all 4 workflows from architecture.md §20 / PRD S3-9:

- WF1: AlphaForge Creation Pipeline (manual trigger, source → EA in library)
- WF2: AlphaForge Enhancement Loop (continuous, EA library → improved EA → live)
- WF3: Performance Intelligence (daily Dead Zone 16:15–18:00 GMT)
- WF4: Weekend Update Cycle (weekly Friday→Monday)

Integration layers (architecture §5.1):
- Prefect: Scheduling and durability (flows/ directory)
- Department Mail: Inter-department coordination (SQLite/Redis)
- Graph Memory: Workflow state persistence and decision memory
- Kanban Board: Visual pipeline tracking for strategy status
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

from src.agents.departments.department_mail import (
    DepartmentMailService,
    DepartmentMessage,
    MessageType,
    Priority,
)
from src.agents.departments.types import Department
from src.agents.departments.workflow_models import (
    DepartmentWorkflow,
    EALifecycleState,
    TaskPriority,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTask,
    WorkflowType,
    WORKFLOW_INITIAL_STAGES,
    WORKFLOW_NEXT_STAGES,
    WORKFLOW_STAGE_DEPARTMENTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kanban status mapping: WorkflowStage → Kanban StrategyStatus
# ---------------------------------------------------------------------------
STAGE_TO_KANBAN_STATUS: Dict[WorkflowStage, str] = {
    WorkflowStage.VIDEO_INGEST: "PENDING",
    WorkflowStage.RESEARCH: "PROCESSING",
    WorkflowStage.RESEARCH_IMPROVEMENT: "PROCESSING",
    WorkflowStage.DEVELOPMENT: "PROCESSING",
    WorkflowStage.BACKTESTING: "PROCESSING",
    WorkflowStage.EA_LIBRARY_SCAN: "PROCESSING",
    WorkflowStage.FULL_BACKTEST_MATRIX: "PROCESSING",
    WorkflowStage.VARIANT_GENERATION: "PROCESSING",
    WorkflowStage.DATA_VARIANT_STRESS: "PROCESSING",
    WorkflowStage.SIT_GATE: "READY",
    WorkflowStage.PAPER_TRADING: "READY",
    WorkflowStage.PAPER_TRADING_MONITOR: "READY",
    WorkflowStage.HUMAN_APPROVAL_GATE: "READY",
    WorkflowStage.COMPLETED: "PRIMAL",
    # WF3 / WF4 stages
    WorkflowStage.EOD_REPORT: "PROCESSING",
    WorkflowStage.SESSION_PERFORMER_ID: "PROCESSING",
    WorkflowStage.DPR_UPDATE: "PROCESSING",
    WorkflowStage.QUEUE_RERANK: "PROCESSING",
    WorkflowStage.FORTNIGHT_ACCUMULATION: "PROCESSING",
    WorkflowStage.FRIDAY_ANALYSIS: "PROCESSING",
    WorkflowStage.SATURDAY_REFINEMENT: "PROCESSING",
    WorkflowStage.SUNDAY_PREMARKET: "PROCESSING",
    WorkflowStage.MONDAY_ROSTER: "READY",
}

# Prefect flow mapping: WorkflowType → Prefect flow module + function
WORKFLOW_PREFECT_FLOWS: Dict[WorkflowType, Dict[str, str]] = {
    WorkflowType.WF1_CREATION: {
        "module": "flows.video_ingest_flow",
        "function": "video_ingest_to_ea_flow",
    },
    WorkflowType.WF2_ENHANCEMENT: {
        "module": "flows.alpha_forge_flow",
        "function": "alpha_forge_enhancement_flow",
    },
    WorkflowType.WF3_PERFORMANCE_INTEL: {
        "module": "flows.research_synthesis_flow",
        "function": "performance_intelligence_flow",
    },
    WorkflowType.WF4_WEEKEND_UPDATE: {
        "module": "flows.weekend_compute_flow",
        "function": "weekend_update_flow",
    },
}


class DepartmentWorkflowCoordinator:
    """
    Orchestrates all 4 canonical workflows between Trading Floor departments.

    Three-layer integration (architecture §5.1):
    1. Prefect — scheduling & durability (flows/*.py)
    2. Agent SDK — agentic steps via BaseAgent.query()
    3. Department Mail — inter-department task dispatch

    Also integrates:
    - Graph Memory — persist workflow decisions/state for agent recall
    - Kanban Board — update strategy pipeline status for TUI display
    - Workflow Database — Prefect SQLite persistence (flows/database.py)

    Usage:
        coordinator = DepartmentWorkflowCoordinator()

        # WF1: Start creation pipeline
        wf_id = coordinator.start_workflow(
            workflow_type=WorkflowType.WF1_CREATION,
            initial_payload={"source_url": "https://youtube.com/..."}
        )

        # WF3: Start daily performance intelligence
        wf_id = coordinator.start_workflow(
            workflow_type=WorkflowType.WF3_PERFORMANCE_INTEL,
        )

        # Process next stage
        coordinator.advance_workflow(wf_id)
    """

    def __init__(
        self,
        mail_db_path: str = os.environ.get(
            "DEPARTMENT_MAIL_DB", ".quantmind/department_mail.db"
        ),
        on_progress: Optional[Callable[[str, float, WorkflowStage], None]] = None,
        use_prefect: bool = True,
        use_memory: bool = True,
    ):
        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self.on_progress = on_progress
        self._workflows: Dict[str, DepartmentWorkflow] = {}
        self._use_prefect = use_prefect
        self._use_memory = use_memory

        # Lazy-init integrations
        self._workflow_db = None
        self._memory_facade = None

        logger.info("DepartmentWorkflowCoordinator initialized (WF1-WF4)")

    # -----------------------------------------------------------------------
    # Integration Helpers — Prefect, Memory, Kanban
    # -----------------------------------------------------------------------

    def _get_workflow_db(self):
        """Lazy-init Prefect workflow database (flows/database.py)."""
        if self._workflow_db is None:
            try:
                from flows.database import get_workflow_database
                self._workflow_db = get_workflow_database()
            except Exception as e:
                logger.warning(f"Prefect workflow DB unavailable: {e}")
        return self._workflow_db

    def _get_memory_facade(self):
        """Lazy-init graph memory facade (architecture §6.1)."""
        if self._memory_facade is None:
            try:
                from src.memory.graph.facade import GraphMemoryFacade
                self._memory_facade = GraphMemoryFacade()
            except Exception as e:
                logger.warning(f"Graph memory facade unavailable: {e}")
        return self._memory_facade

    def _persist_to_prefect_db(
        self, workflow_id: str, workflow_type: str, status: str,
        input_data: Optional[str] = None,
    ) -> None:
        """Persist workflow state to Prefect SQLite database for durability."""
        db = self._get_workflow_db()
        if db:
            try:
                db.save_workflow_run(
                    workflow_id=workflow_id,
                    workflow_name=workflow_type,
                    status=status,
                    input_data=input_data,
                )
            except Exception as e:
                logger.warning(f"Prefect DB persist failed: {e}")

    def _update_prefect_db_status(
        self, workflow_id: str, status: str,
        output_data: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update workflow status in Prefect database."""
        db = self._get_workflow_db()
        if db:
            try:
                db.update_workflow_status(
                    workflow_id=workflow_id,
                    status=status,
                    output_data=output_data,
                    error_message=error_message,
                )
            except Exception as e:
                logger.warning(f"Prefect DB update failed: {e}")

    def _persist_stage_to_prefect(
        self, workflow_id: str, stage_name: str, status: str,
        result_data: Optional[str] = None,
    ) -> None:
        """Persist individual stage result to Prefect database."""
        db = self._get_workflow_db()
        if db:
            try:
                db.save_stage_result(
                    workflow_run_id=workflow_id,
                    stage_name=stage_name,
                    status=status,
                    result_data=result_data,
                )
            except Exception as e:
                logger.warning(f"Prefect stage persist failed: {e}")

    def _write_memory(
        self, workflow_id: str, content: str,
        node_type: str = "DECISION", category: str = "workflow",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Write workflow event to graph memory for agent recall."""
        if not self._use_memory:
            return
        facade = self._get_memory_facade()
        if facade:
            try:
                facade.retain(
                    node_type=node_type,
                    content=content,
                    category=category,
                    tags=tags or ["workflow", workflow_id],
                    metadata={
                        "workflow_id": workflow_id,
                        "created_at": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.debug(f"Memory write skipped: {e}")

    def _update_kanban_status(
        self, strategy_id: Optional[str], stage: WorkflowStage,
        workflow_id: str,
    ) -> None:
        """
        Update Kanban board strategy status based on workflow stage.

        Maps workflow stages to Kanban columns (Inbox → Processing →
        Extracting → Done) for both:
        1. Strategy .meta.json — so the polling `/api/strategies` picks it up
        2. SSE thought stream — so the UI can update in real time
        """
        kanban_status_upper = STAGE_TO_KANBAN_STATUS.get(stage, "PROCESSING")
        # Map to lowercase StrategyStatus values (pending|processing|ready|primal)
        kanban_status = kanban_status_upper.lower()

        # 1. Persist to strategy .meta.json for polling-based Kanban
        if strategy_id:
            try:
                from src.api.ide_models import STRATEGIES_DIR
                meta_path = STRATEGIES_DIR / strategy_id / ".meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["status"] = kanban_status
                    meta["last_stage"] = stage.value
                    meta["workflow_id"] = workflow_id
                    meta["updated_at"] = datetime.now().isoformat()
                    meta_path.write_text(json.dumps(meta, indent=2))
                    logger.debug(f"Kanban .meta.json updated: {strategy_id} → {kanban_status}")
            except Exception as e:
                logger.debug(f"Kanban .meta.json update skipped: {e}")

        # 2. Emit SSE event for real-time Kanban update in the UI
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            get_thought_publisher().publish(
                department="floor_manager",
                thought=json.dumps({
                    "type": "kanban_update",
                    "strategy_id": strategy_id or workflow_id,
                    "status": kanban_status,
                    "stage": stage.value,
                    "workflow_id": workflow_id,
                }),
                thought_type="action",
            )
        except Exception as e:
            logger.debug(f"Kanban SSE update skipped: {e}")

    async def _trigger_prefect_flow(
        self, workflow_type: WorkflowType,
        workflow_id: str, payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        Trigger the corresponding Prefect @flow for durable execution.

        Returns the Prefect flow_run_id if successful, None otherwise.
        The Prefect flow handles retries, checkpointing, and artifact creation.
        """
        if not self._use_prefect:
            return None

        flow_info = WORKFLOW_PREFECT_FLOWS.get(workflow_type)
        if not flow_info:
            logger.debug(f"No Prefect flow mapped for {workflow_type.value}")
            return None

        try:
            import importlib
            mod = importlib.import_module(flow_info["module"])
            flow_func = getattr(mod, flow_info["function"], None)
            if flow_func and callable(flow_func):
                # Schedule Prefect flow run (non-blocking)
                import asyncio
                loop = asyncio.get_event_loop()
                # Prefect flows are sync — run in executor
                flow_run_id = await loop.run_in_executor(
                    None, lambda: flow_func(
                        workflow_id=workflow_id,
                        **payload,
                    )
                )
                logger.info(
                    f"Prefect flow triggered: {flow_info['function']} "
                    f"→ run_id={flow_run_id}"
                )
                return str(flow_run_id) if flow_run_id else None
        except Exception as e:
            logger.warning(f"Prefect flow trigger failed for {workflow_type.value}: {e}")
        return None

    # -----------------------------------------------------------------------
    # Workflow Lifecycle
    # -----------------------------------------------------------------------

    def start_workflow(
        self,
        workflow_type: WorkflowType,
        initial_payload: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new workflow of the specified type.

        Args:
            workflow_type: Which of the 4 canonical workflows to start
            initial_payload: Initial data for the workflow
            strategy_id: Strategy ID (for WF2 enhancement loop)
            priority: Task priority level
            metadata: Additional metadata

        Returns:
            Workflow ID
        """
        workflow_id = f"wf_{workflow_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        initial_stage = WORKFLOW_INITIAL_STAGES.get(
            workflow_type, WorkflowStage.COMPLETED
        )

        # Determine initial EA lifecycle state
        ea_state = None
        if workflow_type == WorkflowType.WF1_CREATION:
            ea_state = EALifecycleState.DRAFT
        elif workflow_type == WorkflowType.WF2_ENHANCEMENT:
            ea_state = EALifecycleState.TESTING

        workflow = DepartmentWorkflow(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            status=WorkflowStatus.PENDING,
            current_stage=initial_stage,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                **(metadata or {}),
                "initial_payload": initial_payload or {},
                "priority": priority.value,
            },
            strategy_id=strategy_id,
            ea_lifecycle_state=ea_state,
        )

        self._workflows[workflow_id] = workflow

        # Create initial task via department mail
        dept_map = WORKFLOW_STAGE_DEPARTMENTS.get(workflow_type, {})
        to_dept = dept_map.get(initial_stage, "floor_manager")

        self._create_task(
            workflow=workflow,
            stage=initial_stage,
            from_dept="floor_manager",
            to_dept=to_dept,
            payload=initial_payload or {},
            priority=priority,
        )

        # --- Integration: Prefect DB persistence ---
        self._persist_to_prefect_db(
            workflow_id=workflow_id,
            workflow_type=workflow_type.value,
            status="pending",
            input_data=json.dumps(initial_payload or {}),
        )

        # --- Integration: Graph memory ---
        self._write_memory(
            workflow_id=workflow_id,
            content=(
                f"Workflow {workflow_type.value} started: {workflow_id}. "
                f"Strategy: {strategy_id or 'N/A'}. "
                f"Initial stage: {initial_stage.value} → dept:{to_dept}"
            ),
            node_type="ACTION",
            tags=["workflow", "start", workflow_type.value, workflow_id],
        )

        # --- Integration: Kanban board ---
        self._update_kanban_status(strategy_id, initial_stage, workflow_id)

        logger.info(
            f"Started {workflow_type.value} workflow {workflow_id} "
            f"→ {initial_stage.value} → dept:{to_dept}"
        )
        return workflow_id

    def advance_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Advance the workflow to its next stage.

        Reads the routing table for the workflow type and dispatches
        to the next department.
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        if workflow.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            return {"error": f"Workflow {workflow_id} is {workflow.status.value}"}

        current_stage = workflow.current_stage
        next_stage_map = WORKFLOW_NEXT_STAGES.get(workflow.workflow_type, {})
        next_stage = next_stage_map.get(current_stage)

        if not next_stage or next_stage == WorkflowStage.COMPLETED:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.current_stage = WorkflowStage.COMPLETED
            workflow.updated_at = datetime.now()

            # Update EA lifecycle on completion
            if workflow.workflow_type == WorkflowType.WF1_CREATION:
                workflow.ea_lifecycle_state = EALifecycleState.DRAFT
            elif workflow.workflow_type == WorkflowType.WF2_ENHANCEMENT:
                workflow.ea_lifecycle_state = EALifecycleState.ACTIVE

            # --- Integration: Prefect DB ---
            self._update_prefect_db_status(workflow_id, "completed")

            # --- Integration: Graph memory ---
            self._write_memory(
                workflow_id=workflow_id,
                content=f"Workflow {workflow_id} completed. EA state: {workflow.ea_lifecycle_state}",
                node_type="RESULT",
                tags=["workflow", "completed", workflow_id],
            )

            # --- Integration: Kanban → PRIMAL ---
            self._update_kanban_status(
                workflow.strategy_id, WorkflowStage.COMPLETED, workflow_id,
            )

            logger.info(f"Workflow {workflow_id} completed")
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "current_stage": WorkflowStage.COMPLETED.value,
            }

        # ── HITL Gate Check ─────────────────────────────────────────────
        # If the next stage is a human approval gate, pause the workflow
        # and create an approval request. The agent/endpoint must call
        # `resolve_gate()` after human approval before advancing further.
        _HITL_GATES = {WorkflowStage.SIT_GATE, WorkflowStage.HUMAN_APPROVAL_GATE}
        if next_stage in _HITL_GATES:
            workflow.current_stage = next_stage
            workflow.status = WorkflowStatus.WAITING
            workflow.updated_at = datetime.now()

            # Create approval request via ApprovalManager
            try:
                from src.agents.approval_manager import (
                    get_approval_manager, ApprovalType, ApprovalUrgency,
                )
                gate_label = (
                    "SIT Gate" if next_stage == WorkflowStage.SIT_GATE
                    else "Human Approval Gate"
                )
                urgency = (
                    ApprovalUrgency.HIGH
                    if next_stage == WorkflowStage.HUMAN_APPROVAL_GATE
                    else ApprovalUrgency.MEDIUM
                )
                req = get_approval_manager().request_approval(
                    approval_type=ApprovalType.WORKFLOW_GATE,
                    title=f"{gate_label}: {workflow.strategy_id or workflow_id}",
                    description=(
                        f"Workflow {workflow_id} has reached {gate_label}. "
                        f"Strategy: {workflow.strategy_id or 'N/A'}. "
                        f"Previous stage: {current_stage.value}. "
                        f"Approve to continue or reject to archive."
                    ),
                    department="floor_manager",
                    agent_id="workflow_coordinator",
                    workflow_id=workflow_id,
                    workflow_stage=next_stage.value,
                    strategy_id=workflow.strategy_id,
                    urgency=urgency,
                    context={
                        "workflow_type": workflow.workflow_type.value,
                        "ea_lifecycle_state": (
                            workflow.ea_lifecycle_state.value
                            if workflow.ea_lifecycle_state else None
                        ),
                        "previous_stage": current_stage.value,
                        "task_count": len(workflow.tasks),
                    },
                    # Resume context — snapshot of workflow state for recovery
                    resume_context=self._snapshot_workflow(workflow),
                )
                # Store approval ID on workflow metadata for later resolution
                workflow.metadata["pending_approval_id"] = req.id
            except Exception as e:
                logger.warning(f"HITL gate approval request failed: {e}")

            self._update_kanban_status(workflow.strategy_id, next_stage, workflow_id)

            logger.info(
                f"Workflow {workflow_id} paused at HITL gate: {next_stage.value}"
            )
            return {
                "workflow_id": workflow_id,
                "status": "waiting",
                "current_stage": next_stage.value,
                "gate": next_stage.value,
                "requires_approval": True,
                "approval_id": workflow.metadata.get("pending_approval_id"),
            }

        # Move to next stage (non-gate stages)
        workflow.current_stage = next_stage
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now()

        # Resolve target department
        dept_map = WORKFLOW_STAGE_DEPARTMENTS.get(workflow.workflow_type, {})
        to_dept = dept_map.get(next_stage, "floor_manager")

        # Get previous results
        previous_results = {}
        if workflow.tasks:
            last = workflow.tasks[-1]
            if last.result:
                previous_results = last.result

        from_dept = workflow.tasks[-1].to_dept if workflow.tasks else "floor_manager"

        task = self._create_task(
            workflow=workflow,
            stage=next_stage,
            from_dept=from_dept,
            to_dept=to_dept,
            payload={
                "workflow_id": workflow_id,
                "stage": next_stage.value,
                "strategy_id": workflow.strategy_id,
                "from_previous_stage": previous_results,
            },
            priority=TaskPriority(workflow.metadata.get("priority", "medium")),
        )

        if self.on_progress:
            self.on_progress(workflow_id, workflow.get_progress(), next_stage)

        # --- Integration: Prefect DB stage tracking ---
        self._persist_stage_to_prefect(
            workflow_id=workflow_id,
            stage_name=next_stage.value,
            status="running",
        )
        self._update_prefect_db_status(workflow_id, "running")

        # --- Integration: Graph memory ---
        self._write_memory(
            workflow_id=workflow_id,
            content=(
                f"Workflow {workflow_id} advanced: "
                f"{current_stage.value} → {next_stage.value} → dept:{to_dept}"
            ),
            node_type="ACTION",
            tags=["workflow", "advance", workflow_id, next_stage.value],
        )

        # --- Integration: Kanban board ---
        self._update_kanban_status(workflow.strategy_id, next_stage, workflow_id)

        logger.info(
            f"Workflow {workflow_id} advanced: {current_stage.value} → {next_stage.value} → dept:{to_dept}"
        )

        return {
            "workflow_id": workflow_id,
            "task_id": task.task_id,
            "previous_stage": current_stage.value,
            "current_stage": next_stage.value,
            "to_dept": to_dept,
            "status": workflow.status.value,
            "progress": workflow.get_progress(),
        }

    def handle_department_response(
        self,
        workflow_id: str,
        from_dept: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a response from a department completing its task."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        pending_task = None
        for task in reversed(workflow.tasks):
            if task.to_dept == from_dept and task.status == WorkflowStatus.RUNNING:
                pending_task = task
                break

        if not pending_task:
            return {"error": f"No pending task for department {from_dept}"}

        pending_task.status = WorkflowStatus.COMPLETED
        pending_task.completed_at = datetime.now()
        pending_task.result = result

        workflow.metadata["latest_results"] = result
        workflow.updated_at = datetime.now()

        # Update EA lifecycle state based on stage completion
        self._update_ea_lifecycle(workflow, pending_task.stage, result)

        # --- Integration: Prefect DB stage completion ---
        self._persist_stage_to_prefect(
            workflow_id=workflow_id,
            stage_name=pending_task.stage.value,
            status="completed",
            result_data=json.dumps(result)[:4000],
        )

        # --- Integration: Graph memory — record department result ---
        self._write_memory(
            workflow_id=workflow_id,
            content=(
                f"Department {from_dept} completed stage {pending_task.stage.value} "
                f"for workflow {workflow_id}. "
                f"EA state: {workflow.ea_lifecycle_state}"
            ),
            node_type="RESULT",
            tags=["workflow", "stage_complete", from_dept, workflow_id],
        )

        return {
            "workflow_id": workflow_id,
            "task_id": pending_task.task_id,
            "status": "completed",
            "ea_lifecycle_state": (
                workflow.ea_lifecycle_state.value
                if workflow.ea_lifecycle_state
                else None
            ),
        }

    def _update_ea_lifecycle(
        self,
        workflow: DepartmentWorkflow,
        completed_stage: WorkflowStage,
        result: Dict[str, Any],
    ) -> None:
        """Update EA lifecycle state based on completed stage."""
        transitions = {
            WorkflowStage.SIT_GATE: EALifecycleState.SIT_PASSED,
            WorkflowStage.PAPER_TRADING_MONITOR: EALifecycleState.PAPER_VALIDATED,
            WorkflowStage.HUMAN_APPROVAL_GATE: (
                EALifecycleState.ACTIVE
                if result.get("approved")
                else EALifecycleState.ARCHIVED
            ),
        }

        new_state = transitions.get(completed_stage)
        if new_state:
            workflow.ea_lifecycle_state = new_state
            logger.info(
                f"EA lifecycle: {workflow.strategy_id} → {new_state.value}"
            )

    # -----------------------------------------------------------------------
    # HITL Gate Resolution
    # -----------------------------------------------------------------------

    def resolve_gate(
        self,
        workflow_id: str,
        approved: bool,
        resolved_by: str = "human",
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Resolve a HITL gate (SIT_GATE or HUMAN_APPROVAL_GATE).

        If approved, advances the workflow past the gate.
        If rejected, marks the workflow as failed/archived.

        Args:
            workflow_id: The workflow to resolve
            approved: True to approve, False to reject
            resolved_by: Who approved/rejected
            reason: Optional rejection reason

        Returns:
            Status dict with next state
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        if workflow.status != WorkflowStatus.WAITING:
            return {"error": f"Workflow {workflow_id} is not waiting for approval"}

        gate_stage = workflow.current_stage
        _HITL_GATES = {WorkflowStage.SIT_GATE, WorkflowStage.HUMAN_APPROVAL_GATE}
        if gate_stage not in _HITL_GATES:
            return {"error": f"Workflow {workflow_id} is not at a gate stage"}

        # Resolve the approval request in ApprovalManager
        approval_id = workflow.metadata.get("pending_approval_id")
        if approval_id:
            try:
                from src.agents.approval_manager import get_approval_manager
                mgr = get_approval_manager()
                if approved:
                    mgr.approve(approval_id, approved_by=resolved_by)
                else:
                    mgr.reject(approval_id, reason=reason, rejected_by=resolved_by)
            except Exception as e:
                logger.debug(f"Approval resolution failed: {e}")
            workflow.metadata.pop("pending_approval_id", None)

        if approved:
            # Record gate result
            self.handle_department_response(
                workflow_id, "floor_manager",
                {"approved": True, "resolved_by": resolved_by},
            )
            # Update EA lifecycle
            self._update_ea_lifecycle(
                workflow, gate_stage, {"approved": True},
            )
            # Advance past the gate
            result = self.advance_workflow(workflow_id)
            self._write_memory(
                workflow_id=workflow_id,
                content=(
                    f"HITL gate {gate_stage.value} APPROVED by {resolved_by}. "
                    f"Workflow advancing."
                ),
                node_type="DECISION",
                tags=["hitl", "approved", gate_stage.value, workflow_id],
            )
            return result
        else:
            # Rejection — archive the workflow
            workflow.status = WorkflowStatus.FAILED
            workflow.updated_at = datetime.now()
            self._update_ea_lifecycle(
                workflow, gate_stage, {"approved": False},
            )
            self._update_prefect_db_status(workflow_id, "failed")
            self._update_kanban_status(
                workflow.strategy_id, WorkflowStage.FAILED, workflow_id,
            )
            self._write_memory(
                workflow_id=workflow_id,
                content=(
                    f"HITL gate {gate_stage.value} REJECTED by {resolved_by}. "
                    f"Reason: {reason or 'No reason given'}. "
                    f"Workflow archived."
                ),
                node_type="DECISION",
                tags=["hitl", "rejected", gate_stage.value, workflow_id],
            )
            logger.info(
                f"Workflow {workflow_id} rejected at {gate_stage.value}: {reason}"
            )
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "gate": gate_stage.value,
                "rejection_reason": reason,
            }

    # -----------------------------------------------------------------------
    # WF2 Enhancement Loop — continuous iteration support
    # -----------------------------------------------------------------------

    def should_loop_wf2(self, workflow_id: str) -> bool:
        """
        Check if WF2 should loop back for another improvement round.

        WF2 runs continuously — when a round completes, check if there
        are more strategies to improve.
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow or workflow.workflow_type != WorkflowType.WF2_ENHANCEMENT:
            return False

        # Loop if: SIT failed, or improvement delta found, or more strategies exist
        latest = workflow.metadata.get("latest_results", {})
        if latest.get("sit_failed"):
            return True
        if latest.get("improvement_found"):
            return True

        return False

    def restart_wf2_round(self, workflow_id: str) -> Dict[str, Any]:
        """Restart WF2 from RESEARCH_IMPROVEMENT stage for another round."""
        workflow = self._workflows.get(workflow_id)
        if not workflow or workflow.workflow_type != WorkflowType.WF2_ENHANCEMENT:
            return {"error": "Not a WF2 workflow"}

        cycle = workflow.metadata.get("improvement_cycle", 0) + 1
        workflow.metadata["improvement_cycle"] = cycle
        workflow.current_stage = WorkflowStage.RESEARCH_IMPROVEMENT
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now()

        logger.info(f"WF2 {workflow_id} restarting round {cycle}")
        return self.advance_workflow(workflow_id)

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _create_task(
        self,
        workflow: DepartmentWorkflow,
        stage: WorkflowStage,
        from_dept: str,
        to_dept: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> WorkflowTask:
        """Create a new task and dispatch via department mail."""
        task_id = f"task_{len(workflow.tasks) + 1}"

        msg_type = (
            MessageType.STRATEGY_DISPATCH
            if stage in (WorkflowStage.RESEARCH, WorkflowStage.RESEARCH_IMPROVEMENT)
            else MessageType.DISPATCH
        )

        pri = (
            Priority.HIGH
            if priority == TaskPriority.HIGH
            else Priority.LOW if priority == TaskPriority.LOW
            else Priority.NORMAL
        )

        message = self.mail_service.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=msg_type,
            subject=f"[{workflow.workflow_type.value}] {workflow.workflow_id}: {stage.value}",
            body=json.dumps({
                "workflow_id": workflow.workflow_id,
                "workflow_type": workflow.workflow_type.value,
                "stage": stage.value,
                "payload": payload,
                "strategy_id": workflow.strategy_id,
            }),
            priority=pri,
        )

        task = WorkflowTask(
            task_id=task_id,
            stage=stage,
            from_dept=from_dept,
            to_dept=to_dept,
            message_id=message.id,
            status=WorkflowStatus.RUNNING,
            created_at=datetime.now(),
            priority=priority,
            payload=payload,
        )
        workflow.tasks.append(task)

        logger.info(f"Task {task_id}: {stage.value} → {to_dept} (pri={priority.value})")
        return task

    # -----------------------------------------------------------------------
    # Query Methods
    # -----------------------------------------------------------------------

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        workflow = self._workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None

    def get_all_workflows(self) -> List[Dict[str, Any]]:
        return [w.to_dict() for w in self._workflows.values()]

    def get_workflows_by_type(self, workflow_type: WorkflowType) -> List[Dict[str, Any]]:
        return [
            w.to_dict()
            for w in self._workflows.values()
            if w.workflow_type == workflow_type
        ]

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        return [
            w.to_dict()
            for w in self._workflows.values()
            if w.status in (WorkflowStatus.RUNNING, WorkflowStatus.WAITING, WorkflowStatus.PENDING)
        ]

    def cancel_workflow(self, workflow_id: str) -> bool:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return False
        if workflow.status in (WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED):
            return False

        workflow.status = WorkflowStatus.CANCELLED
        workflow.updated_at = datetime.now()

        for task in workflow.tasks:
            if task.status == WorkflowStatus.RUNNING:
                task.status = WorkflowStatus.CANCELLED
                task.completed_at = datetime.now()

        logger.info(f"Workflow {workflow_id} cancelled")
        return True

    def check_department_inbox(
        self, workflow_id: str, department: str,
    ) -> List[Dict[str, Any]]:
        """Check inbox for department responses related to a workflow."""
        messages = self.mail_service.check_inbox(
            dept=department, unread_only=True, limit=50,
        )
        wf_messages = []
        for msg in messages:
            try:
                body = json.loads(msg.body)
                if body.get("workflow_id") == workflow_id:
                    wf_messages.append(msg.to_dict())
            except (json.JSONDecodeError, AttributeError):
                continue
        return wf_messages

    # -----------------------------------------------------------------------
    # Resume Logic — Recover paused workflows after server restart
    # -----------------------------------------------------------------------

    def _snapshot_workflow(self, workflow: DepartmentWorkflow) -> Dict[str, Any]:
        """
        Create a resume context snapshot of a workflow.

        This is stored alongside the approval request so that on resume
        the coordinator can re-create the in-memory workflow state.
        """
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type.value,
            "status": workflow.status.value,
            "current_stage": workflow.current_stage.value,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "metadata": workflow.metadata,
            "strategy_id": workflow.strategy_id,
            "ea_lifecycle_state": (
                workflow.ea_lifecycle_state.value
                if workflow.ea_lifecycle_state else None
            ),
            "task_count": len(workflow.tasks),
            "last_task_stage": (
                workflow.tasks[-1].stage.value if workflow.tasks else None
            ),
        }

    def resume_waiting_workflows(self) -> int:
        """
        Resume workflows that were paused at HITL gates before a restart.

        1. Asks ApprovalManager to reload pending approvals from DB
        2. For each pending workflow-gate approval with resume_context:
           - Reconstructs the in-memory DepartmentWorkflow if absent
           - Re-registers the workflow in self._workflows
           - Re-sends department mail reminders

        Returns the count of workflows resumed.

        Call this during server startup after all services are initialized.
        """
        try:
            from src.agents.approval_manager import (
                get_approval_manager, ApprovalType,
            )
        except ImportError:
            logger.warning("ApprovalManager not available for workflow resume")
            return 0

        mgr = get_approval_manager()
        # Step 1: Reload pending approvals from DB
        resumed_approvals = mgr.resume_pending()

        # Step 2: Re-attach workflow state for gate approvals
        resumed_workflows = 0
        pending = mgr.get_pending()

        for item in pending:
            if item.get("approval_type") != ApprovalType.WORKFLOW_GATE.value:
                continue

            wf_id = item.get("workflow_id")
            if not wf_id or wf_id in self._workflows:
                continue  # Already in memory

            ctx = item.get("resume_context") or {}
            if not ctx:
                # Try loading from approval manager
                ctx = mgr.get_resume_context(item["id"]) or {}

            if not ctx.get("workflow_type"):
                logger.debug(f"No resume_context for workflow {wf_id}, skipping")
                continue

            # Reconstruct the in-memory DepartmentWorkflow
            try:
                wf_type = WorkflowType(ctx["workflow_type"])
                wf_stage = WorkflowStage(ctx.get("current_stage", "completed"))
                ea_state = (
                    EALifecycleState(ctx["ea_lifecycle_state"])
                    if ctx.get("ea_lifecycle_state") else None
                )

                workflow = DepartmentWorkflow(
                    workflow_id=wf_id,
                    workflow_type=wf_type,
                    status=WorkflowStatus.WAITING,
                    current_stage=wf_stage,
                    created_at=datetime.fromisoformat(ctx.get("created_at", datetime.now().isoformat())),
                    updated_at=datetime.now(),
                    metadata={
                        **(ctx.get("metadata", {})),
                        "pending_approval_id": item["id"],
                        "resumed_from_db": True,
                    },
                    strategy_id=ctx.get("strategy_id"),
                    ea_lifecycle_state=ea_state,
                )

                self._workflows[wf_id] = workflow

                # Update Kanban to reflect waiting state
                self._update_kanban_status(
                    workflow.strategy_id, wf_stage, wf_id,
                )

                # Write memory about the resume event
                self._write_memory(
                    workflow_id=wf_id,
                    content=(
                        f"Workflow {wf_id} resumed from DB after server restart. "
                        f"Waiting at HITL gate: {wf_stage.value}. "
                        f"Approval ID: {item['id']}."
                    ),
                    node_type="ACTION",
                    tags=["workflow", "resumed", "hitl", wf_id],
                )

                # Send department mail reminder
                self.mail_service.send(
                    from_dept="floor_manager",
                    to_dept=ctx.get("metadata", {}).get("to_dept", "development"),
                    type=MessageType.APPROVAL_REQUEST,
                    subject=f"[Resumed] Workflow {wf_id} waiting at {wf_stage.value}",
                    body=(
                        f"This workflow was paused before server restart and has been recovered.\n"
                        f"Workflow type: {wf_type.value}\n"
                        f"Strategy: {ctx.get('strategy_id', 'N/A')}\n"
                        f"Approval ID: {item['id']}\n\n"
                        f"Please review and approve/reject to continue."
                    ),
                    priority=Priority.HIGH,
                    gate_id=item["id"],
                    workflow_id=wf_id,
                )

                resumed_workflows += 1
                logger.info(
                    f"Resumed workflow {wf_id} at gate {wf_stage.value} "
                    f"(approval={item['id']})"
                )

            except Exception as e:
                logger.warning(f"Failed to resume workflow {wf_id}: {e}")

        if resumed_workflows > 0:
            logger.info(
                f"Workflow resume complete: {resumed_workflows} workflows, "
                f"{resumed_approvals} approvals restored from DB"
            )

        return resumed_workflows

    def close(self):
        self.mail_service.close()
        logger.info("DepartmentWorkflowCoordinator closed")


# ===========================================================================
# Singleton
# ===========================================================================

_coordinator: Optional[DepartmentWorkflowCoordinator] = None


def get_workflow_coordinator() -> DepartmentWorkflowCoordinator:
    """Get or create the global workflow coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = DepartmentWorkflowCoordinator()
    return _coordinator


def create_workflow_coordinator(
    mail_db_path: str = os.environ.get(
        "DEPARTMENT_MAIL_DB", ".quantmind/department_mail.db"
    ),
    on_progress: Optional[Callable[[str, float, WorkflowStage], None]] = None,
) -> DepartmentWorkflowCoordinator:
    return DepartmentWorkflowCoordinator(
        mail_db_path=mail_db_path,
        on_progress=on_progress,
    )
