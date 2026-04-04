"""
Workflow Data Models

Contains all dataclasses and enums for the department workflow system.
Covers all 4 workflows from architecture.md §20 and PRD addendum S3-9:
- WF1: AlphaForge Creation Pipeline
- WF2: AlphaForge Enhancement Loop (continuous)
- WF3: Performance Intelligence (Dead Zone daily)
- WF4: Weekend Update Cycle (weekly improvement)

Also includes the 8-state EA variant promotion lifecycle (architecture §20.4).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# ===========================================================================
# Workflow Type — which of the 4 canonical workflows
# ===========================================================================

class WorkflowType(str, Enum):
    """The 4 canonical workflows from architecture §20 / PRD S3-9."""
    WF1_CREATION = "wf1_creation"
    WF2_ENHANCEMENT = "wf2_enhancement"
    WF3_PERFORMANCE_INTEL = "wf3_performance_intel"
    WF4_WEEKEND_UPDATE = "wf4_weekend_update"
    CUSTOM = "custom"


# ===========================================================================
# Workflow Stage — stages per workflow type
# ===========================================================================

class WorkflowStage(str, Enum):
    """Department workflow stages (unified across all workflow types)."""
    # WF1: AlphaForge Creation Pipeline stages
    VIDEO_INGEST = "video_ingest"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"

    # WF2: Enhancement Loop stages
    EA_LIBRARY_SCAN = "ea_library_scan"
    FULL_BACKTEST_MATRIX = "full_backtest_matrix"
    RESEARCH_IMPROVEMENT = "research_improvement"
    VARIANT_GENERATION = "variant_generation"
    DATA_VARIANT_STRESS = "data_variant_stress"
    SIT_GATE = "sit_gate"
    PAPER_TRADING_MONITOR = "paper_trading_monitor"
    HUMAN_APPROVAL_GATE = "human_approval_gate"

    # WF3: Performance Intelligence stages (Dead Zone 16:15–18:00 GMT)
    EOD_REPORT = "eod_report"
    SESSION_PERFORMER_ID = "session_performer_id"
    DPR_UPDATE = "dpr_update"
    QUEUE_RERANK = "queue_rerank"
    FORTNIGHT_ACCUMULATION = "fortnight_accumulation"

    # WF4: Weekend Update Cycle stages
    FRIDAY_ANALYSIS = "friday_analysis"
    SATURDAY_REFINEMENT = "saturday_refinement"
    SUNDAY_PREMARKET = "sunday_premarket"
    MONDAY_ROSTER = "monday_roster"

    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"


# ===========================================================================
# EA Variant Promotion Lifecycle (architecture §20.4)
# ===========================================================================

class EALifecycleState(str, Enum):
    """8-state EA variant promotion lifecycle."""
    DRAFT = "draft"
    TESTING = "testing"
    SIT_PASSED = "sit_passed"
    PAPER_TRADING = "paper_trading"
    PAPER_VALIDATED = "paper_validated"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    ARCHIVED = "archived"
    RETIRED = "retired"


# ===========================================================================
# Task Priority (architecture §4.2)
# ===========================================================================

class TaskPriority(str, Enum):
    """Three-tier priority on all task entries."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ===========================================================================
# Workflow Status
# ===========================================================================

class WorkflowStatus(str, Enum):
    """Status of a workflow run."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED_REVIEW = "expired_review"


# ===========================================================================
# Workflow Task
# ===========================================================================

@dataclass
class WorkflowTask:
    """A single task in the workflow."""
    task_id: str
    stage: WorkflowStage
    from_dept: str
    to_dept: str
    message_id: str
    status: WorkflowStatus
    created_at: datetime
    priority: TaskPriority = TaskPriority.MEDIUM
    completed_at: Optional[datetime] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===========================================================================
# Department Workflow
# ===========================================================================

@dataclass
class DepartmentWorkflow:
    """Represents a complete department workflow."""
    workflow_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    current_stage: WorkflowStage
    created_at: datetime
    updated_at: datetime
    tasks: List[WorkflowTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    strategy_id: Optional[str] = None
    ea_lifecycle_state: Optional[EALifecycleState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "stage": t.stage.value,
                    "from_dept": t.from_dept,
                    "to_dept": t.to_dept,
                    "message_id": t.message_id,
                    "status": t.status.value,
                    "priority": t.priority.value,
                    "created_at": t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "payload": t.payload,
                    "result": t.result,
                    "error": t.error,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
            "error": self.error,
            "strategy_id": self.strategy_id,
            "ea_lifecycle_state": self.ea_lifecycle_state.value if self.ea_lifecycle_state else None,
        }

    def get_progress(self) -> float:
        """Calculate progress percentage based on workflow type."""
        stage_maps = {
            WorkflowType.WF1_CREATION: [
                WorkflowStage.VIDEO_INGEST, WorkflowStage.RESEARCH,
                WorkflowStage.DEVELOPMENT, WorkflowStage.BACKTESTING,
                WorkflowStage.COMPLETED,
            ],
            WorkflowType.WF2_ENHANCEMENT: [
                WorkflowStage.EA_LIBRARY_SCAN, WorkflowStage.FULL_BACKTEST_MATRIX,
                WorkflowStage.RESEARCH_IMPROVEMENT, WorkflowStage.VARIANT_GENERATION,
                WorkflowStage.DATA_VARIANT_STRESS, WorkflowStage.SIT_GATE,
                WorkflowStage.PAPER_TRADING_MONITOR, WorkflowStage.HUMAN_APPROVAL_GATE,
                WorkflowStage.COMPLETED,
            ],
            WorkflowType.WF3_PERFORMANCE_INTEL: [
                WorkflowStage.EOD_REPORT, WorkflowStage.SESSION_PERFORMER_ID,
                WorkflowStage.DPR_UPDATE, WorkflowStage.QUEUE_RERANK,
                WorkflowStage.FORTNIGHT_ACCUMULATION, WorkflowStage.COMPLETED,
            ],
            WorkflowType.WF4_WEEKEND_UPDATE: [
                WorkflowStage.FRIDAY_ANALYSIS, WorkflowStage.SATURDAY_REFINEMENT,
                WorkflowStage.SUNDAY_PREMARKET, WorkflowStage.MONDAY_ROSTER,
                WorkflowStage.COMPLETED,
            ],
        }

        stages = stage_maps.get(self.workflow_type, [WorkflowStage.COMPLETED])
        try:
            idx = stages.index(self.current_stage)
            return (idx / max(1, len(stages) - 1)) * 100
        except ValueError:
            return 0.0


# ===========================================================================
# Workflow Stage Routing Tables
# ===========================================================================

# WF1: Creation Pipeline — stage → department mapping
WF1_STAGE_DEPARTMENTS = {
    WorkflowStage.VIDEO_INGEST: "research",
    WorkflowStage.RESEARCH: "research",
    WorkflowStage.DEVELOPMENT: "development",
    WorkflowStage.BACKTESTING: "development",
}

# WF1: stage → next stage
WF1_NEXT_STAGE = {
    WorkflowStage.VIDEO_INGEST: WorkflowStage.RESEARCH,
    WorkflowStage.RESEARCH: WorkflowStage.DEVELOPMENT,
    WorkflowStage.DEVELOPMENT: WorkflowStage.BACKTESTING,
    WorkflowStage.BACKTESTING: WorkflowStage.COMPLETED,
}

# WF2: Enhancement Loop — stage → department mapping
WF2_STAGE_DEPARTMENTS = {
    WorkflowStage.EA_LIBRARY_SCAN: "portfolio",
    WorkflowStage.FULL_BACKTEST_MATRIX: "development",
    WorkflowStage.RESEARCH_IMPROVEMENT: "research",
    WorkflowStage.VARIANT_GENERATION: "development",
    WorkflowStage.DATA_VARIANT_STRESS: "development",
    WorkflowStage.SIT_GATE: "risk",
    WorkflowStage.PAPER_TRADING_MONITOR: "trading",
    WorkflowStage.HUMAN_APPROVAL_GATE: "floor_manager",
}

WF2_NEXT_STAGE = {
    WorkflowStage.EA_LIBRARY_SCAN: WorkflowStage.FULL_BACKTEST_MATRIX,
    WorkflowStage.FULL_BACKTEST_MATRIX: WorkflowStage.RESEARCH_IMPROVEMENT,
    WorkflowStage.RESEARCH_IMPROVEMENT: WorkflowStage.VARIANT_GENERATION,
    WorkflowStage.VARIANT_GENERATION: WorkflowStage.DATA_VARIANT_STRESS,
    WorkflowStage.DATA_VARIANT_STRESS: WorkflowStage.SIT_GATE,
    WorkflowStage.SIT_GATE: WorkflowStage.PAPER_TRADING_MONITOR,
    WorkflowStage.PAPER_TRADING_MONITOR: WorkflowStage.HUMAN_APPROVAL_GATE,
    WorkflowStage.HUMAN_APPROVAL_GATE: WorkflowStage.COMPLETED,
}

# WF3: Performance Intelligence — stage → department mapping
WF3_STAGE_DEPARTMENTS = {
    WorkflowStage.EOD_REPORT: "trading",
    WorkflowStage.SESSION_PERFORMER_ID: "trading",
    WorkflowStage.DPR_UPDATE: "portfolio",
    WorkflowStage.QUEUE_RERANK: "portfolio",
    WorkflowStage.FORTNIGHT_ACCUMULATION: "risk",
}

WF3_NEXT_STAGE = {
    WorkflowStage.EOD_REPORT: WorkflowStage.SESSION_PERFORMER_ID,
    WorkflowStage.SESSION_PERFORMER_ID: WorkflowStage.DPR_UPDATE,
    WorkflowStage.DPR_UPDATE: WorkflowStage.QUEUE_RERANK,
    WorkflowStage.QUEUE_RERANK: WorkflowStage.FORTNIGHT_ACCUMULATION,
    WorkflowStage.FORTNIGHT_ACCUMULATION: WorkflowStage.COMPLETED,
}

# WF4: Weekend Update Cycle — stage → department mapping
WF4_STAGE_DEPARTMENTS = {
    WorkflowStage.FRIDAY_ANALYSIS: "research",
    WorkflowStage.SATURDAY_REFINEMENT: "development",
    WorkflowStage.SUNDAY_PREMARKET: "risk",
    WorkflowStage.MONDAY_ROSTER: "portfolio",
}

WF4_NEXT_STAGE = {
    WorkflowStage.FRIDAY_ANALYSIS: WorkflowStage.SATURDAY_REFINEMENT,
    WorkflowStage.SATURDAY_REFINEMENT: WorkflowStage.SUNDAY_PREMARKET,
    WorkflowStage.SUNDAY_PREMARKET: WorkflowStage.MONDAY_ROSTER,
    WorkflowStage.MONDAY_ROSTER: WorkflowStage.COMPLETED,
}

# Unified lookup
WORKFLOW_STAGE_DEPARTMENTS = {
    WorkflowType.WF1_CREATION: WF1_STAGE_DEPARTMENTS,
    WorkflowType.WF2_ENHANCEMENT: WF2_STAGE_DEPARTMENTS,
    WorkflowType.WF3_PERFORMANCE_INTEL: WF3_STAGE_DEPARTMENTS,
    WorkflowType.WF4_WEEKEND_UPDATE: WF4_STAGE_DEPARTMENTS,
}

WORKFLOW_NEXT_STAGES = {
    WorkflowType.WF1_CREATION: WF1_NEXT_STAGE,
    WorkflowType.WF2_ENHANCEMENT: WF2_NEXT_STAGE,
    WorkflowType.WF3_PERFORMANCE_INTEL: WF3_NEXT_STAGE,
    WorkflowType.WF4_WEEKEND_UPDATE: WF4_NEXT_STAGE,
}

# First stage per workflow type
WORKFLOW_INITIAL_STAGES = {
    WorkflowType.WF1_CREATION: WorkflowStage.VIDEO_INGEST,
    WorkflowType.WF2_ENHANCEMENT: WorkflowStage.EA_LIBRARY_SCAN,
    WorkflowType.WF3_PERFORMANCE_INTEL: WorkflowStage.EOD_REPORT,
    WorkflowType.WF4_WEEKEND_UPDATE: WorkflowStage.FRIDAY_ANALYSIS,
}
