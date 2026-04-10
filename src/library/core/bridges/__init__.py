"""
QuantMindLib V1 — Bridge Definitions
"""
from src.library.core.bridges.registry_journal_bridges import RegistryBridge, JournalEntry, JournalBridge
from src.library.core.bridges.sentinel_dpr_bridges import SentinelBridge, DPRScore, DPRBridge
from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    LifecycleBridge,
    LifecycleTransition,
    EvaluationBridge,
    WorkflowBridge,
    WorkflowArtifact,
    WorkflowState,
)

__all__ = [
    # Sentinel + DPR
    "SentinelBridge",
    "DPRScore",
    "DPRBridge",
    # Registry + Journal
    "RegistryBridge",
    "JournalEntry",
    "JournalBridge",
    # Lifecycle + Evaluation + Workflow
    "LifecycleBridge",
    "LifecycleTransition",
    "EvaluationBridge",
    "WorkflowBridge",
    "WorkflowArtifact",
    "WorkflowState",
]
