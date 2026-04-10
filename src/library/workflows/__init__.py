"""
QuantMindLib V1 — WF1 + WF2 Library-Side Bridges

Phase 9 Packet 9A: Library-side bridges that define the interface between
QuantMindLib and the (future) Prefect flows (flows/ does not exist yet).

Provides WF1Bridge (AlgoForge) and WF2Bridge (Improvement Loop) which wire
TRD/BotSpec through EvaluationOrchestrator, with state tracked via WorkflowBridge.
"""
from __future__ import annotations

from src.library.workflows.wf1_bridge import WF1Bridge
from src.library.workflows.wf2_bridge import WF2Bridge
from src.library.workflows.stub_flows import AlgoForgeFlowStub, ImprovementLoopFlowStub

__all__ = [
    "WF1Bridge",
    "WF2Bridge",
    "AlgoForgeFlowStub",
    "ImprovementLoopFlowStub",
]
