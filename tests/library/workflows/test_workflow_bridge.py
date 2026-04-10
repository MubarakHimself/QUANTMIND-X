"""
Tests for QuantMindLib V1 — WorkflowBridge (Packet 9B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    WorkflowArtifact,
    WorkflowBridge,
    WorkflowState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wb() -> WorkflowBridge:
    """Fresh WorkflowBridge for each test."""
    return WorkflowBridge()


# ---------------------------------------------------------------------------
# TestWorkflowBridge
# ---------------------------------------------------------------------------


class TestWorkflowBridge:
    """Tests for WorkflowBridge state management."""

    def test_register_workflow(self, wb: WorkflowBridge):
        """register_workflow() creates a new workflow state."""
        state = wb.register_workflow(
            workflow_id="wf1-test-001",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        assert state is not None
        assert state.workflow_id == "wf1-test-001"
        assert state.workflow_name == "WF1_ALGOFORGE"
        assert state.status == "PENDING"
        assert state.inputs == []
        assert state.outputs == []

    def test_register_workflow_with_artifacts(self, wb: WorkflowBridge):
        """register_workflow() stores input artifacts."""
        artifact = WorkflowArtifact(
            artifact_id="art-001",
            artifact_type="TRD",
            workflow_id="wf1-test-002",
            created_at_ms=1000,
        )
        state = wb.register_workflow(
            workflow_id="wf1-test-002",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[artifact],
        )

        assert state is not None
        assert len(state.inputs) == 1
        assert state.inputs[0].artifact_type == "TRD"

    def test_start_workflow(self, wb: WorkflowBridge):
        """start_workflow() marks workflow as RUNNING with timestamp."""
        wb.register_workflow(
            workflow_id="wf1-test-003",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        result = wb.start_workflow("wf1-test-003")

        assert result is True
        state = wb.get_workflow("wf1-test-003")
        assert state is not None
        assert state.status == "RUNNING"
        assert state.started_at_ms is not None
        assert state.started_at_ms > 0

    def test_start_workflow_unknown_id(self, wb: WorkflowBridge):
        """start_workflow() returns False for unknown workflow_id."""
        result = wb.start_workflow("unknown-workflow")
        assert result is False

    def test_complete_workflow(self, wb: WorkflowBridge):
        """complete_workflow() marks workflow as COMPLETED with outputs."""
        wb.register_workflow(
            workflow_id="wf1-test-004",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        wb.start_workflow("wf1-test-004")

        output_artifact = WorkflowArtifact(
            artifact_id="eval-art-001",
            artifact_type="EvaluationResult",
            workflow_id="wf1-test-004",
            created_at_ms=2000,
        )
        result = wb.complete_workflow(
            "wf1-test-004",
            output_artifacts=[output_artifact],
        )

        assert result is True
        state = wb.get_workflow("wf1-test-004")
        assert state is not None
        assert state.status == "COMPLETED"
        assert state.completed_at_ms is not None
        assert state.completed_at_ms > 0
        assert len(state.outputs) == 1
        assert state.outputs[0].artifact_type == "EvaluationResult"

    def test_complete_workflow_unknown_id(self, wb: WorkflowBridge):
        """complete_workflow() returns False for unknown workflow_id."""
        result = wb.complete_workflow("unknown-workflow")
        assert result is False

    def test_fail_workflow(self, wb: WorkflowBridge):
        """fail_workflow() marks workflow as FAILED."""
        wb.register_workflow(
            workflow_id="wf1-test-005",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        result = wb.fail_workflow("wf1-test-005")

        assert result is True
        state = wb.get_workflow("wf1-test-005")
        assert state is not None
        assert state.status == "FAILED"

    def test_fail_workflow_unknown_id(self, wb: WorkflowBridge):
        """fail_workflow() returns False for unknown workflow_id."""
        result = wb.fail_workflow("unknown-workflow")
        assert result is False

    def test_get_workflow(self, wb: WorkflowBridge):
        """get_workflow() returns the workflow state."""
        wb.register_workflow(
            workflow_id="wf1-test-006",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        state = wb.get_workflow("wf1-test-006")

        assert state is not None
        assert state.workflow_id == "wf1-test-006"

    def test_get_workflow_unknown_returns_none(self, wb: WorkflowBridge):
        """get_workflow() returns None for unknown workflow_id."""
        state = wb.get_workflow("nonexistent-id")
        assert state is None

    def test_is_wf1_to_wf2_ready(self, wb: WorkflowBridge):
        """is_wf1_to_wf2_ready() True when AlgoForge workflow is handoff-ready."""
        # No workflows yet
        assert wb.is_wf1_to_wf2_ready() is False

        # PENDING workflow is not ready
        wb.register_workflow(
            workflow_id="wf1-pending",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        assert wb.is_wf1_to_wf2_ready() is False

        # COMPLETED workflow with outputs is ready
        wb.start_workflow("wf1-pending")
        output_artifact = WorkflowArtifact(
            artifact_id="out-001",
            artifact_type="BotSpec",
            workflow_id="wf1-pending",
            created_at_ms=1000,
        )
        wb.complete_workflow("wf1-pending", output_artifacts=[output_artifact])
        assert wb.is_wf1_to_wf2_ready() is True

    def test_handoff_ready_workflows(self, wb: WorkflowBridge):
        """get_handoff_ready_workflows() returns COMPLETED workflows."""
        # Register and complete a WF1 workflow
        artifact = WorkflowArtifact(
            artifact_id="wf1-art",
            artifact_type="BotSpec",
            workflow_id="wf1-ready",
            created_at_ms=1000,
        )
        wb.register_workflow(
            workflow_id="wf1-ready",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[artifact],
        )
        wb.start_workflow("wf1-ready")
        output_artifact = WorkflowArtifact(
            artifact_id="out-art",
            artifact_type="EvaluationResult",
            workflow_id="wf1-ready",
            created_at_ms=2000,
        )
        wb.complete_workflow("wf1-ready", output_artifacts=[output_artifact])

        # WF2_IMPROVEMENT_LOOP should not be included
        ready_list = wb.get_handoff_ready_workflows("WF2_IMPROVEMENT_LOOP")
        wf1_ready = [wf for wf in ready_list if wf.workflow_id == "wf1-ready"]
        assert len(wf1_ready) == 1

    def test_workflow_state_is_handoff_ready(self, wb: WorkflowBridge):
        """WorkflowState.is_handoff_ready() True when COMPLETED with outputs."""
        wb.register_workflow(
            workflow_id="wf1-ready-check",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        wb.start_workflow("wf1-ready-check")
        output_artifact = WorkflowArtifact(
            artifact_id="out-002",
            artifact_type="EvaluationResult",
            workflow_id="wf1-ready-check",
            created_at_ms=3000,
        )
        wb.complete_workflow("wf1-ready-check", output_artifacts=[output_artifact])

        state = wb.get_workflow("wf1-ready-check")
        assert state is not None
        assert state.is_handoff_ready() is True

    def test_workflow_state_is_blocked(self, wb: WorkflowBridge):
        """WorkflowState.is_blocked() True when PENDING with no inputs."""
        state = wb.register_workflow(
            workflow_id="wf1-blocked",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        assert state.is_blocked() is True

        # Has inputs = not blocked
        artifact = WorkflowArtifact(
            artifact_id="art-block",
            artifact_type="TRD",
            workflow_id="wf1-unblocked",
            created_at_ms=1000,
        )
        state2 = wb.register_workflow(
            workflow_id="wf1-unblocked",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[artifact],
        )
        assert state2.is_blocked() is False
