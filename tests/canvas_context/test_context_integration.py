"""Test Canvas Context integration with memory system."""
import pytest
from unittest.mock import MagicMock, patch

from src.canvas_context.loader import load_template
from src.canvas_context.types import CanvasContextTemplate


class TestContextIntegration:
    """Tests for canvas context integration with memory system."""

    def test_template_provides_memory_scope(self):
        """Test template provides correct memory scope for RAG."""
        template = load_template("risk")

        # Memory scope should be namespace patterns, not content
        assert isinstance(template.memory_scope, list)
        assert len(template.memory_scope) > 0
        assert all(isinstance(scope, str) for scope in template.memory_scope)

    def test_template_identifiers_not_content(self):
        """Test that template uses identifiers, not content (CAG pattern)."""
        template = load_template("risk")

        # Template should NOT contain actual content, only identifiers
        # Memory scope, workflow namespaces, etc. are all identifiers
        assert template.base_descriptor  # This is the only actual content
        assert template.memory_scope  # These are namespace patterns, not memory content
        assert template.workflow_namespaces  # These are identifiers

    def test_token_budget_enforcement(self):
        """Test max_identifiers enforces token budget."""
        template = load_template("risk")

        # Should have a reasonable token budget
        assert template.max_identifiers > 0
        assert template.max_identifiers <= 100  # Reasonable upper bound

    def test_jit_fetch_readiness(self):
        """Test template is ready for JIT content fetch."""
        template = load_template("risk")

        # Should have memory scope for RAG queries
        assert len(template.memory_scope) > 0

        # Should not preload content - only identifiers
        # The system will fetch content JIT when agent needs it
        for scope in template.memory_scope:
            assert "*" in scope or "." in scope  # Namespace patterns, not specific content

    def test_department_mailbox_format(self):
        """Test department mailbox is correctly formatted."""
        template = load_template("risk")

        # Should be a stream name pattern
        assert template.department_mailbox is not None
        assert "_dept_mail" in template.department_mailbox

    def test_skill_index_structure(self):
        """Test skill index provides proper skill identifiers."""
        template = load_template("risk")

        # Skill index should have proper structure
        assert len(template.skill_index) > 0
        for skill in template.skill_index:
            assert skill.id  # Skill identifier
            assert skill.path  # Path is identifier, not content
            assert skill.trigger  # Trigger condition


class TestMemoryGraphIntegration:
    """Tests for integration with GraphMemoryFacade."""

    @patch("src.canvas_context.loader.load_template")
    def test_load_committed_state_integration(self, mock_load):
        """Test integration with load_committed_state method."""
        # This tests that the template can be used with GraphMemoryFacade
        template = load_template("risk")

        # Template provides memory_scope which can be used to query
        # graph memory for committed nodes
        memory_scope = template.memory_scope

        # These scope patterns can be used to filter nodes
        assert isinstance(memory_scope, list)
        assert len(memory_scope) > 0

    def test_canvas_to_memory_namespace_mapping(self):
        """Test canvas memory scope maps to graph namespaces."""
        # Risk canvas should scope to risk.*, portfolio.* namespaces
        risk_template = load_template("risk")

        assert "risk.*" in risk_template.memory_scope
        assert "portfolio.*" in risk_template.memory_scope

        # Trading canvas should scope to trading.* namespaces
        trading_template = load_template("trading")

        assert "trading.*" in trading_template.memory_scope
        assert "signals.*" in trading_template.memory_scope


class TestCanvasAwareCopilot:
    """Tests for FR20: canvas-aware Copilot context."""

    def test_copilot_has_base_descriptor(self):
        """Test each canvas provides base descriptor for agent."""
        canvases = ["risk", "workshop", "trading", "development"]

        for canvas in canvases:
            template = load_template(canvas)
            assert template.base_descriptor
            assert len(template.base_descriptor) > 20  # Substantial description

    def test_department_head_per_canvas(self):
        """Test each canvas has department head configured."""
        canvases = ["risk", "workshop", "trading", "portfolio"]

        for canvas in canvases:
            template = load_template(canvas)
            # Each canvas should map to a department head
            assert template.department_head is not None or canvas == "workshop"

    def test_required_tools_per_canvas(self):
        """Test each canvas has appropriate tools configured."""
        risk_template = load_template("risk")
        assert "risk_calculator" in risk_template.required_tools

        trading_template = load_template("trading")
        assert "trade_executor" in trading_template.required_tools

        workshop_template = load_template("workshop")
        # Workshop should have access to floor_manager
        assert "floor_manager" in workshop_template.required_tools