"""
Test Agent Config Provider/Model Fields

Tests that AgentConfig and DepartmentHeadConfig have provider and model fields.

**Validates: Task 3 - Provider/Model Fields**
"""

import pytest

from src.agents.config import AgentConfig
from src.agents.departments.types import DepartmentHeadConfig, Department


class TestAgentConfigProviderModel:
    """Test provider/model fields on AgentConfig."""

    def test_agent_config_has_provider_field(self):
        """AgentConfig should have provider field."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="analyst",
            name="Test Agent"
        )
        assert hasattr(config, "provider"), "AgentConfig should have provider field"

    def test_agent_config_has_model_field(self):
        """AgentConfig should have model field."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="analyst",
            name="Test Agent"
        )
        assert hasattr(config, "model"), "AgentConfig should have model field"

    def test_agent_config_provider_default(self):
        """AgentConfig provider should have default value."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="analyst",
            name="Test Agent"
        )
        assert config.provider == "anthropic"

    def test_agent_config_model_default(self):
        """AgentConfig model should have default value."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="analyst",
            name="Test Agent"
        )
        assert config.model == "claude-sonnet-4-20250514"


class TestDepartmentHeadConfigProviderModel:
    """Test provider/model fields on DepartmentHeadConfig."""

    def test_department_head_config_has_provider_field(self):
        """DepartmentHeadConfig should have provider field."""
        config = DepartmentHeadConfig(
            department=Department.RESEARCH,
            agent_type="research_head",
            system_prompt="Test prompt"
        )
        assert hasattr(config, "provider"), "DepartmentHeadConfig should have provider field"

    def test_department_head_config_has_model_field(self):
        """DepartmentHeadConfig should have model field."""
        config = DepartmentHeadConfig(
            department=Department.RESEARCH,
            agent_type="research_head",
            system_prompt="Test prompt"
        )
        assert hasattr(config, "model"), "DepartmentHeadConfig should have model field"

    def test_department_head_config_provider_default(self):
        """DepartmentHeadConfig provider should have default value."""
        config = DepartmentHeadConfig(
            department=Department.RESEARCH,
            agent_type="research_head",
            system_prompt="Test prompt"
        )
        assert config.provider == "anthropic"

    def test_department_head_config_model_default(self):
        """DepartmentHeadConfig model should have default value."""
        config = DepartmentHeadConfig(
            department=Department.RESEARCH,
            agent_type="research_head",
            system_prompt="Test prompt"
        )
        assert config.model == "claude-sonnet-4-20250514"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
