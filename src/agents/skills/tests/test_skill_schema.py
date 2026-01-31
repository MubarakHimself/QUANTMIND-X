"""
Tests for Skill Schema and Definitions.

Tests for Task Group 12.1: Skill validation, dependencies, versioning, and interface conformance.
"""

import pytest
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.agents.skills.skill_schema import (
    SkillDefinition,
    SkillDependencyResolver,
    SkillValidator,
    SkillRegistry,
    SkillCategory,
)


class TestSkillDefinitionValidation:
    """Test SkillDefinition validates all required fields."""

    def test_valid_skill_definition(self):
        """Test creating a valid skill definition."""
        skill = SkillDefinition(
            name="calculate_rsi",
            category="trading_skills",
            description="Calculate Relative Strength Index",
            input_schema={
                "type": "object",
                "properties": {
                    "period": {"type": "integer"},
                    "symbol": {"type": "string"}
                },
                "required": ["period", "symbol"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "rsi_value": {"type": "number"},
                    "signal": {"type": "string"}
                }
            },
            code="def calculate_rsi(period, symbol): return 70.0",
            dependencies=[],
            example_usage="rsi = calculate_rsi(14, 'EURUSD')",
            version="1.0.0"
        )
        assert skill.name == "calculate_rsi"
        assert skill.category == "trading_skills"
        assert skill.version == "1.0.0"

    def test_invalid_skill_name_uppercase(self):
        """Test skill name must be lowercase."""
        with pytest.raises(ValueError, match="lowercase with underscores"):
            SkillDefinition(
                name="CalculateRSI",
                category="trading_skills",
                description="Calculate RSI",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                code="code",
                dependencies=[],
                example_usage="usage",
                version="1.0.0"
            )

    def test_invalid_skill_name_special_chars(self):
        """Test skill name cannot have special characters."""
        with pytest.raises(ValueError, match="lowercase with underscores"):
            SkillDefinition(
                name="calculate-rsi!",
                category="trading_skills",
                description="Calculate RSI",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                code="code",
                dependencies=[],
                example_usage="usage",
                version="1.0.0"
            )

    def test_invalid_version_format(self):
        """Test version must follow semantic versioning."""
        with pytest.raises(ValueError, match="semantic versioning"):
            SkillDefinition(
                name="calculate_rsi",
                category="trading_skills",
                description="Calculate RSI",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                code="code",
                dependencies=[],
                example_usage="usage",
                version="1.0"
            )

    def test_valid_version_format(self):
        """Test valid semantic versions are accepted."""
        skill = SkillDefinition(
            name="test_skill",
            category="system_skills",
            description="Test",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code",
            dependencies=[],
            example_usage="usage",
            version="2.1.3"
        )
        assert skill.version == "2.1.3"

    def test_invalid_json_schema(self):
        """Test invalid JSON Schema is rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError before our validator
            SkillDefinition(
                name="test_skill",
                category="data_skills",
                description="Test",
                input_schema="not a dict",
                output_schema={"type": "object"},
                code="code",
                dependencies=[],
                example_usage="usage",
                version="1.0.0"
            )


class TestSkillDependencyResolution:
    """Test skill dependencies are resolved recursively."""

    def test_no_dependencies(self):
        """Test skill with no dependencies."""
        skill1 = SkillDefinition(
            name="skill_a",
            category="trading_skills",
            description="Skill A",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_a",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skills = {"skill_a": skill1}
        resolver = SkillDependencyResolver(skills)
        result = resolver.resolve("skill_a")
        assert len(result) == 1
        assert result[0].name == "skill_a"

    def test_single_dependency(self):
        """Test skill with one dependency."""
        skill1 = SkillDefinition(
            name="fetch_data",
            category="data_skills",
            description="Fetch data",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_fetch",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skill2 = SkillDefinition(
            name="calculate_rsi",
            category="trading_skills",
            description="Calculate RSI",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_rsi",
            dependencies=["fetch_data"],
            example_usage="usage",
            version="1.0.0"
        )
        skills = {"fetch_data": skill1, "calculate_rsi": skill2}
        resolver = SkillDependencyResolver(skills)
        result = resolver.resolve("calculate_rsi")
        assert len(result) == 2
        assert result[0].name == "fetch_data"
        assert result[1].name == "calculate_rsi"

    def test_nested_dependencies(self):
        """Test nested dependencies are resolved correctly."""
        skill1 = SkillDefinition(
            name="base_skill",
            category="data_skills",
            description="Base",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_base",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skill2 = SkillDefinition(
            name="mid_skill",
            category="trading_skills",
            description="Mid",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_mid",
            dependencies=["base_skill"],
            example_usage="usage",
            version="1.0.0"
        )
        skill3 = SkillDefinition(
            name="top_skill",
            category="trading_skills",
            description="Top",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_top",
            dependencies=["mid_skill"],
            example_usage="usage",
            version="1.0.0"
        )
        skills = {
            "base_skill": skill1,
            "mid_skill": skill2,
            "top_skill": skill3
        }
        resolver = SkillDependencyResolver(skills)
        result = resolver.resolve("top_skill")
        assert len(result) == 3
        assert result[0].name == "base_skill"
        assert result[1].name == "mid_skill"
        assert result[2].name == "top_skill"

    def test_circular_dependency_detection(self):
        """Test circular dependencies are detected."""
        skill1 = SkillDefinition(
            name="skill_a",
            category="trading_skills",
            description="A",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_a",
            dependencies=["skill_b"],
            example_usage="usage",
            version="1.0.0"
        )
        skill2 = SkillDefinition(
            name="skill_b",
            category="trading_skills",
            description="B",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_b",
            dependencies=["skill_a"],
            example_usage="usage",
            version="1.0.0"
        )
        skills = {"skill_a": skill1, "skill_b": skill2}
        resolver = SkillDependencyResolver(skills)
        with pytest.raises(ValueError, match="Circular dependency"):
            resolver.resolve("skill_a")

    def test_missing_dependency(self):
        """Test missing dependency raises error."""
        skill1 = SkillDefinition(
            name="skill_a",
            category="trading_skills",
            description="A",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_a",
            dependencies=["nonexistent_skill"],
            example_usage="usage",
            version="1.0.0"
        )
        skills = {"skill_a": skill1}
        resolver = SkillDependencyResolver(skills)
        with pytest.raises(ValueError, match="not found"):
            resolver.resolve("skill_a")


class TestSkillVersioning:
    """Test skill versioning allows multiple versions."""

    def test_register_multiple_versions(self):
        """Test registering multiple versions of same skill."""
        registry = SkillRegistry()
        skill_v1 = SkillDefinition(
            name="calculate_rsi",
            category="trading_skills",
            description="RSI v1",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_v1",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skill_v2 = SkillDefinition(
            name="calculate_rsi",
            category="trading_skills",
            description="RSI v2",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code_v2",
            dependencies=[],
            example_usage="usage",
            version="2.0.0"
        )
        registry.register(skill_v1)
        registry.register(skill_v2)

        versions = registry.list_versions("calculate_rsi")
        assert versions == ["2.0.0", "1.0.0"]

    def test_get_latest_version(self):
        """Test getting latest version by default."""
        registry = SkillRegistry()
        skill_v1 = SkillDefinition(
            name="test_skill",
            category="system_skills",
            description="Test v1",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="v1",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skill_v2 = SkillDefinition(
            name="test_skill",
            category="system_skills",
            description="Test v2",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="v2",
            dependencies=[],
            example_usage="usage",
            version="1.2.0"
        )
        registry.register(skill_v1)
        registry.register(skill_v2)

        latest = registry.get("test_skill")
        assert latest is not None
        assert latest.version == "1.2.0"
        assert latest.code == "v2"

    def test_get_specific_version(self):
        """Test getting specific version."""
        registry = SkillRegistry()
        skill_v1 = SkillDefinition(
            name="test_skill",
            category="trading_skills",
            description="Test v1",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="v1",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        skill_v2 = SkillDefinition(
            name="test_skill",
            category="trading_skills",
            description="Test v2",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="v2",
            dependencies=[],
            example_usage="usage",
            version="2.0.0"
        )
        registry.register(skill_v1)
        registry.register(skill_v2)

        v1_skill = registry.get("test_skill", "1.0.0")
        v2_skill = registry.get("test_skill", "2.0.0")
        assert v1_skill.code == "v1"
        assert v2_skill.code == "v2"

    def test_list_skills_by_category(self):
        """Test listing skills filtered by category."""
        registry = SkillRegistry()
        trading_skill = SkillDefinition(
            name="rsi",
            category="trading_skills",
            description="RSI",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        system_skill = SkillDefinition(
            name="logger",
            category="system_skills",
            description="Logger",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        registry.register(trading_skill)
        registry.register(system_skill)

        trading = registry.list_skills("trading_skills")
        system = registry.list_skills("system_skills")
        assert "rsi" in trading
        assert "logger" in system
        assert "logger" not in trading


class TestAgentSkillInterfaceConformance:
    """Test skill conforms to AgentSkill interface."""

    def test_conforms_to_interface(self):
        """Test valid skill conforms to AgentSkill interface."""
        skill = SkillDefinition(
            name="valid_skill",
            category="trading_skills",
            description="Valid skill",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="implementation code",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        assert skill.conforms_to_agent_skill_interface() is True

    def test_missing_name_fails(self):
        """Test skill without name fails conformance."""
        # Create skill with minimal valid name, then test conformance logic
        skill = SkillDefinition(
            name="test",  # Valid name for creation
            category="trading_skills",
            description="Valid skill",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code",
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        # Test conformance - check if name is empty
        assert skill.conforms_to_agent_skill_interface() is True
        # If we manually clear the name, conformance would fail
        skill.name = ""
        assert skill.conforms_to_agent_skill_interface() is False

    def test_to_agent_skill_dict(self):
        """Test conversion to AgentSkill compatible dictionary."""
        skill = SkillDefinition(
            name="test_skill",
            category="data_skills",
            description="Test skill",
            input_schema={"type": "object", "properties": {"value": {"type": "number"}}},
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
            code="def test(): return 'result'",
            dependencies=["dep1"],
            example_usage="result = test()",
            version="1.5.0"
        )
        result = skill.to_agent_skill_dict()
        assert result["name"] == "test_skill"
        assert result["category"] == "data_skills"
        assert result["version"] == "1.5.0"
        assert "dep1" in result["dependencies"]
        assert "properties" in result["input_schema"]


class TestSkillValidationFramework:
    """Test skill validation framework."""

    def test_validate_skill_success(self):
        """Test valid skill passes validation."""
        skill = SkillDefinition(
            name="valid_skill",
            category="trading_skills",
            description="Valid skill",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="implementation code",
            dependencies=[],
            example_usage="usage example",
            version="1.0.0"
        )
        errors = SkillValidator.validate_skill(skill)
        assert len(errors) == 0

    def test_validate_skill_empty_code(self):
        """Test skill with empty code fails validation."""
        skill = SkillDefinition(
            name="test",
            category="trading_skills",
            description="Test",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="",  # Empty code
            dependencies=[],
            example_usage="usage",
            version="1.0.0"
        )
        errors = SkillValidator.validate_skill(skill)
        assert len(errors) > 0
        assert any("Code field" in e for e in errors)

    def test_validate_skill_empty_example(self):
        """Test skill with empty example_usage fails validation."""
        skill = SkillDefinition(
            name="test",
            category="trading_skills",
            description="Test",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            code="code",
            dependencies=[],
            example_usage="",  # Empty example
            version="1.0.0"
        )
        errors = SkillValidator.validate_skill(skill)
        assert len(errors) > 0
        assert any("example_usage" in e for e in errors)

    def test_validate_input_data(self):
        """Test input data validation against schema."""
        schema = {
            "type": "object",
            "properties": {
                "period": {"type": "integer"},
                "symbol": {"type": "string"}
            },
            "required": ["period", "symbol"]
        }
        valid_data = {"period": 14, "symbol": "EURUSD"}
        invalid_data = {"period": 14}  # Missing symbol

        assert SkillValidator.validate_input(schema, valid_data) is True
        assert SkillValidator.validate_input(schema, invalid_data) is False

    def test_validate_output_data(self):
        """Test output data validation against schema."""
        schema = {
            "type": "object",
            "properties": {
                "rsi_value": {"type": "number"},
                "signal": {"type": "string"}
            }
        }
        valid_output = {"rsi_value": 70.5, "signal": "overbought"}
        # Invalid: string where number expected
        invalid_output = {"rsi_value": "not_a_number", "signal": "overbought"}

        assert SkillValidator.validate_output(schema, valid_output) is True
        # Type mismatch returns False
        assert SkillValidator.validate_output(schema, invalid_output) is False
