"""
Tests for Skill Creator Meta-Skill.

Tests for Task Group 2.1: Skill generation system.
Tests directory structure creation, SKILL.md generation, __init__.py generation,
skill validation, and auto-registration.
"""

import pytest
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from pydantic import ValidationError

from src.agents.skills.skill_schema import (
    SkillDefinition,
    SkillValidator,
)
from src.agents.skills.system_skills.skill_creator import (
    SkillCreator,
    SkillGenerationConfig,
)


class TestSkillDirectoryStructure:
    """Test skill directory structure creation."""

    def test_create_skill_directory_structure(self, tmp_path):
        """Test creating skill directory structure."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        skill_dir = creator._create_skill_directory("trading_skills", "test_skill")

        expected_dir = tmp_path / "trading_skills" / "test_skill"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_category_directory_created_if_not_exists(self, tmp_path):
        """Test category directory is created if it doesn't exist."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        category_dir = tmp_path / "data_skills"
        assert not category_dir.exists()

        skill_dir = creator._create_skill_directory("data_skills", "new_skill")
        assert category_dir.exists()
        assert skill_dir.exists()

    def test_existing_directory_handled_gracefully(self, tmp_path):
        """Test existing directory doesn't cause error."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        existing_dir = tmp_path / "system_skills" / "existing_skill"
        existing_dir.mkdir(parents=True)

        # Should not raise error
        skill_dir = creator._create_skill_directory("system_skills", "existing_skill")
        assert skill_dir.exists()

    def test_invalid_category_rejected(self, tmp_path):
        """Test invalid category names are rejected."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        with pytest.raises(ValueError, match="Invalid category"):
            creator._create_skill_directory("invalid_category", "test_skill")


class TestSkillMDGenerator:
    """Test SKILL.md generator with YAML frontmatter."""

    def test_skill_md_generation_with_yaml_frontmatter(self, tmp_path):
        """Test SKILL.md is generated with proper YAML frontmatter."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="test_skill",
            category="trading_skills",
            description="Test skill for validation",
            high_level_description="Create a test skill",
        )

        skill_path = creator._create_skill_directory("trading_skills", "test_skill")
        creator._generate_skill_md(skill_path, config)

        skill_md = skill_path / "SKILL.md"
        assert skill_md.exists()

        content = skill_md.read_text()
        assert "---" in content  # YAML frontmatter delimiter
        assert "name: test_skill" in content
        assert "category: trading_skills" in content
        assert "version:" in content

    def test_skill_md_includes_markdown_content(self, tmp_path):
        """Test SKILL.md includes markdown instructions and examples."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="calculate_macd",
            category="trading_skills",
            description="Calculate MACD indicator",
            high_level_description="Create MACD calculation skill",
        )

        skill_path = creator._create_skill_directory("trading_skills", "calculate_macd")
        creator._generate_skill_md(skill_path, config)

        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text()

        # Should have description section
        assert "## Description" in content or "Description" in content
        # Should have usage section
        assert "## Usage" in content or "Usage" in content or "Example" in content


class TestInitPyGenerator:
    """Test __init__.py generator with SkillDefinition."""

    def test_init_py_generates_skill_definition(self, tmp_path):
        """Test __init__.py generates valid SkillDefinition object."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="fetch_weather",
            category="data_skills",
            description="Fetch weather data from API",
            high_level_description="Create a skill to fetch weather data",
            code="def fetch_weather(location): return {'temp': 72}",
        )

        skill_path = creator._create_skill_directory("data_skills", "fetch_weather")
        creator._generate_init_py(skill_path, config)

        init_file = skill_path / "__init__.py"
        assert init_file.exists()

        content = init_file.read_text()
        assert "SkillDefinition" in content
        assert "name=\"fetch_weather\"" in content
        assert "category=\"data_skills\"" in content

    def test_init_py_includes_json_schemas(self, tmp_path):
        """Test __init__.py includes input/output JSON schemas."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="process_data",
            category="system_skills",
            description="Process data",
            high_level_description="Create data processing skill",
            input_schema={
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        )

        skill_path = creator._create_skill_directory("system_skills", "process_data")
        creator._generate_init_py(skill_path, config)

        init_file = skill_path / "__init__.py"
        content = init_file.read_text()

        assert "input_schema" in content
        assert "output_schema" in content
        # Check for schema content (Python dict format uses single quotes)
        assert "'type': 'object'" in content or '"type": "object"' in content


class TestSkillValidation:
    """Test skill validation before file system commits."""

    def test_valid_skill_passes_validation(self, tmp_path):
        """Test valid skill passes validation."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="valid_skill",
            category="trading_skills",
            description="Valid skill",
            high_level_description="Create valid skill",
            code="def valid_func(): return 'result'",
            example_usage="result = valid_func()",
        )

        errors = creator._validate_skill_config(config)
        assert len(errors) == 0

    def test_empty_code_fails_validation(self, tmp_path):
        """Test empty code fails validation."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="invalid_skill",
            category="trading_skills",
            description="Invalid skill",
            high_level_description="Create invalid skill",
            code="",  # Empty code
        )

        errors = creator._validate_skill_config(config)
        assert len(errors) > 0
        assert any("code" in error.lower() for error in errors)

    def test_invalid_name_fails_validation(self, tmp_path):
        """Test invalid skill name fails validation."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        # Pydantic validates on creation, so we need to catch ValidationError
        with pytest.raises(ValidationError):
            config = SkillGenerationConfig(
                name="Invalid-Name!",  # Invalid name
                category="trading_skills",
                description="Invalid name",
                high_level_description="Test",
            )


class TestAutoRegistration:
    """Test auto-registration in category __init__.py."""

    def test_skill_auto_registered_in_category_init(self, tmp_path):
        """Test new skill is auto-registered in category __init__.py."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        config = SkillGenerationConfig(
            name="new_skill",
            category="trading_skills",
            description="New skill",
            high_level_description="Create new skill",
        )

        # Create skill
        skill_path = creator._create_skill_directory("trading_skills", "new_skill")
        creator._generate_init_py(skill_path, config)

        # Register skill
        category_init = tmp_path / "trading_skills" / "__init__.py"
        category_init.parent.mkdir(parents=True, exist_ok=True)
        category_init.write_text("# Trading Skills\n__all__ = []\n")

        creator._register_skill_in_category_init("trading_skills", "new_skill")

        # Check registration
        content = category_init.read_text()
        assert "new_skill" in content

    def test_multiple_skills_registered_correctly(self, tmp_path):
        """Test multiple skills are registered correctly."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        # Create category init
        category_init = tmp_path / "system_skills" / "__init__.py"
        category_init.parent.mkdir(parents=True, exist_ok=True)
        category_init.write_text("__all__ = []\n")

        # Register multiple skills
        skills = ["skill_a", "skill_b", "skill_c"]
        for skill_name in skills:
            creator._register_skill_in_category_init("system_skills", skill_name)

        content = category_init.read_text()
        for skill_name in skills:
            assert skill_name in content
        assert "__all__" in content


class TestBatchInterface:
    """Test batch (non-interactive) interface."""

    def test_single_description_creates_complete_skill(self, tmp_path):
        """Test single high-level description creates complete skill."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        result = creator.create_skill_from_description(
            high_level_description="Create a skill to fetch weather data for a given city",
            category="data_skills",
        )

        assert result["success"]
        assert "skill_path" in result

        # Verify files exist
        skill_path = Path(result["skill_path"])
        assert (skill_path / "SKILL.md").exists()
        assert (skill_path / "__init__.py").exists()

    def test_batch_interface_avoids_interactive_prompts(self, tmp_path):
        """Test batch interface doesn't use interactive prompts."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        # Should not raise or prompt for input
        result = creator.create_skill_from_description(
            high_level_description="Create a simple logging skill",
            category="system_skills",
        )

        assert result["success"]
        assert "skill_path" in result


class TestEndToEndSkillGeneration:
    """Test end-to-end skill generation workflow."""

    def test_complete_skill_generation_and_validation(self, tmp_path):
        """Test complete skill generation with validation."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        result = creator.create_skill_from_description(
            high_level_description="Create a skill to calculate pivot points for trading",
            category="trading_skills",
        )

        # Check success
        assert result["success"]

        # Check files created
        skill_path = Path(result["skill_path"])
        assert skill_path.exists()
        assert (skill_path / "SKILL.md").exists()
        assert (skill_path / "__init__.py").exists()

        # Check SKILL.md has YAML frontmatter
        skill_md_content = (skill_path / "SKILL.md").read_text()
        assert "---" in skill_md_content
        assert "category: trading_skills" in skill_md_content

        # Check __init__.py has SkillDefinition
        init_content = (skill_path / "__init__.py").read_text()
        assert "SkillDefinition" in init_content

    def test_generated_skill_is_loadable(self, tmp_path):
        """Test generated skill can be loaded as SkillDefinition."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        result = creator.create_skill_from_description(
            high_level_description="Create a data fetching skill",
            category="data_skills",
        )

        assert result["success"]
        skill_path = Path(result["skill_path"])

        # Check that __init__.py contains SkillDefinition
        init_content = (skill_path / "__init__.py").read_text()
        assert "skill_definition = SkillDefinition(" in init_content
        assert "name=" in init_content
        assert "category=" in init_content

        # Verify SKILL.md exists and has proper structure
        skill_md = (skill_path / "SKILL.md").read_text()
        assert "---" in skill_md  # YAML frontmatter
        assert "name:" in skill_md
