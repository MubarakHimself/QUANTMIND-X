"""
Skill Creator Meta-Skill for QuantMind Agents.

This module provides a batch (non-interactive) code generation system that
creates new skills from high-level descriptions. It handles directory structure
creation, SKILL.md generation, __init__.py generation with SkillDefinition,
validation, and auto-registration.

Implementation for Task Group 2: Skill Generation System
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

from ..skill_schema import (
    SkillCategory,
    SkillDefinition,
    SkillValidator,
)


logger = logging.getLogger(__name__)


# Valid skill categories matching SkillCategory literal
VALID_CATEGORIES = ["trading_skills", "system_skills", "data_skills"]


class SkillGenerationConfig(BaseModel):
    """
    Configuration for skill generation.

    This model captures the input parameters for generating a new skill.
    """

    name: str = Field(..., description="Generated skill name (snake_case)")
    category: str = Field(..., description="Skill category")
    description: str = Field(..., description="Human-readable skill description")
    high_level_description: str = Field(
        ..., description="Original high-level description from user"
    )
    code: Optional[str] = Field(
        default="pass  # TODO: Implement skill logic",
        description="Generated Python code"
    )
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        },
        description="JSON Schema for input validation"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
        },
        description="JSON Schema for output validation"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of required skill names"
    )
    example_usage: Optional[str] = Field(
        default=None,
        description="Example demonstrating skill usage"
    )
    version: str = Field(default="1.0.0", description="Semantic version")

    @validator("category")
    def validate_category(cls, v: str) -> str:
        """Validate category is one of the valid categories."""
        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {v}. Must be one of {VALID_CATEGORIES}"
            )
        return v

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate skill name is lowercase with underscores."""
        if not re.match(r"^[a-z][a-z0-9_]*[a-z0-9]$", v):
            raise ValueError(
                "Skill name must be lowercase with underscores, starting with a letter"
            )
        return v

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError("Version must follow semantic versioning (e.g., '1.0.0')")
        return v


class SkillCreator:
    """
    Skill Creator Meta-Skill for automated skill generation.

    This class provides a batch (non-interactive) interface for creating
    new skills from high-level descriptions. It handles:

    1. Directory structure creation
    2. SKILL.md generation with YAML frontmatter
    3. __init__.py generation with SkillDefinition
    4. Skill validation before file system commits
    5. Auto-registration in category __init__.py

    Usage:
        creator = SkillCreator(skills_base_dir="src/agents/skills")
        result = creator.create_skill_from_description(
            high_level_description="Create a skill to fetch weather data",
            category="data_skills"
        )
    """

    def __init__(
        self,
        skills_base_dir: str = "src/agents/skills",
        enable_validation: bool = True,
        enable_auto_registration: bool = True,
    ):
        """
        Initialize SkillCreator.

        Args:
            skills_base_dir: Base directory for skills (default: src/agents/skills)
            enable_validation: Whether to validate skills before file commits
            enable_auto_registration: Whether to auto-register in category __init__.py
        """
        self.skills_base_dir = Path(skills_base_dir)
        self.enable_validation = enable_validation
        self.enable_auto_registration = enable_auto_registration
        self.logger = logging.getLogger(f"{__name__}.SkillCreator")

        # Validate base directory exists
        if not self.skills_base_dir.exists():
            self.logger.warning(
                f"Skills base directory does not exist: {self.skills_base_dir}"
            )

    def create_skill_from_description(
        self,
        high_level_description: str,
        category: str = "system_skills",
        name: Optional[str] = None,
        code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete skill from a high-level description.

        This is the main entry point for skill generation. It takes a single
        high-level description and generates all necessary files.

        Args:
            high_level_description: Single description like "Create a skill to fetch weather data"
            category: Skill category (trading_skills, system_skills, data_skills)
            name: Optional skill name (auto-generated if not provided)
            code: Optional Python code (auto-generated if not provided)

        Returns:
            Dict with success status, skill_path, and any errors
        """
        try:
            # Generate config from description
            config = self._generate_config_from_description(
                high_level_description, category, name, code
            )

            # Validate config
            if self.enable_validation:
                errors = self._validate_skill_config(config)
                if errors:
                    return {
                        "success": False,
                        "errors": errors,
                        "message": "Skill validation failed",
                    }

            # Create skill directory
            skill_path = self._create_skill_directory(config.category, config.name)

            # Generate SKILL.md
            self._generate_skill_md(skill_path, config)

            # Generate __init__.py
            self._generate_init_py(skill_path, config)

            # Auto-register in category __init__.py
            if self.enable_auto_registration:
                self._register_skill_in_category_init(config.category, config.name)

            self.logger.info(f"Successfully created skill: {config.name} at {skill_path}")

            return {
                "success": True,
                "skill_path": str(skill_path),
                "skill_name": config.name,
                "category": config.category,
                "message": f"Skill '{config.name}' created successfully",
            }

        except Exception as e:
            self.logger.error(f"Error creating skill: {e}")
            return {
                "success": False,
                "errors": [str(e)],
                "message": f"Failed to create skill: {e}",
            }

    def _generate_config_from_description(
        self,
        high_level_description: str,
        category: str,
        name: Optional[str] = None,
        code: Optional[str] = None,
    ) -> SkillGenerationConfig:
        """
        Generate SkillGenerationConfig from high-level description.

        This method infers skill name, code, input/output schemas, and other
        metadata from the description using pattern matching and heuristics.

        Args:
            high_level_description: User's high-level description
            category: Skill category
            name: Optional explicit skill name
            code: Optional explicit code

        Returns:
            SkillGenerationConfig with inferred/generated values
        """
        # Generate name from description if not provided
        if name is None:
            name = self._infer_name_from_description(high_level_description)

        # Generate description
        description = self._generate_description_from_input(high_level_description)

        # Generate code if not provided
        if code is None:
            code = self._generate_code_from_description(
                high_level_description, name
            )

        # Infer input/output schemas from code
        input_schema, output_schema = self._infer_schemas_from_code(code)

        # Generate example usage
        example_usage = self._generate_example_usage(name, input_schema)

        return SkillGenerationConfig(
            name=name,
            category=category,
            description=description,
            high_level_description=high_level_description,
            code=code,
            input_schema=input_schema,
            output_schema=output_schema,
            example_usage=example_usage,
            version="1.0.0",
        )

    def _infer_name_from_description(self, description: str) -> str:
        """
        Infer skill name from description.

        Examples:
            "Create a skill to fetch weather data" -> "fetch_weather_data"
            "Create RSI calculator" -> "calculate_rsi"
            "Logger for events" -> "log_event"
        """
        # Remove common prefixes
        desc = description.lower()
        for prefix in ["create a skill to ", "create ", "make a ", "build a ", "a ", "an "]:
            desc = desc.replace(prefix, "")

        # Extract key action and object
        words = desc.split()[:5]  # Take first 5 words

        # Convert to snake_case
        name_parts = []
        for word in words:
            # Remove punctuation
            word = re.sub(r"[^\w]", "", word)
            if word:
                name_parts.append(word)

        if not name_parts:
            return "generated_skill"

        return "_".join(name_parts)

    def _generate_description_from_input(self, description: str) -> str:
        """Generate a clean skill description from user input."""
        # Remove prefixes and capitalize
        desc = description.strip()
        for prefix in [
            "Create a skill to ",
            "Create ",
            "Make a ",
            "Build a ",
            "I want a skill to ",
            "I need ",
        ]:
            if desc.lower().startswith(prefix.lower()):
                desc = desc[len(prefix):]
                break

        # Capitalize first letter
        desc = desc[0].upper() + desc[1:] if desc else "Generated skill"

        # Add period if missing
        if not desc.endswith("."):
            desc += "."

        return desc

    def _generate_code_from_description(
        self, description: str, skill_name: str
    ) -> str:
        """Generate basic Python code from description."""
        # Infer function name from skill name
        func_name = skill_name.replace("skill_", "")

        # Generate docstring from description
        docstring = f'"""{self._generate_description_from_input(description)}"""'

        # Generate basic function structure
        code = f'''"""
{skill_name.replace("_", " ").title()} Skill

Auto-generated from: {description}
"""

from typing import Dict, Any


def {func_name}(**kwargs) -> Dict[str, Any]:
    {docstring}

    # TODO: Implement skill logic based on:
    #   {description}

    # Example implementation
    result = {{
        "status": "success",
        "message": "{skill_name} executed",
        "data": kwargs
    }}

    return result


__all__ = ["{func_name}"]
'''
        return code

    def _infer_schemas_from_code(
        self, code: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Infer input/output JSON schemas from code.

        This uses AST parsing to extract function signatures and return types.
        """
        try:
            tree = ast.parse(code)

            # Look for function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function arguments
                    args = node.args
                    input_properties = {}

                    # Skip 'self' argument
                    for arg in args.args[1:]:
                        arg_name = arg.arg
                        input_properties[arg_name] = {"type": "string"}

                    input_schema = {
                        "type": "object",
                        "properties": input_properties,
                        "required": list(input_properties.keys()) if input_properties else [],
                    }

                    # Output schema (default)
                    output_schema = {
                        "type": "object",
                        "properties": {
                            "result": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    }

                    return input_schema, output_schema

        except Exception as e:
            self.logger.warning(f"Failed to parse code for schema inference: {e}")

        # Fallback schemas
        return (
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        )

    def _generate_example_usage(
        self, skill_name: str, input_schema: Dict[str, Any]
    ) -> str:
        """Generate example usage string."""
        func_name = skill_name.replace("skill_", "")

        # Get required properties from input schema
        required = input_schema.get("required", [])
        if not required:
            args = ""
        else:
            args = ", ".join(f'{arg}="value"' for arg in required)

        return f'result = {func_name}({args})'

    def _validate_skill_config(
        self, config: SkillGenerationConfig
    ) -> List[str]:
        """
        Validate skill configuration before file system commit.

        Args:
            config: SkillGenerationConfig to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate category
        if config.category not in VALID_CATEGORIES:
            errors.append(f"Invalid category: {config.category}")

        # Validate name format
        if not re.match(r"^[a-z][a-z0-9_]*[a-z0-9]$", config.name):
            errors.append(
                f"Invalid name: {config.name}. Must be lowercase with underscores"
            )

        # Validate code is not empty
        if not config.code or not config.code.strip():
            errors.append("Code cannot be empty")

        # Validate version format
        if not re.match(r"^\d+\.\d+\.\d+$", config.version):
            errors.append(f"Invalid version: {config.version}")

        # Try to create SkillDefinition to validate full schema
        try:
            skill_def = SkillDefinition(
                name=config.name,
                category=config.category,  # type: ignore
                description=config.description,
                input_schema=config.input_schema,
                output_schema=config.output_schema,
                code=config.code,
                dependencies=config.dependencies,
                example_usage=config.example_usage or "",
                version=config.version,
            )

            # Use SkillValidator for additional checks
            validation_errors = SkillValidator.validate_skill(skill_def)
            errors.extend(validation_errors)

        except Exception as e:
            errors.append(f"SkillDefinition validation failed: {e}")

        return errors

    def _create_skill_directory(
        self, category: str, skill_name: str
    ) -> Path:
        """
        Create skill directory structure.

        Args:
            category: Skill category (trading_skills, system_skills, data_skills)
            skill_name: Name of the skill

        Returns:
            Path to created skill directory
        """
        # Validate category
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. Must be one of {VALID_CATEGORIES}"
            )

        # Create full path
        skill_path = self.skills_base_dir / category / skill_name

        # Create directory structure
        skill_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Created skill directory: {skill_path}")
        return skill_path

    def _generate_skill_md(
        self, skill_path: Path, config: SkillGenerationConfig
    ) -> None:
        """
        Generate SKILL.md with YAML frontmatter.

        Args:
            skill_path: Path to skill directory
            config: Skill generation configuration
        """
        skill_md_path = skill_path / "SKILL.md"

        # Generate YAML frontmatter
        yaml_frontmatter = f"""---
name: {config.name}
category: {config.category}
version: {config.version}
---

# {config.name.replace("_", " ").title()}

## Description

{config.description}

## High-Level Description

{config.high_level_description}

## Input Schema

```json
{self._format_json_schema(config.input_schema)}
```

## Output Schema

```json
{self._format_json_schema(config.output_schema)}
```

## Usage

```python
{config.example_usage or f"# Example usage for {config.name}"}
```

## Dependencies

{", ".join(config.dependencies) if config.dependencies else "None"}

## Implementation Notes

- Version: {config.version}
- Category: {config.category}
- Generated by: SkillCreator Meta-Skill
"""

        skill_md_path.write_text(yaml_frontmatter)
        self.logger.info(f"Generated SKILL.md at: {skill_md_path}")

    def _format_json_schema(self, schema: Dict[str, Any]) -> str:
        """Format JSON schema for markdown display."""
        import json
        return json.dumps(schema, indent=2)

    def _generate_init_py(
        self, skill_path: Path, config: SkillGenerationConfig
    ) -> None:
        """
        Generate __init__.py with SkillDefinition.

        Args:
            skill_path: Path to skill directory
            config: Skill generation configuration
        """
        init_path = skill_path / "__init__.py"

        # Generate SkillDefinition code
        init_content = f'''"""
{config.name.replace("_", " ").title()} Skill

{config.description}
"""

from typing import Dict, Any
from ..skill_schema import SkillDefinition


# Skill definition for {config.name}
skill_definition = SkillDefinition(
    name="{config.name}",
    category="{config.category}",
    description="{config.description}",
    input_schema={self._format_python_dict(config.input_schema)},
    output_schema={self._format_python_dict(config.output_schema)},
    """{self._escape_triple_quotes(config.code)}""",
    dependencies={config.dependencies},
    example_usage="{config.example_usage or f'result = {config.name}()'}",
    version="{config.version}",
)

# Export skill function
{config.code}

__all__ = ["skill_definition"]
'''

        init_path.write_text(init_content)
        self.logger.info(f"Generated __init__.py at: {init_path}")

    def _format_python_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary as Python literal."""
        return str(d)

    def _escape_triple_quotes(self, code: str) -> str:
        """Escape triple quotes in code for Python string."""
        return code.replace('"""', '\\"\\"\\"')

    def _register_skill_in_category_init(
        self, category: str, skill_name: str
    ) -> None:
        """
        Auto-register skill in category __init__.py.

        Args:
            category: Skill category
            skill_name: Name of skill to register
        """
        category_init_path = self.skills_base_dir / category / "__init__.py"

        # Create category init if it doesn't exist
        if not category_init_path.exists():
            category_init_path.parent.mkdir(parents=True, exist_ok=True)
            category_init_path.write_text(
                f'"""{category.replace("_", " ").title()} Module"""\n\n__all__ = []\n'
            )

        # Read current content
        content = category_init_path.read_text()

        # Check if already registered
        if f'from .{skill_name} import' in content or skill_name in content:
            self.logger.info(f"Skill {skill_name} already registered in {category}")
            return

        # Add import statement
        import_statement = f"from .{skill_name} import skill_definition"

        # Update __all__ list
        if "__all__" in content:
            # Find and update __all__ list
            all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
            if all_match:
                all_content = all_match.group(1)
                if all_content.strip():
                    # Add to existing list
                    new_all = f'__all__ = [{all_content}, "{skill_name}"]\n'
                else:
                    # First item in list
                    new_all = f'__all__ = ["{skill_name}"]\n'

                content = re.sub(
                    r"__all__\s*=\s*\[.*?\]",
                    new_all,
                    content,
                    count=1,
                    flags=re.DOTALL,
                )
            else:
                # Append __all__ at end
                content += f'\n__all__ = ["{skill_name}"]\n'
        else:
            # Add __all__ at end
            content += f'\n__all__ = ["{skill_name}"]\n'

        # Add import statement at the top after docstring
        lines = content.split("\n")
        insert_index = 0

        # Find end of docstring
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("#"):
                insert_index = i
                break

        # Insert import
        lines.insert(insert_index, import_statement)
        lines.insert(insert_index + 1, "")

        # Write back
        category_init_path.write_text("\n".join(lines))
        self.logger.info(f"Registered {skill_name} in {category}/__init__.py")


# Convenience function for quick skill creation
def create_skill(
    description: str,
    category: str = "system_skills",
    base_dir: str = "src/agents/skills",
) -> Dict[str, Any]:
    """
    Convenience function to create a skill from description.

    Args:
        description: High-level description like "Create a skill to fetch weather"
        category: Skill category (trading_skills, system_skills, data_skills)
        base_dir: Base skills directory

    Returns:
        Dict with success status and skill path
    """
    creator = SkillCreator(skills_base_dir=base_dir)
    return creator.create_skill_from_description(description, category)


__all__ = [
    "SkillCreator",
    "SkillGenerationConfig",
    "create_skill",
]
