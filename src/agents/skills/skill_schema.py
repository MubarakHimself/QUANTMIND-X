"""
Skill Schema and Definitions for QuantMind Agents.

This module defines the schema for skill definitions following the myCodingAgents.com
and QuantMindX Architecture specification (lines 474-486).

SkillDefinition Schema:
- name: Unique skill identifier
- category: trading_skills | system_skills | data_skills
- description: Human-readable description
- input_schema: JSON Schema for input validation
- output_schema: JSON Schema for output validation
- code: Python/MQL5 code implementing the skill
- dependencies: List of required skill names
- example_usage: Example demonstrating how to use the skill
- version: Semantic version (e.g., "1.0.0")
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator
import json
import re


class SkillDefinition(BaseModel):
    """
    Schema for skill definition.

    Following spec lines 474-486 from myCodingAgents.com & QuantMindX Architecture.
    """

    name: str = Field(..., description="Unique skill identifier")
    category: Literal["trading_skills", "system_skills", "data_skills"] = Field(
        ..., description="Skill category for classification"
    )
    description: str = Field(..., description="Human-readable skill description")
    input_schema: Dict[str, Any] = Field(
        ..., description="JSON Schema for input validation"
    )
    output_schema: Dict[str, Any] = Field(
        ..., description="JSON Schema for output validation"
    )
    code: str = Field(..., description="Python or MQL5 code implementing the skill")
    dependencies: List[str] = Field(
        default_factory=list, description="List of required skill names"
    )
    example_usage: str = Field(..., description="Example demonstrating skill usage")
    version: str = Field(..., description="Semantic version (e.g., '1.0.0')")

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

    @validator("input_schema", "output_schema")
    def validate_json_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that schemas are valid JSON Schema."""
        if not isinstance(v, dict):
            raise ValueError("Schema must be a dictionary")

        # Basic JSON Schema validation - must have type
        if "type" not in v and "$schema" not in v:
            # Allow schema reference or inline type
            if "properties" not in v and "items" not in v:
                raise ValueError(
                    "JSON Schema must define 'type', 'properties', or 'items'"
                )

        return v

    def conforms_to_agent_skill_interface(self) -> bool:
        """
        Check if this skill definition conforms to the AgentSkill interface.

        The AgentSkill interface (from base.py) requires:
        - name: str
        - description: str
        - tools: List[BaseTool]
        - system_prompt_addition: str

        Since SkillDefinition is a data model for storage/retrieval from ChromaDB,
        we check that it has the minimal fields needed to create an AgentSkill.
        """
        return bool(
            self.name
            and self.description
            and self.category
            and self.code
        )

    def to_agent_skill_dict(self) -> Dict[str, Any]:
        """
        Convert SkillDefinition to dictionary compatible with AgentSkill.

        Returns a dictionary with fields that can be used to create an AgentSkill.
        The code field can be used to dynamically create tools.
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "code": self.code,
            "dependencies": self.dependencies,
            "example_usage": self.example_usage,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class SkillDependencyResolver:
    """
    Resolves skill dependencies recursively.

    Ensures that all required skills are available and resolves dependency order.
    """

    def __init__(self, skills: Dict[str, SkillDefinition]):
        """
        Initialize resolver with available skills.

        Args:
            skills: Dictionary mapping skill names to SkillDefinitions
        """
        self.skills = skills
        self._resolved: List[str] = []
        self._resolving: set = set()

    def resolve(self, skill_name: str) -> List[SkillDefinition]:
        """
        Resolve skill and its dependencies in dependency order.

        Args:
            skill_name: Name of the skill to resolve

        Returns:
            List of SkillDefinitions in dependency order (dependencies first)

        Raises:
            ValueError: If circular dependency detected or skill not found
        """
        if skill_name in self._resolving:
            raise ValueError(f"Circular dependency detected: {skill_name}")

        if skill_name in self._resolved:
            return []

        if skill_name not in self.skills:
            raise ValueError(f"Skill not found: {skill_name}")

        self._resolving.add(skill_name)
        skill = self.skills[skill_name]

        resolved_deps: List[SkillDefinition] = []
        for dep in skill.dependencies:
            resolved_deps.extend(self.resolve(dep))

        self._resolving.remove(skill_name)
        self._resolved.append(skill_name)
        resolved_deps.append(skill)

        return resolved_deps

    def get_load_order(self, skill_names: List[str]) -> List[str]:
        """
        Get the correct load order for multiple skills.

        Args:
            skill_names: List of skill names to load

        Returns:
            List of skill names in dependency order
        """
        result: List[str] = []
        self._resolved = []
        self._resolving = set()

        for name in skill_names:
            if name not in self._resolved:
                deps = self.resolve(name)
                for dep in deps:
                    if dep.name not in result:
                        result.append(dep.name)

        return result


class SkillValidator:
    """
    Validates skill definitions against schema and interface requirements.

    Ensures skills conform to JSON Schema specifications and AgentSkill interface.
    """

    @staticmethod
    def validate_input(schema: Dict[str, Any], data: Any) -> bool:
        """
        Validate input data against JSON Schema.

        Args:
            schema: JSON Schema for validation
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic JSON Schema type validation
            if "type" in schema:
                expected_type = schema["type"]
                type_map = {
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }

                if expected_type in type_map:
                    if not isinstance(data, type_map[expected_type]):
                        return False

            # Validate properties and their types
            if "properties" in schema and isinstance(data, dict):
                for prop, prop_schema in schema["properties"].items():
                    if prop in data:
                        prop_type = prop_schema.get("type")
                        if prop_type:
                            type_map = {
                                "string": str,
                                "number": (int, float),
                                "integer": int,
                                "boolean": bool,
                                "array": list,
                                "object": dict,
                            }
                            if prop_type in type_map:
                                if not isinstance(data[prop], type_map[prop_type]):
                                    return False

            # Validate required properties
            if "required" in schema and isinstance(data, dict):
                for prop in schema["required"]:
                    if prop not in data:
                        return False

            return True

        except Exception:
            return False

    @staticmethod
    def validate_output(schema: Dict[str, Any], data: Any) -> bool:
        """
        Validate output data against JSON Schema.

        Args:
            schema: JSON Schema for validation
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        return SkillValidator.validate_input(schema, data)

    @staticmethod
    def validate_skill(skill: SkillDefinition) -> List[str]:
        """
        Validate skill definition and return list of issues.

        Args:
            skill: SkillDefinition to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []

        # Check AgentSkill interface conformance
        if not skill.conforms_to_agent_skill_interface():
            errors.append(
                "Skill does not conform to AgentSkill interface"
            )

        # Validate JSON Schema format
        try:
            json.dumps(skill.input_schema)
        except Exception as e:
            errors.append(f"Invalid input_schema JSON: {e}")

        try:
            json.dumps(skill.output_schema)
        except Exception as e:
            errors.append(f"Invalid output_schema JSON: {e}")

        # Check code is not empty
        if not skill.code or not skill.code.strip():
            errors.append("Code field cannot be empty")

        # Check example_usage is not empty
        if not skill.example_usage or not skill.example_usage.strip():
            errors.append("example_usage field cannot be empty")

        return errors


# Skill registry for version management
class SkillRegistry:
    """
    Registry for managing skill versions and retrieval.

    Supports multiple versions of the same skill with latest version lookup.
    """

    def __init__(self):
        """Initialize empty skill registry."""
        self._skills: Dict[str, Dict[str, SkillDefinition]] = {}

    def register(self, skill: SkillDefinition) -> None:
        """
        Register a skill in the registry.

        Args:
            skill: SkillDefinition to register
        """
        if skill.name not in self._skills:
            self._skills[skill.name] = {}

        self._skills[skill.name][skill.version] = skill

    def get(
        self, name: str, version: Optional[str] = None
    ) -> Optional[SkillDefinition]:
        """
        Get a skill by name and optional version.

        Args:
            name: Skill name
            version: Optional version (defaults to latest)

        Returns:
            SkillDefinition or None if not found
        """
        if name not in self._skills:
            return None

        if version is None:
            # Return latest version (sort by version string)
            versions = sorted(
                self._skills[name].keys(),
                key=lambda v: [int(x) for x in v.split(".")],
                reverse=True,
            )
            return self._skills[name][versions[0]] if versions else None

        return self._skills[name].get(version)

    def list_versions(self, name: str) -> List[str]:
        """
        List all versions of a skill.

        Args:
            name: Skill name

        Returns:
            List of version strings (sorted newest first)
        """
        if name not in self._skills:
            return []

        return sorted(
            self._skills[name].keys(),
            key=lambda v: [int(x) for x in v.split(".")],
            reverse=True,
        )

    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered skill names.

        Args:
            category: Optional category filter

        Returns:
            List of skill names
        """
        if category is None:
            return list(self._skills.keys())

        return [
            name
            for name, versions in self._skills.items()
            if any(
                skill.category == category
                for skill in versions.values()
            )
        ]


# Predefined category enum for type safety
SkillCategory = Literal["trading_skills", "system_skills", "data_skills"]


__all__ = [
    "SkillDefinition",
    "SkillDependencyResolver",
    "SkillValidator",
    "SkillRegistry",
    "SkillCategory",
]
