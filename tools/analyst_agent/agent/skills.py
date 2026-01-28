"""
Skills system for QuantMindX agents.

A skill is a high-level capability that an agent can perform.
Skills can be composed of multiple tools.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class Skill:
    """A skill that an agent can perform."""

    name: str
    description: str
    category: str = "general"

    # Skill execution
    execute: Callable = None

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    requires_tools: List[str] = field(default_factory=list)
    requires_skills: List[str] = field(default_factory=list)

    # Metadata
    examples: List[str] = field(default_factory=list)
    version: str = "1.0"

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the skill."""
        if self.execute is None:
            raise NotImplementedError(f"Skill '{self.name}' has no execute function")
        return self.execute(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "requires_tools": self.requires_tools,
            "requires_skills": self.requires_skills,
            "examples": self.examples,
            "version": self.version
        }


class SkillRegistry:
    """Registry for managing agent skills."""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a new skill."""
        self.skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self.skills.get(name)

    def list(self, category: Optional[str] = None) -> List[Skill]:
        """List all skills, optionally filtered by category."""
        if category:
            return [s for s in self.skills.values() if s.category == category]
        return list(self.skills.values())

    def has(self, name: str) -> bool:
        """Check if skill exists."""
        return name in self.skills

    def remove(self, name: str) -> bool:
        """Remove a skill."""
        if name in self.skills:
            del self.skills[name]
            return True
        return False

    def categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(s.category for s in self.skills.values()))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert registry to dictionary."""
        return {name: skill.to_dict() for name, skill in self.skills.items()}


# Built-in skills

def create_search_skill(kb_client=None):
    """Create KB search skill."""
    def execute(query: str, n: int = 3):
        if not kb_client:
            return []
        return kb_client.search(query, collection="analyst_kb", n=n)

    return Skill(
        name="kb_search",
        description="Search the knowledge base for relevant articles",
        category="knowledge",
        execute=execute,
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "n": {"type": "integer", "default": 3, "description": "Number of results"}
        },
        examples=[
            "Search for: Kelly criterion position sizing",
            "Find articles about: ORB breakout strategy"
        ]
    )


def create_trd_generation_skill(generator_func=None):
    """Create TRD generation skill."""
    def execute(nprd_path: str, **kwargs):
        if not generator_func:
            raise NotImplementedError("No generator function provided")
        return generator_func(nprd_path, **kwargs)

    return Skill(
        name="generate_trd",
        description="Generate Technical Requirements Document from NPRD",
        category="generation",
        execute=execute,
        parameters={
            "nprd_path": {"type": "string", "description": "Path to NPRD file"},
            "output_dir": {"type": "string", "optional": True, "description": "Output directory"}
        },
        examples=[
            "Generate TRD from: data/nprds/strategy.json",
            "Convert NPRD to TRD: orb_strategy.json"
        ]
    )


def create_analysis_skill(kb_client=None):
    """Create strategy analysis skill."""
    def execute(trd_content: str):
        # Analyze TRD for completeness, issues, etc.
        return {
            "completeness": "good",
            "missing_sections": [],
            "recommendations": []
        }

    return Skill(
        name="analyze_strategy",
        description="Analyze a trading strategy for completeness and best practices",
        category="analysis",
        execute=execute,
        parameters={
            "trd_content": {"type": "string", "description": "TRD content to analyze"}
        }
    )


BUILTIN_SKILLS = {
    "kb_search": create_search_skill,
    "generate_trd": create_trd_generation_skill,
    "analyze_strategy": create_analysis_skill
}
