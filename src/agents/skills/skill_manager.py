"""
Skill Manager for QuantMind Agent SDK

Provides centralized skill registration, execution, and chaining capabilities.
Supports skill parameters, return values, skill dependencies, categories, and caching.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import asyncio
from functools import lru_cache
import hashlib
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SkillError(Exception):
    """Base exception for skill-related errors."""
    pass


class SkillNotFoundError(SkillError):
    """Raised when a requested skill is not registered."""
    pass


class SkillExecutionError(SkillError):
    """Raised when skill execution fails."""
    pass


class SkillValidationError(SkillError):
    """Raised when skill parameters fail validation."""
    pass


class ChainMode(Enum):
    """Defines how skill chain results are combined."""
    SEQUENTIAL = "sequential"  # Pass output of one as input to next
    PARALLEL = "parallel"     # Run all skills with same input
    FALLBACK = "fallback"     # Try skills in order until one succeeds


class SkillCategory(str, Enum):
    """Predefined skill categories for classification."""
    RESEARCH = "research"
    TRADING = "trading"
    RISK = "risk"
    CODING = "coding"
    DATA = "data"
    SYSTEM = "system"
    PORTFOLIO = "portfolio"
    ANALYSIS = "analysis"
    GENERAL = "general"


# Skill category metadata
SKILL_CATEGORY_METADATA = {
    SkillCategory.RESEARCH: {
        "description": "Research and knowledge-based skills",
        "color": "blue",
        "icon": "book"
    },
    SkillCategory.TRADING: {
        "description": "Trading and execution skills",
        "color": "green",
        "icon": "chart"
    },
    SkillCategory.RISK: {
        "description": "Risk management and calculation skills",
        "color": "red",
        "icon": "shield"
    },
    SkillCategory.CODING: {
        "description": "Code analysis and generation skills",
        "color": "purple",
        "icon": "code"
    },
    SkillCategory.DATA: {
        "description": "Data processing and analysis skills",
        "color": "orange",
        "icon": "database"
    },
    SkillCategory.SYSTEM: {
        "description": "System operations and monitoring skills",
        "color": "gray",
        "icon": "cog"
    },
    SkillCategory.PORTFOLIO: {
        "description": "Portfolio management skills",
        "color": "teal",
        "icon": "briefcase"
    },
    SkillCategory.ANALYSIS: {
        "description": "General analysis skills",
        "color": "cyan",
        "icon": "chart-bar"
    },
    SkillCategory.GENERAL: {
        "description": "General purpose skills",
        "color": "gray",
        "icon": "star"
    }
}


@dataclass
class SkillMetadata:
    """Metadata for a registered skill."""
    name: str
    description: str
    category: str
    departments: List[str] = field(default_factory=list)  # Departments this skill belongs to
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)  # Required skills
    tags: List[str] = field(default_factory=list)


@dataclass
class SkillResult:
    """Result from skill execution."""
    skill_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    chain_output: Optional[Dict[str, Any]] = None


class Skill(Generic[T]):
    """
    A callable skill with typed parameters and return values.

    Skills are registered functions that can be executed with parameters
    and support chaining (passing outputs as inputs to other skills).
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., T],
        metadata: SkillMetadata,
    ):
        self.name = name
        self.func = func
        self.metadata = metadata

    def __call__(self, **params: Any) -> T:
        """Execute the skill with given parameters."""
        return self.func(**params)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate that required parameters are provided."""
        required = self.metadata.parameters.get("required", [])
        for param in required:
            if param not in params:
                raise SkillValidationError(
                    f"Missing required parameter '{param}' for skill '{self.name}'"
                )
        return True


class SkillManager:
    """
    Central manager for skill registration and execution.

    Provides:
    - Skill registration and discovery
    - Parameter validation
    - Skill execution with timing
    - Skill chaining (sequential, parallel, fallback)
    - Skill caching for performance
    - Async execution support
    - Category management
    """

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 300):
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[str, List[str]] = {}
        self._departments: Dict[str, List[str]] = {}
        self._skill_aliases: Dict[str, str] = {}
        self._execution_history: List[SkillResult] = []
        self._max_history = 1000
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl  # Cache TTL in seconds
        self._cache: Dict[str, tuple] = {}  # Cache: key -> (result, timestamp)
        self._category_metadata = SKILL_CATEGORY_METADATA.copy()

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        category: str = "general",
        departments: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        requires: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """
        Register a new skill.

        Args:
            name: Unique skill identifier
            func: Callable function implementing the skill
            description: Human-readable description
            category: Skill category (e.g., 'trading', 'research')
            departments: List of departments this skill belongs to
            parameters: Parameter schema
            returns: Return value schema
            requires: List of required skill names
            tags: List of tags for discovery

        Returns:
            Registered Skill instance
        """
        if name in self._skills:
            logger.warning(f"Overwriting existing skill: {name}")

        metadata = SkillMetadata(
            name=name,
            description=description,
            category=category,
            departments=departments or [],
            parameters=parameters or {},
            returns=returns or {},
            requires=requires or [],
            tags=tags or [],
        )

        skill = Skill(name=name, func=func, metadata=metadata)
        self._skills[name] = skill

        # Track by category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Track by department
        for dept in departments or []:
            if dept not in self._departments:
                self._departments[dept] = []
            if name not in self._departments[dept]:
                self._departments[dept].append(name)

        logger.info(f"Registered skill: {name} (category: {category}, departments: {departments})")
        return skill

    def register_alias(self, alias: str, skill_name: str) -> None:
        """Register an alias for an existing skill."""
        if skill_name not in self._skills:
            raise SkillNotFoundError(f"Cannot create alias: skill '{skill_name}' not found")
        self._skill_aliases[alias] = skill_name

    def get_skill(self, name: str) -> Skill:
        """Get a skill by name or alias."""
        resolved = self._skill_aliases.get(name, name)
        if resolved not in self._skills:
            raise SkillNotFoundError(f"Skill '{name}' not found")
        return self._skills[resolved]

    def list_skills(
        self,
        category: Optional[str] = None,
        department: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """List registered skill names, optionally filtered by category, department, or tags."""
        skills = list(self._skills.keys())

        if category:
            category_skills = self._categories.get(category, [])
            skills = [s for s in skills if s in category_skills]

        if department:
            skills = [
                s for s in skills
                if department in self._skills[s].metadata.departments
            ]

        if tags:
            skills = [
                s for s in skills
                if any(tag in self._skills[s].metadata.tags for tag in tags)
            ]

        return skills

    def get_skills_by_department(self, department: str) -> List[str]:
        """
        Get all skills available to a specific department.

        Args:
            department: Department name (e.g., 'research', 'trading')

        Returns:
            List of skill names available to the department
        """
        return self.list_skills(department=department)

    def get_skill_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a skill."""
        skill = self.get_skill(name)
        return {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "category": skill.metadata.category,
            "departments": skill.metadata.departments,
            "parameters": skill.metadata.parameters,
            "returns": skill.metadata.returns,
            "requires": skill.metadata.requires,
            "tags": skill.metadata.tags,
        }

    def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """
        Execute a single skill with parameters.

        Args:
            name: Skill name or alias
            params: Input parameters for the skill
            context: Optional execution context

        Returns:
            SkillResult with execution data
        """
        import time
        start_time = time.time()

        params = params or {}
        context = context or {}

        try:
            skill = self.get_skill(name)
            skill.validate_params(params)

            # Check dependencies
            for required_skill in skill.metadata.requires:
                if required_skill not in self._skills:
                    raise SkillExecutionError(
                        f"Required skill '{required_skill}' not registered"
                    )

            # Execute with context merged into params
            full_params = {**context, **params}
            result_data = skill(**full_params)

            execution_time = (time.time() - start_time) * 1000

            result = SkillResult(
                skill_name=name,
                success=True,
                data=result_data,
                execution_time_ms=execution_time,
            )

            logger.info(f"Skill '{name}' executed successfully in {execution_time:.2f}ms")

        except (SkillNotFoundError, SkillValidationError) as e:
            execution_time = (time.time() - start_time) * 1000
            result = SkillResult(
                skill_name=name,
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time,
            )
            logger.error(f"Skill '{name}' validation failed: {e}")

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = SkillResult(
                skill_name=name,
                success=False,
                data=None,
                error=f"Execution error: {str(e)}",
                execution_time_ms=execution_time,
            )
            logger.exception(f"Skill '{name}' execution failed")

        # Add to history
        self._execution_history.append(result)
        if len(self._execution_history) > self._max_history:
            self._execution_history.pop(0)

        return result

    # =========================================================================
    # Caching Support
    # =========================================================================

    def _get_cache_key(self, skill_name: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for skill execution."""
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{skill_name}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if not self._enable_cache or cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]
        if time.time() - timestamp < self._cache_ttl:
            return result
        else:
            # Cache expired
            del self._cache[cache_key]
            return None

    def _set_cached(self, cache_key: str, result: Any) -> None:
        """Cache a skill execution result."""
        if self._enable_cache:
            self._cache[cache_key] = (result, time.time())

    def clear_cache(self, skill_name: Optional[str] = None) -> int:
        """
        Clear cached results.

        Args:
            skill_name: If provided, only clear cache for this skill

        Returns:
            Number of cache entries cleared
        """
        if skill_name:
            keys_to_remove = [
                k for k in self._cache.keys()
                if k.startswith(skill_name)
            ]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
        else:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self._enable_cache,
            "ttl_seconds": self._cache_ttl,
            "entries": len(self._cache),
            "cache_keys": list(self._cache.keys())[:10]  # Show first 10
        }

    # =========================================================================
    # Async Execution Support
    # =========================================================================

    async def execute_async(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """
        Execute a skill asynchronously.

        Args:
            name: Skill name or alias
            params: Input parameters for the skill
            context: Optional execution context

        Returns:
            SkillResult with execution data
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.execute,
            name,
            params,
            context
        )

    async def chain_async(
        self,
        skills: List[Dict[str, Any]],
        mode: ChainMode = ChainMode.SEQUENTIAL,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> List[SkillResult]:
        """
        Execute multiple skills in chain asynchronously.

        Args:
            skills: List of skill execution specs
            mode: ChainMode (sequential, parallel, fallback)
            initial_params: Initial parameters for the chain

        Returns:
            List of SkillResult for each skill execution
        """
        if mode == ChainMode.PARALLEL:
            # Execute all skills concurrently
            tasks = [
                self.execute_async(
                    name=spec.get("name", ""),
                    params=spec.get("params", {}),
                    context=initial_params or {}
                )
                for spec in skills
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(SkillResult(
                        skill_name=skills[i].get("name", ""),
                        success=False,
                        data=None,
                        error=str(result),
                        execution_time_ms=0.0
                    ))
                else:
                    final_results.append(result)
            return final_results
        else:
            # Sequential or fallback - use regular chain
            return self.chain(skills, mode, initial_params)

    # =========================================================================
    # Category Support
    # =========================================================================

    def get_category_metadata(self, category: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a skill category."""
        return self._category_metadata.get(category)

    def list_categories(self) -> List[str]:
        """List all registered skill categories."""
        return list(self._categories.keys())

    def get_skills_by_category(self, category: str) -> List[str]:
        """Get all skills in a specific category."""
        return self._categories.get(category, [])

    def set_category_metadata(self, category: str, metadata: Dict[str, Any]) -> None:
        """Set metadata for a category."""
        self._category_metadata[category] = metadata

    def chain(
        self,
        skills: List[Dict[str, Any]],
        mode: ChainMode = ChainMode.SEQUENTIAL,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> List[SkillResult]:
        """
        Execute multiple skills in chain.

        Args:
            skills: List of skill execution specs
                Each: {"name": str, "params": dict, "use_output_as": str}
            mode: ChainMode (sequential, parallel, fallback)
            initial_params: Initial parameters for the chain

        Returns:
            List of SkillResult for each skill execution
        """
        results: List[SkillResult] = []
        chain_context = initial_params or {}

        if mode == ChainMode.PARALLEL:
            # Execute all skills with same initial params
            for spec in skills:
                result = self.execute(
                    name=spec.get("name", ""),
                    params=spec.get("params", {}),
                    context=chain_context,
                )
                results.append(result)

        elif mode == ChainMode.FALLBACK:
            # Try skills in order until one succeeds
            for spec in skills:
                result = self.execute(
                    name=spec.get("name", ""),
                    params=spec.get("params", {}),
                    context=chain_context,
                )
                results.append(result)
                if result.success:
                    break

        else:  # SEQUENTIAL
            # Pass output of one as input to next
            for spec in skills:
                name = spec.get("name", "")
                params = spec.get("params", {})

                # Apply output mapping if specified
                use_output_as = spec.get("use_output_as")
                if use_output_as and results:
                    last_result = results[-1]
                    if last_result.success and last_result.data:
                        params[use_output_as] = last_result.data

                result = self.execute(name=name, params=params, context=chain_context)
                results.append(result)

                # Stop chain on failure (optional - can be configured)
                if not result.success:
                    logger.warning(f"Chain stopped at '{name}' due to failure")
                    break

        return results

    def get_execution_history(
        self,
        skill_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[SkillResult]:
        """Get execution history, optionally filtered by skill name."""
        history = self._execution_history

        if skill_name:
            history = [r for r in history if r.skill_name == skill_name]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
            }

        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)
        total_time = sum(r.execution_time_ms for r in self._execution_history)

        return {
            "total_executions": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_execution_time_ms": total_time / total if total > 0 else 0.0,
            "registered_skills": len(self._skills),
            "categories": list(self._categories.keys()),
            "departments": list(self._departments.keys()),
        }


# Global skill manager instance
_global_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get the global SkillManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SkillManager()
    return _global_manager


def set_skill_manager(manager: SkillManager) -> None:
    """Set the global SkillManager instance."""
    global _global_manager
    _global_manager = manager
