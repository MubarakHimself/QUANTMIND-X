"""Prompt composition helpers for department agents and sub-agents."""

from .department_contracts import (
    compose_department_head_prompt,
    compose_floor_manager_prompt,
    compose_subagent_prompt,
    get_department_prompt_seed,
    get_floor_manager_prompt_seed,
)

__all__ = [
    "compose_department_head_prompt",
    "compose_floor_manager_prompt",
    "compose_subagent_prompt",
    "get_department_prompt_seed",
    "get_floor_manager_prompt_seed",
]
