"""Canvas Context Template types and models.

Pydantic models for CanvasContextTemplate per department.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SkillIndexEntry(BaseModel):
    """A skill entry in the skill index."""

    id: str = Field(..., description="Unique identifier for the skill")
    path: str = Field(..., description="Path to the skill definition file")
    trigger: str = Field(..., description="Condition when this skill should be triggered")


class CanvasSuggestionChip(BaseModel):
    """A suggestion chip for canvas navigation."""

    id: str
    label: str
    target_canvas: str
    target_entity: Optional[str] = None
    icon: Optional[str] = None


class CanvasContextTemplate(BaseModel):
    """Canvas Context Template for a specific canvas/department.

    This template provides the base context for agents when working on a specific canvas.
    It includes memory scope, workflow namespaces, department mailbox, shared assets,
    and skill index - all as identifiers (not content) for JIT loading.
    """

    canvas: str = Field(..., description="Canvas identifier (e.g., 'risk', 'live_trading')")

    base_descriptor: str = Field(
        ...,
        description="Base system prompt for the agent on this canvas"
    )

    memory_scope: list[str] = Field(
        default_factory=list,
        description="Graph memory namespaces to load (e.g., ['risk.*', 'portfolio.*'])"
    )

    workflow_namespaces: list[str] = Field(
        default_factory=list,
        description="Workflow namespaces available on this canvas"
    )

    department_mailbox: Optional[str] = Field(
        default=None,
        description="Department mailbox stream name"
    )

    shared_assets: list[str] = Field(
        default_factory=list,
        description="Shared asset categories available on this canvas"
    )

    skill_index: list[SkillIndexEntry] = Field(
        default_factory=list,
        description="Indexed skills for this canvas"
    )

    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools required for this canvas"
    )

    suggestion_chips: list[CanvasSuggestionChip] = Field(
        default_factory=list,
        description="Suggestion chips for quick navigation and actions on this canvas"
    )

    # Token budget configuration
    max_identifiers: int = Field(
        default=50,
        description="Maximum number of memory identifiers to pre-load"
    )

    # Canvas metadata
    canvas_display_name: Optional[str] = Field(
        default=None,
        description="Human-readable display name for the canvas"
    )

    canvas_icon: Optional[str] = Field(
        default=None,
        description="Icon name for the canvas"
    )

    # Department configuration
    department_head: Optional[str] = Field(
        default=None,
        description="Department head agent for this canvas"
    )

    model_config = {
        "extra": "forbid"
    }


class CanvasContextState(BaseModel):
    """Runtime state for canvas context.

    This is assembled when a user opens a chat on a canvas and includes:
    - The template (from YAML)
    - Memory identifiers (from graph, committed only)
    - Current session info
    """

    template: CanvasContextTemplate
    memory_identifiers: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    loaded_at: Optional[str] = None

    # JIT fetch tracking
    fetched_content: dict[str, str] = Field(default_factory=dict)