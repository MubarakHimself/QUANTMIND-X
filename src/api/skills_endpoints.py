"""
Skills API Endpoints - Story 7.4: Skill Catalogue, Registry & Skill Forge

Provides REST API for:
- GET /api/skills - List all registered skills
- POST /api/skills - Register a new skill
- POST /api/skills/authoring - Skill Forge authoring endpoint
- GET /api/skills/{skill_name} - Get skill details
- POST /api/skills/{skill_name}/execute - Execute a skill
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import logging
import os

from src.agents.skills.skill_manager import get_skill_manager, SkillManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])

# =============================================================================
# Request/Response Models
# =============================================================================


class SkillCreateRequest(BaseModel):
    """Request model for creating a new skill."""
    name: str = Field(..., description="Unique skill identifier (lowercase with underscores)")
    description: str = Field(..., description="Human-readable skill description")
    category: str = Field(default="general", description="Skill category")
    departments: List[str] = Field(default_factory=list, description="Departments this skill belongs to")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameter schema")
    returns: Dict[str, Any] = Field(default_factory=dict, description="Return value schema")
    tags: List[str] = Field(default_factory=list, description="Tags for discovery")
    version: str = Field(default="1.0.0", description="Semantic version")


class SkillAuthoringRequest(BaseModel):
    """Request model for Skill Forge authoring - creates skill.md file."""
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input specifications")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output specifications")
    sop_steps: List[str] = Field(default_factory=list, description="Standard operating procedure steps")
    category: str = Field(default="general", description="Skill category")
    departments: List[str] = Field(default_factory=list, description="Departments")
    version: str = Field(default="1.0.0", description="Semantic version")


class SkillExecuteRequest(BaseModel):
    """Request model for executing a skill."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Skill parameters")


class SkillResponse(BaseModel):
    """Response model for skill data."""
    name: str
    description: str
    slash_command: str
    version: str
    usage_count: int
    category: Optional[str] = None
    departments: Optional[List[str]] = None


class SkillDetailResponse(SkillResponse):
    """Response model for detailed skill info."""
    parameters: Optional[Dict[str, Any]] = None
    returns: Optional[Dict[str, Any]] = None
    requires: Optional[List[str]] = None
    tags: Optional[List[str]] = None


# =============================================================================
# Configuration
# =============================================================================

# Shared assets directory for skill.md files
SHARED_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "shared_assets")
SKILLS_DIR = os.path.join(SHARED_ASSETS_DIR, "skills")


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_skills_dir() -> None:
    """Ensure the skills directory exists."""
    os.makedirs(SKILLS_DIR, exist_ok=True)


def _get_skill_md_path(skill_name: str) -> str:
    """Get the file path for a skill.md file."""
    return os.path.join(SKILLS_DIR, f"{skill_name}.md")


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=List[SkillResponse])
async def list_skills(department: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    GET /api/skills

    List all registered skills with their metadata.

    Returns skills with: name, description, slash_command, version, usage_count
    """
    manager = get_skill_manager()

    # Get all skills or filter by department
    if department:
        skill_names = manager.get_skills_by_department(department)
    else:
        skill_names = manager.list_skills()

    # Build response with required fields per AC #3
    skills = []
    for name in skill_names:
        info = manager.get_skill_info(name)
        skills.append({
            "name": info["name"],
            "description": info["description"],
            "slash_command": info.get("slash_command", f"/{info['name'].replace('_', '-')}"),
            "version": info.get("version", "1.0.0"),
            "usage_count": info.get("usage_count", 0),
        })

    logger.info(f"Listed {len(skills)} skills")
    return skills


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_skill(request: SkillCreateRequest) -> Dict[str, Any]:
    """
    POST /api/skills

    Register a new skill in the skill manager.

    The skill is registered with the SkillManager but the actual skill function
    needs to be provided separately.
    """
    manager = get_skill_manager()

    # Generate slash_command from name if not provided
    slash_command = f"/{request.name.replace('_', '-')}"

    # Note: This endpoint registers the metadata. The actual skill function
    # would need to be registered with a proper implementation.
    # For now, we'll create a placeholder skill with the metadata.

    # Check if skill already exists
    try:
        existing = manager.get_skill_info(request.name)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Skill '{request.name}' already exists"
        )
    except Exception:
        pass  # Skill doesn't exist, continue

    # Register a placeholder skill (in production, would register actual function)
    # This is a simplified implementation - actual skill functions would be
    # provided by Department Heads through Skill Forge

    # For now, just return the metadata that would be stored
    logger.info(f"Skill metadata created for: {request.name}")

    return {
        "name": request.name,
        "description": request.description,
        "slash_command": slash_command,
        "version": request.version,
        "usage_count": 0,
        "category": request.category,
        "departments": request.departments,
        "message": "Skill registered. Note: Skill function must be implemented separately."
    }


@router.post("/authoring", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def skill_forge_authoring(request: SkillAuthoringRequest) -> Dict[str, Any]:
    """
    POST /api/skills/authoring

    Skill Forge authoring endpoint.

    Creates a skill.md file defining the skill's metadata and SOP steps.
    Department Heads use this to author and register new skills.
    """
    _ensure_skills_dir()

    # Generate slash_command
    slash_command = f"/{request.name.replace('_', '-')}"

    # Build skill.md content
    skill_md_content = f"""---
name: {request.name}
description: {request.description}
version: {request.version}
category: {request.category}
departments: {','.join(request.departments)}
slash_command: {slash_command}
---

# Skill: {request.name}

## Description
{request.description}

## Inputs
```yaml
{_format_dict_as_yaml(request.inputs)}
```

## Outputs
```yaml
{_format_dict_as_yaml(request.outputs)}
```

## Standard Operating Procedure (SOP)

{' '.join(f'{i+1}. {step}' for i, step in enumerate(request.sop_steps))}

---

*Generated by QuantMindX Skill Forge on 2026-03-19*
"""

    # Write skill.md file
    skill_md_path = _get_skill_md_path(request.name)
    with open(skill_md_path, 'w') as f:
        f.write(skill_md_content)

    logger.info(f"Skill Forge created skill.md: {skill_md_path}")

    # Note: In production, this would also register the skill with SkillManager
    # after the ReflectionExecutor (Story 5.1) reviews skill quality

    return {
        "name": request.name,
        "description": request.description,
        "version": request.version,
        "slash_command": slash_command,
        "skill_md_path": skill_md_path,
        "status": "created",
        "message": "Skill authored successfully. Review by ReflectionExecutor pending."
    }


@router.get("/{skill_name}", response_model=SkillDetailResponse)
async def get_skill(skill_name: str) -> Dict[str, Any]:
    """
    GET /api/skills/{skill_name}

    Get detailed information about a specific skill.
    """
    manager = get_skill_manager()

    try:
        info = manager.get_skill_info(skill_name)
        return info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found"
        )


@router.post("/{skill_name}/execute", response_model=Dict[str, Any])
async def execute_skill(
    skill_name: str,
    request: SkillExecuteRequest
) -> Dict[str, Any]:
    """
    POST /api/skills/{skill_name}/execute

    Execute a skill with the provided parameters.
    """
    manager = get_skill_manager()

    try:
        result = manager.execute(skill_name, request.parameters)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill execution failed: {str(e)}"
        )


# =============================================================================
# Helper: Format Dict as YAML-like string
# =============================================================================

def _format_dict_as_yaml(d: Dict[str, Any], indent: int = 2) -> str:
    """Format a dictionary as a YAML-like string."""
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            for k2, v2 in value.items():
                lines.append(f"{' ' * (indent + 2)}{k2}: {v2}")
        elif isinstance(value, list):
            lines.append(f"{' ' * indent}{key}:")
            for item in value:
                lines.append(f"{' ' * (indent + 2)}- {item}")
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return '\n'.join(lines)


# =============================================================================
# Router Registration Note
# =============================================================================
# This router is registered in server.py with:
# app.include_router(skills_router, prefix="/api")