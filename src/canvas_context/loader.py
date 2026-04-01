"""Canvas Context Template Loader.

Loads CanvasContextTemplate YAML files and assembles canvas context.
"""
import logging
from pathlib import Path

import yaml

from src.canvas_context.types import (
    CanvasContextTemplate,
    SkillIndexEntry,
)

logger = logging.getLogger(__name__)

# Template directory - relative to project root
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Allowed templates directory - absolute path for path traversal validation
ALLOWED_TEMPLATES_DIR = TEMPLATE_DIR.resolve()

# Known canvas identifiers
SUPPORTED_CANVASES = {
    "live_trading",
    "risk",
    "portfolio",
    "research",
    "development",
    "trading",
    "workshop",
    "flowforge",
    "shared_assets",
}

# Template cache
_template_cache: dict[str, CanvasContextTemplate] = {}


def normalize_canvas_name(canvas_name: str) -> str:
    """Normalize UI-facing canvas aliases to canonical template identifiers."""
    return canvas_name.strip().lower().replace("-", "_")


def _parse_skill_index(skill_list: list[dict]) -> list[SkillIndexEntry]:
    """Parse skill index from YAML list."""
    if not skill_list:
        return []
    return [SkillIndexEntry(**skill) for skill in skill_list]


def load_template(canvas_name: str, use_cache: bool = True) -> CanvasContextTemplate:
    """Load a CanvasContextTemplate for a specific canvas.

    Args:
        canvas_name: Canvas identifier (e.g., 'risk', 'live_trading')
        use_cache: Whether to use cached templates

    Returns:
        CanvasContextTemplate for the specified canvas

    Raises:
        FileNotFoundError: If template file doesn't exist
        ValueError: If canvas_name is invalid
    """
    normalized_canvas = normalize_canvas_name(canvas_name)

    if normalized_canvas not in SUPPORTED_CANVASES:
        raise ValueError(
            f"Unknown canvas: {canvas_name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_CANVASES))}"
        )

    # Check cache
    if use_cache and normalized_canvas in _template_cache:
        return _template_cache[normalized_canvas]

    # Load from YAML
    template_path = TEMPLATE_DIR / f"{normalized_canvas}.yaml"

    # Security: Validate resolved path is within allowed templates directory
    resolved_path = template_path.resolve()
    if not resolved_path.is_relative_to(ALLOWED_TEMPLATES_DIR):
        raise ValueError(
            f"Template path '{resolved_path}' is outside allowed directory. "
            f"Path traversal attempt detected."
        )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}. "
            f"Please create {canvas_name}.yaml in the templates directory."
        )

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse skill index
        if "skill_index" in data and data["skill_index"]:
            data["skill_index"] = _parse_skill_index(data["skill_index"])
        else:
            data["skill_index"] = []

        template = CanvasContextTemplate(**data)

        # Cache it
        _template_cache[normalized_canvas] = template

        logger.info(f"Loaded CanvasContextTemplate for: {normalized_canvas}")
        return template

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse template {template_path}: {e}")
        raise ValueError(f"Invalid YAML in {template_path}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load template {template_path}: {e}")
        raise


def get_all_templates() -> dict[str, CanvasContextTemplate]:
    """Load all available canvas templates.

    Returns:
        Dictionary mapping canvas name to template
    """
    templates = {}

    for canvas_name in SUPPORTED_CANVASES:
        try:
            templates[canvas_name] = load_template(canvas_name)
        except FileNotFoundError:
            logger.warning(f"Template not found for canvas: {canvas_name}")
        except Exception as e:
            logger.error(f"Failed to load template for {canvas_name}: {e}")

    return templates


def get_canvas_list() -> list[dict]:
    """Get list of available canvases with metadata.

    Returns:
        List of canvas metadata dictionaries
    """
    canvases = []

    for canvas_name in SUPPORTED_CANVASES:
        try:
            template = load_template(canvas_name)
            canvases.append({
                "id": canvas_name,
                "name": template.canvas_display_name or canvas_name,
                "icon": template.canvas_icon,
                "department_head": template.department_head,
            })
        except FileNotFoundError:
            pass

    return canvases


def clear_cache() -> None:
    """Clear the template cache."""
    global _template_cache
    _template_cache.clear()
    logger.info("Cleared CanvasContextTemplate cache")


def get_template_for_department(department: str) -> CanvasContextTemplate:
    """Get the canvas template for a department.

    Args:
        department: Department name (e.g., 'trading', 'risk')

    Returns:
        CanvasContextTemplate for the department

    Note:
        This maps department names to canvas names.
    """
    # Department to canvas mapping
    dept_to_canvas = {
        "trading": "trading",
        "risk": "risk",
        "portfolio": "portfolio",
        "research": "research",
        "development": "development",
    }

    canvas_name = normalize_canvas_name(
        dept_to_canvas.get(department.lower(), department.lower())
    )

    try:
        return load_template(canvas_name)
    except (FileNotFoundError, ValueError):
        # Fallback to workshop as default
        return load_template("workshop")
