"""
Input validation utilities for the analyst agent.

This module provides validation functions for NPRD structure and TRD content.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_nprd_structure(data: dict[str, Any]) -> bool:
    """
    Validate NPRD (Natural Language Product Requirements Document) structure.

    Args:
        data: NPRD data dictionary to validate.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValueError: If critical validation errors are found.
    """
    errors = []

    # Check if data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("NPRD data must be a dictionary")

    # Required top-level fields
    required_fields = ["title", "content"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")

    # Validate title
    if "title" in data:
        if not isinstance(data["title"], str):
            errors.append("Title must be a string")
        elif not data["title"].strip():
            errors.append("Title cannot be empty")

    # Validate content
    if "content" in data:
        if not isinstance(data["content"], str):
            errors.append("Content must be a string")
        elif not data["content"].strip():
            errors.append("Content cannot be empty")

    # Optional but recommended fields
    recommended_fields = ["id", "version", "author", "created"]
    missing_recommended = [field for field in recommended_fields if field not in data]

    if missing_recommended:
        logger.warning(f"Missing recommended fields: {missing_recommended}")

    # Validate metadata if present
    if "metadata" in data:
        if not isinstance(data["metadata"], dict):
            errors.append("Metadata must be a dictionary")

    # Validate requirements if present
    if "requirements" in data:
        if not isinstance(data["requirements"], list):
            errors.append("Requirements must be a list")
        else:
            for i, req in enumerate(data["requirements"]):
                if not isinstance(req, (str, dict)):
                    errors.append(f"Requirement {i} must be a string or dictionary")

    # Log validation results
    if errors:
        for error in errors:
            logger.error(f"NPRD validation error: {error}")
        return False

    logger.info("NPRD structure validation passed")
    return True


def validate_trd_content(content: str) -> bool:
    """
    Validate TRD (Technical Requirements Document) content.

    Args:
        content: TRD content string to validate.

    Returns:
        True if valid, False otherwise.

    Raises:
        ValueError: If critical validation errors are found.
    """
    if not isinstance(content, str):
        raise ValueError("TRD content must be a string")

    if not content.strip():
        raise ValueError("TRD content cannot be empty")

    errors = []

    # Check for frontmatter delimiters
    if "---" not in content:
        errors.append("TRD must have YAML frontmatter delimited by ---")
    else:
        # Extract frontmatter section
        parts = content.split("---", 2)
        if len(parts) < 3:
            errors.append("TRD frontmatter must have opening and closing --- delimiters")
        else:
            frontmatter = parts[1].strip()
            if not frontmatter:
                errors.append("TRD frontmatter cannot be empty")

            # Check for required frontmatter fields
            required_fm_fields = ["id", "title", "version"]
            for field in required_fm_fields:
                if f"{field}:" not in frontmatter and f'{field} =' not in frontmatter:
                    errors.append(f"TRD frontmatter missing required field: {field}")

    # Check for content sections
    body = content.split("---", 2)[-1] if "---" in content else content

    # Check for main heading (##)
    if "##" not in body:
        errors.append("TRD should have at least one section heading (##)")

    # Check for executive summary
    if "## Executive Summary" not in body and "## Summary" not in body:
        logger.warning("TRD missing Executive Summary section")

    # Check for requirements section
    if "## Technical Requirements" not in body and "## Requirements" not in body:
        logger.warning("TRD missing Technical Requirements section")

    # Check minimum content length
    if len(content.strip()) < 100:
        errors.append("TRD content seems too short (< 100 characters)")

    # Log validation results
    if errors:
        for error in errors:
            logger.error(f"TRD validation error: {error}")
        return False

    logger.info("TRD content validation passed")
    return True


def validate_path_exists(path: Path | str, must_be_file: bool = False) -> bool:
    """
    Validate that a path exists.

    Args:
        path: Path to validate.
        must_be_file: If True, path must be a file (not a directory).

    Returns:
        True if valid, False otherwise.
    """
    path = Path(path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return False

    if must_be_file and not path.is_file():
        logger.error(f"Path is not a file: {path}")
        return False

    return True


def validate_json_file(path: Path | str) -> bool:
    """
    Validate that a file contains valid JSON.

    Args:
        path: Path to JSON file.

    Returns:
        True if valid JSON, False otherwise.
    """
    path = Path(path)

    if not validate_path_exists(path, must_be_file=True):
        return False

    if path.suffix != ".json":
        logger.error(f"File is not a JSON file: {path}")
        return False

    try:
        import json

        with open(path, "r") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {path}: {e}")
        return False


def validate_config(data: dict[str, Any]) -> bool:
    """
    Validate configuration structure.

    Args:
        data: Configuration dictionary to validate.

    Returns:
        True if valid, False otherwise.
    """
    errors = []

    # Check top-level sections
    valid_sections = {"llm", "kb", "paths"}
    for section in data.keys():
        if section not in valid_sections:
            logger.warning(f"Unknown config section: {section}")

    # Validate LLM config
    if "llm" in data:
        llm = data["llm"]
        if not isinstance(llm, dict):
            errors.append("LLM config must be a dictionary")
        else:
            if "provider" in llm and not isinstance(llm["provider"], str):
                errors.append("LLM provider must be a string")
            if "model" in llm and not isinstance(llm["model"], str):
                errors.append("LLM model must be a string")

    # Validate KB config
    if "kb" in data:
        kb = data["kb"]
        if not isinstance(kb, dict):
            errors.append("KB config must be a dictionary")
        else:
            if "chroma_path" in kb and not isinstance(kb["chroma_path"], str):
                errors.append("KB chroma_path must be a string")

    # Validate paths config
    if "paths" in data:
        paths = data["paths"]
        if not isinstance(paths, dict):
            errors.append("Paths config must be a dictionary")

    # Log results
    if errors:
        for error in errors:
            logger.error(f"Config validation error: {error}")
        return False

    logger.info("Config validation passed")
    return True
