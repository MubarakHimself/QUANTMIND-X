"""
TRD (Technical Requirements Document) markdown formatting utilities.

This module provides functions for formatting TRD documents with proper
frontmatter and section structure.
"""

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def format_trd_frontmatter(
    metadata: dict[str, Any],
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Format TRD frontmatter in YAML format.

    Args:
        metadata: Dictionary containing metadata fields.
        title: Optional title override.
        description: Optional description override.

    Returns:
        YAML frontmatter block as string.
    """
    # Extract common metadata fields
    doc_id = metadata.get("id", metadata.get("document_id", "unknown"))
    version = metadata.get("version", "1.0")
    author = metadata.get("author", "Analyst Agent")
    created = metadata.get("created", datetime.now().isoformat())
    updated = metadata.get("updated", datetime.now().isoformat())
    status = metadata.get("status", "draft")

    # Build frontmatter
    lines = ["---"]

    # Basic document info
    lines.append(f"id: {doc_id}")
    lines.append(f"title: {title or metadata.get('title', 'Technical Requirements Document')}")
    lines.append(f"version: {version}")
    lines.append(f"author: {author}")
    lines.append(f"created: {created}")
    lines.append(f"updated: {updated}")
    lines.append(f"status: {status}")

    # Add description if provided
    if description or "description" in metadata:
        desc = description or metadata.get("description", "")
        lines.append(f"description: {desc}")

    # Add any additional metadata
    for key, value in metadata.items():
        if key not in {
            "id",
            "document_id",
            "title",
            "version",
            "author",
            "created",
            "updated",
            "status",
            "description",
        }:
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def format_trd_sections(
    concepts: Optional[list[dict[str, Any]]] = None,
    kb_results: Optional[list[dict[str, Any]]] = None,
    analysis: Optional[str] = None,
    requirements: Optional[list[str]] = None,
) -> str:
    """
    Format TRD content sections.

    Args:
        concepts: List of concept analysis results.
        kb_results: List of knowledge base search results.
        analysis: Text analysis content.
        requirements: List of technical requirements.

    Returns:
        Formatted markdown content.
    """
    sections = []

    # Executive Summary
    sections.append("## Executive Summary\n")
    if analysis:
        sections.append(f"{analysis}\n")
    else:
        sections.append("_Analysis pending_\n")

    # Concept Analysis
    if concepts:
        sections.append("## Concept Analysis\n")
        for i, concept in enumerate(concepts, 1):
            name = concept.get("name", f"Concept {i}")
            description = concept.get("description", "")
            relevance = concept.get("relevance", "N/A")

            sections.append(f"### {name}")
            sections.append(f"**Relevance:** {relevance}\n")
            if description:
                sections.append(f"{description}\n")

    # Knowledge Base References
    if kb_results:
        sections.append("## Knowledge Base References\n")
        for i, result in enumerate(kb_results, 1):
            title = result.get("title", f"Reference {i}")
            source = result.get("source", "Unknown")
            snippet = result.get("content", result.get("snippet", ""))
            score = result.get("score", result.get("similarity", "N/A"))

            sections.append(f"### {title}")
            sections.append(f"**Source:** {source}")
            sections.append(f"**Relevance Score:** {score}\n")

            if snippet:
                # Truncate long snippets
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                sections.append(f"```\n{snippet}\n```\n")

    # Technical Requirements
    if requirements:
        sections.append("## Technical Requirements\n")
        for i, req in enumerate(requirements, 1):
            sections.append(f"{i}. {req}\n")
    else:
        sections.append("## Technical Requirements\n")
        sections.append("_Requirements pending_\n")

    return "\n".join(sections)


def format_trd_document(
    metadata: dict[str, Any],
    concepts: Optional[list[dict[str, Any]]] = None,
    kb_results: Optional[list[dict[str, Any]]] = None,
    analysis: Optional[str] = None,
    requirements: Optional[list[str]] = None,
) -> str:
    """
    Format a complete TRD document.

    Args:
        metadata: Document metadata.
        concepts: List of concept analysis results.
        kb_results: List of knowledge base search results.
        analysis: Text analysis content.
        requirements: List of technical requirements.

    Returns:
        Complete TRD markdown document.
    """
    frontmatter = format_trd_frontmatter(metadata)
    sections = format_trd_sections(concepts, kb_results, analysis, requirements)

    return f"{frontmatter}{sections}"


def format_requirement(
    id: str,
    title: str,
    description: str,
    priority: str = "medium",
    category: Optional[str] = None,
    acceptance_criteria: Optional[list[str]] = None,
) -> str:
    """
    Format a single requirement in markdown.

    Args:
        id: Requirement identifier (e.g., "REQ-001").
        title: Requirement title.
        description: Requirement description.
        priority: Priority level (low/medium/high/critical).
        category: Optional category.
        acceptance_criteria: Optional list of acceptance criteria.

    Returns:
        Formatted requirement markdown.
    """
    lines = [f"### {id}: {title}"]
    lines.append(f"**Priority:** {priority}")

    if category:
        lines.append(f"**Category:** {category}")

    lines.append("")
    lines.append(f"{description}")
    lines.append("")

    if acceptance_criteria:
        lines.append("**Acceptance Criteria:**")
        for criteria in acceptance_criteria:
            lines.append(f"- {criteria}")
        lines.append("")

    return "\n".join(lines)


def format_concept(
    name: str,
    definition: str,
    context: Optional[str] = None,
    examples: Optional[list[str]] = None,
    related_concepts: Optional[list[str]] = None,
) -> str:
    """
    Format a concept explanation in markdown.

    Args:
        name: Concept name.
        definition: Concept definition.
        context: Optional context information.
        examples: Optional list of examples.
        related_concepts: Optional list of related concepts.

    Returns:
        Formatted concept markdown.
    """
    lines = [f"### {name}"]
    lines.append("")
    lines.append(f"**Definition:** {definition}")
    lines.append("")

    if context:
        lines.append(f"**Context:** {context}")
        lines.append("")

    if examples:
        lines.append("**Examples:**")
        for example in examples:
            lines.append(f"- {example}")
        lines.append("")

    if related_concepts:
        lines.append("**Related Concepts:**")
        for concept in related_concepts:
            lines.append(f"- {concept}")
        lines.append("")

    return "\n".join(lines)
