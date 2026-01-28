"""
File I/O operations for the analyst agent.

This module provides utilities for loading NPRD JSON files, saving TRD outputs,
and managing output directories.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_nprd_json(path: Path | str) -> dict[str, Any]:
    """
    Load NPRD (Natural Language Product Requirements Document) from JSON file.

    Args:
        path: Path to the NPRD JSON file.

    Returns:
        Dictionary containing NPRD data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contains invalid JSON.
        KeyError: If required fields are missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NPRD file not found: {path}")

    if not path.suffix == ".json":
        raise ValueError(f"NPRD file must be JSON format: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in NPRD file {path}: {e}")

    # Basic validation
    if not isinstance(data, dict):
        raise ValueError(f"NPRD data must be a dictionary: {path}")

    logger.info(f"Loaded NPRD from {path}")
    return data


def save_trd(
    content: str,
    output_dir: Path | str,
    filename: Optional[str] = None,
) -> Path:
    """
    Save TRD (Technical Requirements Document) to file.

    Args:
        content: TRD content in markdown format.
        output_dir: Directory to save the TRD file.
        filename: Optional filename. If not provided, generates timestamp-based name.

    Returns:
        Path to the saved TRD file.

    Raises:
        ValueError: If content is empty.
        OSError: If unable to write to the output directory.
    """
    if not content or not content.strip():
        raise ValueError("TRD content cannot be empty")

    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}")

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trd_{timestamp}.md"
    elif not filename.endswith(".md"):
        filename += ".md"

    output_path = output_dir / filename

    # Write content to file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        raise OSError(f"Failed to write TRD to {output_path}: {e}")

    logger.info(f"Saved TRD to {output_path}")
    return output_path


def list_nprd_outputs(
    output_dir: Path | str,
    pattern: str = "*.json",
) -> list[Path]:
    """
    List all NPRD output files in the specified directory.

    Args:
        output_dir: Directory containing NPRD files.
        pattern: Glob pattern for matching files (default: "*.json").

    Returns:
        List of Path objects for NPRD files, sorted by modification time (newest first).

    Raises:
        FileNotFoundError: If the output directory doesn't exist.
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"NPRD output directory not found: {output_dir}")

    if not output_dir.is_dir():
        raise ValueError(f"NPRD output path is not a directory: {output_dir}")

    # Find all matching files
    files = list(output_dir.glob(pattern))

    # Filter out directories
    files = [f for f in files if f.is_file()]

    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    logger.debug(f"Found {len(files)} NPRD files in {output_dir}")
    return files


def ensure_directory(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        Path object for the directory.

    Raises:
        OSError: If unable to create the directory.
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}")

    return path


def read_text_file(path: Path | str, encoding: str = "utf-8") -> str:
    """
    Read text content from a file.

    Args:
        path: Path to the file.
        encoding: File encoding (default: utf-8).

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        UnicodeDecodeError: If the file cannot be decoded with the specified encoding.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            encoding,
            e.object,
            e.start,
            e.end,
            f"Failed to decode file {path} with encoding {encoding}",
        )


def write_text_file(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """
    Write text content to a file.

    Args:
        path: Path to the file.
        content: Content to write.
        encoding: File encoding (default: utf-8).

    Returns:
        Path object for the written file.

    Raises:
        OSError: If unable to write to the file.
    """
    path = Path(path)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
    except OSError as e:
        raise OSError(f"Failed to write file {path}: {e}")

    logger.debug(f"Wrote text file to {path}")
    return path
