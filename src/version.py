"""
Version Management Module for QuantMindX

Provides a single source of truth for versioning across the system.
Used by the API, CLI, and deployment scripts.
"""

import os
from datetime import datetime
from typing import Dict

# Single source of truth for version
__version__: str = "1.0.0"

# Version history with release notes
VERSION_HISTORY: Dict[str, str] = {
    "1.0.0": "Initial release with core trading infrastructure, API server, MT5 bridge, and monitoring stack"
}


def get_version() -> str:
    """Return the current version string."""
    return __version__


def get_version_info() -> dict:
    """
    Return comprehensive version information.
    
    Returns:
        dict containing:
            - version: Current version string
            - env: Environment (from QUANTMIND_ENV)
            - build_timestamp: ISO format timestamp
            - history: Version history dict
    """
    return {
        "version": __version__,
        "env": os.getenv("QUANTMIND_ENV", "development"),
        "build_timestamp": datetime.now().isoformat(),
        "history": VERSION_HISTORY
    }