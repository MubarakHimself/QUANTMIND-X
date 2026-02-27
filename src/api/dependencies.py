"""
API Dependencies for the Department-Based Agent Framework.
"""

from src.agents.tools.ea_lifecycle import EALifecycleTools


def get_ea_lifecycle_tools() -> EALifecycleTools:
    """
    Get EA lifecycle tools instance.

    Returns:
        EALifecycleTools: Instance of EA lifecycle tools
    """
    return EALifecycleTools()