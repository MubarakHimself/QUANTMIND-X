"""
Trading Session Risk Endpoints

Exposes Reverse House Money (RHM) session risk state via REST API.
Part of Story 11: Reverse House Money Session Indicator
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.risk.sizing.session_kelly_modifiers import SessionKellyModifiers, SessionKellyState

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton instances per account_id
_modifiers: dict[str, SessionKellyModifiers] = {}


def get_modifiers(account_id: str = "default") -> SessionKellyModifiers:
    """Get or create SessionKellyModifiers instance for account."""
    if account_id not in _modifiers:
        _modifiers[account_id] = SessionKellyModifiers(account_id=account_id)
    return _modifiers[account_id]


@router.get("/session-risk-state")
async def get_session_risk_state(account_id: str = "default") -> SessionKellyState:
    """
    Get current Reverse House Money session risk state.

    Returns the session loss counter, RHM multiplier, and HMM state
    for the live trading canvas.

    Args:
        account_id: Trading account identifier (default: "default")

    Returns:
        SessionKellyState with current RHM session risk metrics
    """
    try:
        modifiers = get_modifiers(account_id)
        return modifiers.get_current_state()
    except Exception as e:
        logger.error(f"Failed to get session risk state: {e}")
        raise HTTPException(status_code=500, detail=str(e))
