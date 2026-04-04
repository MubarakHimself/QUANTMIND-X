"""
Paper trading runtime helpers.

This module keeps API routes deployment-safe on hosts where the MT5 paper
trading runtime is intentionally unavailable, such as local Linux workstations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException


@dataclass
class PaperTradingUnavailableDeployer:
    """Read-safe deployer stub used when the MT5 runtime is not installed."""

    unavailable_reason: str
    available: bool = False

    def list_agents(self) -> list[Any]:
        return []

    def get_agent(self, agent_id: str) -> None:
        return None

    def stop_agent(self, agent_id: str) -> bool:
        return False

    def get_agent_logs(self, agent_id: str, tail_lines: int = 100) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "logs": [],
            "available": False,
            "detail": self.unavailable_reason,
        }

    def deploy_agent(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(self.unavailable_reason)


def is_paper_trading_runtime_available(deployer: Any) -> bool:
    return bool(getattr(deployer, "available", True))


def ensure_paper_trading_runtime(deployer: Any) -> None:
    """Raise a clean 503 when a write/action endpoint requires MT5 runtime."""
    if is_paper_trading_runtime_available(deployer):
        return

    raise HTTPException(
        status_code=503,
        detail=getattr(
            deployer,
            "unavailable_reason",
            "Paper trading runtime is unavailable on this host.",
        ),
    )
