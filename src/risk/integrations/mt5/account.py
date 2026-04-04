"""Account data models and managers for MT5 integration."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _default_simulated_fallback() -> bool:
    value = os.environ.get("MT5_ALLOW_SIMULATED_FALLBACK")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AccountInfo:
    """Account information model."""
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    leverage: int
    currency_base: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "login": self.login,
            "server": self.server,
            "currency": self.currency,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "margin_free": self.margin_free,
            "margin_level": self.margin_level,
            "profit": self.profit,
            "leverage": self.leverage,
            "currency_base": self.currency_base
        }


@dataclass
class MarginInfo:
    """Margin information model."""
    margin_required: float
    margin_free: float
    margin_level: float
    margin_used: float
    equity: float
    balance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "margin_required": self.margin_required,
            "margin_free": self.margin_free,
            "margin_level": self.margin_level,
            "margin_used": self.margin_used,
            "equity": self.equity,
            "balance": self.balance
        }


# Simulated data for fallback mode
SIMULATED_ACCOUNT = {
    "login": 12345678,
    "server": "Demo-Server",
    "currency": "USD",
    "balance": 10000.0,
    "equity": 10500.0,
    "margin": 200.0,
    "margin_free": 10300.0,
    "margin_level": 5250.0,
    "profit": 500.0,
    "leverage": 100
}


class AccountManager:
    """
    MT5 Account Manager for account operations.

    Provides account information retrieval, margin calculations,
    and account switching capabilities.
    """

    def __init__(self, account_manager=None, fallback_to_simulated: Optional[bool] = None):
        """
        Initialize Account Manager.

        Args:
            account_manager: MT5 account manager instance
            fallback_to_simulated: Use simulated data when MT5 unavailable
        """
        self._account_manager = account_manager
        self._fallback_to_simulated = (
            _default_simulated_fallback()
            if fallback_to_simulated is None
            else fallback_to_simulated
        )

    def get_account_balance(self) -> Optional[float]:
        """Get account balance."""
        if not self._account_manager:
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["balance"]
            return None

        try:
            status = self._account_manager.get_connection_status()
            if status.get("connected") and "balance" in status:
                return float(status["balance"])
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["balance"]
            return None
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["balance"]
            return None

    def get_account_equity(self) -> Optional[float]:
        """Get account equity."""
        if not self._account_manager:
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["equity"]
            return None

        try:
            status = self._account_manager.get_connection_status()
            if status.get("connected") and "equity" in status:
                return float(status["equity"])
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["equity"]
            return None
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            if self._fallback_to_simulated:
                return SIMULATED_ACCOUNT["equity"]
            return None

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get comprehensive account information."""
        if not self._account_manager:
            if self._fallback_to_simulated:
                return AccountInfo(**SIMULATED_ACCOUNT)
            return None

        try:
            status = self._account_manager.get_connection_status()
            if status.get("connected"):
                return AccountInfo(
                    login=int(status.get("current_login", 0)),
                    server=status.get("current_server", ""),
                    currency=status.get("currency", "USD"),
                    balance=float(status.get("balance", 0)),
                    equity=float(status.get("equity", 0)),
                    margin=float(status.get("margin", 0)),
                    margin_free=float(status.get("free_margin", 0)),
                    margin_level=float(status.get("margin_level", 0)),
                    profit=float(status.get("profit", 0)),
                    leverage=int(status.get("leverage", 100)),
                    currency_base=status.get("currency", "USD")
                )
            if self._fallback_to_simulated:
                return AccountInfo(**SIMULATED_ACCOUNT)
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            if self._fallback_to_simulated:
                return AccountInfo(**SIMULATED_ACCOUNT)
            return None

    def get_margin_info(self) -> Optional[MarginInfo]:
        """Get margin information."""
        if not self._account_manager:
            if self._fallback_to_simulated:
                return MarginInfo(
                    margin_required=SIMULATED_ACCOUNT["margin"],
                    margin_free=SIMULATED_ACCOUNT["margin_free"],
                    margin_level=SIMULATED_ACCOUNT["margin_level"],
                    margin_used=SIMULATED_ACCOUNT["margin"],
                    equity=SIMULATED_ACCOUNT["equity"],
                    balance=SIMULATED_ACCOUNT["balance"]
                )
            return None

        try:
            status = self._account_manager.get_connection_status()
            if status.get("connected"):
                return MarginInfo(
                    margin_required=float(status.get("margin", 0)),
                    margin_free=float(status.get("free_margin", 0)),
                    margin_level=float(status.get("margin_level", 0)),
                    margin_used=float(status.get("margin", 0)),
                    equity=float(status.get("equity", 0)),
                    balance=float(status.get("balance", 0))
                )
            if self._fallback_to_simulated:
                return MarginInfo(
                    margin_required=SIMULATED_ACCOUNT["margin"],
                    margin_free=SIMULATED_ACCOUNT["margin_free"],
                    margin_level=SIMULATED_ACCOUNT["margin_level"],
                    margin_used=SIMULATED_ACCOUNT["margin"],
                    equity=SIMULATED_ACCOUNT["equity"],
                    balance=SIMULATED_ACCOUNT["balance"]
                )
            return None
        except Exception as e:
            logger.error(f"Error getting margin info: {e}")
            if self._fallback_to_simulated:
                return MarginInfo(
                    margin_required=SIMULATED_ACCOUNT["margin"],
                    margin_free=SIMULATED_ACCOUNT["margin_free"],
                    margin_level=SIMULATED_ACCOUNT["margin_level"],
                    margin_used=SIMULATED_ACCOUNT["margin"],
                    equity=SIMULATED_ACCOUNT["equity"],
                    balance=SIMULATED_ACCOUNT["balance"]
                )
            return None
