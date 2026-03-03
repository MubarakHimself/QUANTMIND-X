import pytest
from src.router.commands import CommandHandler, TradeCommand, RiskCommand


def test_command_imports():
    """Command classes should be importable from new module."""
    assert CommandHandler is not None
    assert TradeCommand is not None
    assert RiskCommand is not None
