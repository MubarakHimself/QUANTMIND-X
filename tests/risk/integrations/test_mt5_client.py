"""Tests for MT5 modular imports."""
import pytest
from src.risk.integrations.mt5 import MT5Client, AccountManager, SymbolInfo


def test_mt5_imports():
    """Scanners should be importable from new module."""
    assert MT5Client is not None
    assert AccountManager is not None
    assert SymbolInfo is not None
