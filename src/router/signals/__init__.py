"""
Signals module for trading signal generation.

Provides confluence-based signal detection using multiple technical indicators.
"""

from src.router.signals.confluence_signal import ConfluenceSignal, SignalType

__all__ = ["ConfluenceSignal", "SignalType"]
