"""
Test for IDE endpoints modular structure.

This test verifies that the IDE endpoints have been properly split
into modular components: SessionEndpoint, FileEndpoint, TerminalEndpoint.
"""
from src.api.ide import SessionEndpoint, FileEndpoint, TerminalEndpoint


def test_ide_imports():
    """Test that all IDE endpoint classes can be imported."""
    assert SessionEndpoint is not None
    assert FileEndpoint is not None
    assert TerminalEndpoint is not None
