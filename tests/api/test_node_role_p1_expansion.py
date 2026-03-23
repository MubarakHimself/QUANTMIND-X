"""
P1-P2 Tests: NODE_ROLE Environment Variable Scenarios (Epic 1 - Story 1-3)

Tests verify NODE_ROLE handling beyond basic routing:
- Case insensitivity
- Whitespace handling
- Import-time flag setting
- Validation warnings

Priority: P1-P2 (environment variable edge cases)
Risk: R-001 (NODE_ROLE invalid defaults silently)
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestNodeRoleCaseInsensitivity:
    """P1: NODE_ROLE values should be case-insensitive."""

    def test_cloudzy_lowercase_includes_cloudzy_routers(self):
        """P1: 'cloudzy' should include cloudzy routers."""
        with patch.dict(os.environ, {"NODE_ROLE": "cloudzy"}):
            # Re-import to get fresh values
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.INCLUDE_CLOUDZY is True
            assert server.NODE_ROLE == "cloudzy"

    def test_cloudzy_uppercase_includes_cloudzy_routers(self):
        """P1: 'CLOUDZY' should be normalized to 'cloudzy'."""
        with patch.dict(os.environ, {"NODE_ROLE": "CLOUDZY"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "cloudzy"
            assert server.INCLUDE_CLOUDZY is True

    def test_cloudzy_mixed_case_includes_cloudzy_routers(self):
        """P1: 'CloudZy' should be normalized to 'cloudzy'."""
        with patch.dict(os.environ, {"NODE_ROLE": "CloudZy"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "cloudzy"

    def test_contabo_uppercase_normalized(self):
        """P1: 'CONTABO' should be normalized to 'contabo'."""
        with patch.dict(os.environ, {"NODE_ROLE": "CONTABO"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "contabo"
            assert server.INCLUDE_CONTABO is True

    def test_local_uppercase_normalized(self):
        """P1: 'LOCAL' should be normalized to 'local'."""
        with patch.dict(os.environ, {"NODE_ROLE": "LOCAL"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"


class TestNodeRoleWhitespaceHandling:
    """P1: NODE_ROLE with whitespace should be handled."""

    def test_role_with_leading_space(self):
        """P1: NODE_ROLE with leading space should be trimmed."""
        with patch.dict(os.environ, {"NODE_ROLE": "  cloudzy"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "cloudzy"

    def test_role_with_trailing_space(self):
        """P1: NODE_ROLE with trailing space should be trimmed."""
        with patch.dict(os.environ, {"NODE_ROLE": "cloudzy  "}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "cloudzy"

    def test_role_with_surrounding_spaces(self):
        """P1: NODE_ROLE with surrounding spaces should be trimmed."""
        with patch.dict(os.environ, {"NODE_ROLE": "  contabo  "}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "contabo"


class TestNodeRoleValidationWarnings:
    """P1: Invalid NODE_ROLE should trigger warnings."""

    def test_invalid_role_defaults_to_local(self):
        """P1: Invalid NODE_ROLE value should default to 'local'."""
        with patch.dict(os.environ, {"NODE_ROLE": "invalid_role"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"

    def test_empty_role_defaults_to_local(self):
        """P1: Empty NODE_ROLE should default to 'local'."""
        with patch.dict(os.environ, {"NODE_ROLE": ""}):
            import importlib
            from src.api import server
            importlib.reload(server)

            # Empty string is not in valid roles, should default
            assert server.NODE_ROLE == "local"

    def test_numeric_role_defaults_to_local(self):
        """P1: Numeric NODE_ROLE should default to 'local'."""
        with patch.dict(os.environ, {"NODE_ROLE": "12345"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"


class TestNodeRoleFlagCombinations:
    """P1: INCLUDE_CLOUDZY and INCLUDE_CONTABO flag combinations."""

    def test_cloudzy_mode_excludes_contabo(self):
        """P1: NODE_ROLE=cloudzy should exclude contabo routers."""
        with patch.dict(os.environ, {"NODE_ROLE": "cloudzy"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.INCLUDE_CLOUDZY is True
            assert server.INCLUDE_CONTABO is False

    def test_contabo_mode_excludes_cloudzy(self):
        """P1: NODE_ROLE=contabo should exclude cloudzy routers."""
        with patch.dict(os.environ, {"NODE_ROLE": "contabo"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.INCLUDE_CONTABO is True
            assert server.INCLUDE_CLOUDZY is False

    def test_local_mode_includes_both(self):
        """P1: NODE_ROLE=local should include both cloudzy and contabo routers."""
        with patch.dict(os.environ, {"NODE_ROLE": "local"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.INCLUDE_CLOUDZY is True
            assert server.INCLUDE_CONTABO is True


class TestNodeRoleImportTime:
    """P2: Flags are set at module import time."""

    def test_flags_set_at_import_time(self):
        """P2: INCLUDE_CLOUDZY and INCLUDE_CONTABO are set when module loads.

        Note: Due to Python module caching, changing os.environ after import
        will NOT update the flags. This is the expected behavior.
        """
        # The flags are set at lines 108-109 in server.py:
        # INCLUDE_CLOUDZY = NODE_ROLE in ("cloudzy", "local")
        # INCLUDE_CONTABO = NODE_ROLE in ("contabo", "local")
        #
        # This means if NODE_ROLE changes after import, the flags won't update.
        # This is by design for the deployment architecture.

        # Verify flags exist and are boolean
        from src.api.server import INCLUDE_CLOUDZY, INCLUDE_CONTABO, NODE_ROLE

        assert isinstance(INCLUDE_CLOUDZY, bool)
        assert isinstance(INCLUDE_CONTABO, bool)
        assert isinstance(NODE_ROLE, str)
        assert NODE_ROLE in ("cloudzy", "contabo", "local")

    def test_valid_roles_set_unchanged(self):
        """P2: Valid NODE_ROLE values should not be modified."""
        valid_roles = ["cloudzy", "contabo", "local"]

        for role in valid_roles:
            with patch.dict(os.environ, {"NODE_ROLE": role}):
                import importlib
                from src.api import server
                importlib.reload(server)

                assert server.NODE_ROLE == role


class TestNodeRoleRouterSets:
    """P2: Router classification sets are correctly defined."""

    def test_cloudzy_routers_defined(self):
        """P2: CLOUDZY_ROUTERS set contains expected routers."""
        from src.api.server import CLOUDZY_ROUTERS

        expected = {
            "kill_switch_router",
            "trading_router",
            "broker_router",
            "paper_trading_router",
            "tradingview_router",
            "lifecycle_scanner_router"
        }

        assert CLOUDZY_ROUTERS == expected

    def test_contabo_routers_defined(self):
        """P2: CONTABO_ROUTERS set contains expected routers."""
        from src.api.server import CONTABO_ROUTERS

        # Check for key routers that should be in contabo
        assert "settings_router" in CONTABO_ROUTERS
        assert "provider_config_router" in CONTABO_ROUTERS
        assert "memory_router" in CONTABO_ROUTERS
        assert "floor_manager_router" in CONTABO_ROUTERS
        assert "workflow_router" in CONTABO_ROUTERS
        assert "knowledge_router" in CONTABO_ROUTERS

    def test_both_routers_defined(self):
        """P2: BOTH_ROUTERS set contains routers needed in all modes."""
        from src.api.server import BOTH_ROUTERS

        expected = {"health_router", "version_router", "metrics_router"}
        assert BOTH_ROUTERS == expected


class TestNodeRoleEdgeCases:
    """P2: Edge case handling for NODE_ROLE."""

    def test_role_with_special_characters(self):
        """P2: NODE_ROLE with special characters should default to local."""
        with patch.dict(os.environ, {"NODE_ROLE": "cloudzy!"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"

    def test_role_with_path_traversal(self):
        """P2: NODE_ROLE with path-like value should default to local."""
        with patch.dict(os.environ, {"NODE_ROLE": "../etc/passwd"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"

    def test_role_with_sql_injection_attempt(self):
        """P2: NODE_ROLE with SQL-like value should default to local."""
        with patch.dict(os.environ, {"NODE_ROLE": "'; DROP TABLE users; --"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            assert server.NODE_ROLE == "local"

    def test_role_with_unicode(self):
        """P2: NODE_ROLE with unicode should be handled."""
        with patch.dict(os.environ, {"NODE_ROLE": "cloudzy\x00"}):
            import importlib
            from src.api import server
            importlib.reload(server)

            # Null character should cause default
            assert server.NODE_ROLE == "local"
