"""
Unit tests for BotManifest AccountBook fields.

Tests account_book_type, prop_firm_name, and max_drawdown_pct fields.
"""

import pytest
from datetime import datetime
from src.router.bot_manifest import (
    BotManifest,
    BotRegistry,
    StrategyType,
    TradeFrequency,
    AccountBook,
)


class TestAccountBookFields:
    """Tests for BotManifest AccountBook fields."""

    def test_manifest_with_personal_account(self):
        """Test manifest with PERSONAL account book type."""
        manifest = BotManifest(
            bot_id="personal_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.MEDIUM,
            account_book_type=AccountBook.PERSONAL,
        )

        assert manifest.account_book_type == AccountBook.PERSONAL
        assert manifest.prop_firm_name is None

    def test_manifest_with_prop_firm_account(self):
        """Test manifest with PROP_FIRM account book type."""
        manifest = BotManifest(
            bot_id="prop_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.MEDIUM,
            account_book_type=AccountBook.PROP_FIRM,
            prop_firm_name="Blue Wave",
        )

        assert manifest.account_book_type == AccountBook.PROP_FIRM
        assert manifest.prop_firm_name == "Blue Wave"

    def test_manifest_with_max_drawdown(self):
        """Test manifest with max_drawdown_pct field."""
        manifest = BotManifest(
            bot_id="dd_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.MEDIUM,
            max_drawdown_pct=10.0,
        )

        assert manifest.max_drawdown_pct == 10.0

    def test_manifest_full_account_config(self):
        """Test manifest with all account book fields."""
        manifest = BotManifest(
            bot_id="full_config_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            account_book_type=AccountBook.PROP_FIRM,
            prop_firm_name="True Forex Funds",
            max_drawdown_pct=8.5,
        )

        assert manifest.account_book_type == AccountBook.PROP_FIRM
        assert manifest.prop_firm_name == "True Forex Funds"
        assert manifest.max_drawdown_pct == 8.5


class TestAccountBookSerialization:
    """Tests for AccountBook field serialization/deserialization."""

    def test_to_dict_with_personal_account(self):
        """Test to_dict includes account_book_type field for PERSONAL."""
        manifest = BotManifest(
            bot_id="test_bot",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            account_book_type=AccountBook.PERSONAL,
        )

        manifest_dict = manifest.to_dict()

        assert manifest_dict["account_book_type"] == "PERSONAL"
        assert manifest_dict["prop_firm_name"] is None
        assert manifest_dict["max_drawdown_pct"] is None

    def test_to_dict_with_prop_firm_account(self):
        """Test to_dict includes account_book_type field for PROP_FIRM."""
        manifest = BotManifest(
            bot_id="prop_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            account_book_type=AccountBook.PROP_FIRM,
            prop_firm_name="Blue Wave",
            max_drawdown_pct=10.0,
        )

        manifest_dict = manifest.to_dict()

        assert manifest_dict["account_book_type"] == "PROP_FIRM"
        assert manifest_dict["prop_firm_name"] == "Blue Wave"
        assert manifest_dict["max_drawdown_pct"] == 10.0

    def test_from_dict_with_personal_account(self):
        """Test from_dict deserializes PERSONAL account."""
        data = {
            "bot_id": "test_bot",
            "strategy_type": "SCALPER",
            "frequency": "HIGH",
            "account_book_type": "PERSONAL",
        }

        manifest = BotManifest.from_dict(data)

        assert manifest.account_book_type == AccountBook.PERSONAL
        assert manifest.prop_firm_name is None

    def test_from_dict_with_prop_firm_account(self):
        """Test from_dict deserializes PROP_FIRM account."""
        data = {
            "bot_id": "prop_bot",
            "strategy_type": "STRUCTURAL",
            "frequency": "LOW",
            "account_book_type": "PROP_FIRM",
            "prop_firm_name": "True Forex Funds",
            "max_drawdown_pct": 8.5,
        }

        manifest = BotManifest.from_dict(data)

        assert manifest.account_book_type == AccountBook.PROP_FIRM
        assert manifest.prop_firm_name == "True Forex Funds"
        assert manifest.max_drawdown_pct == 8.5

    def test_from_dict_without_account_book_type(self):
        """Test from_dict handles missing account_book_type (backward compatibility)."""
        data = {
            "bot_id": "test_bot",
            "strategy_type": "SCALPER",
            "frequency": "HIGH",
        }

        manifest = BotManifest.from_dict(data)

        # Should default to PERSONAL
        assert manifest.account_book_type == AccountBook.PERSONAL

    def test_roundtrip_with_all_fields(self):
        """Test serialization/deserialization roundtrip with all account fields."""
        original = BotManifest(
            bot_id="roundtrip_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            account_book_type=AccountBook.PROP_FIRM,
            prop_firm_name="Blue Wave",
            max_drawdown_pct=10.0,
        )

        # Serialize
        manifest_dict = original.to_dict()

        # Deserialize
        restored = BotManifest.from_dict(manifest_dict)

        assert restored.bot_id == original.bot_id
        assert restored.account_book_type == original.account_book_type
        assert restored.prop_firm_name == original.prop_firm_name
        assert restored.max_drawdown_pct == original.max_drawdown_pct


class TestAccountBookEnum:
    """Tests for AccountBook enum values."""

    def test_account_book_personal_value(self):
        """Test AccountBook.PERSONAL has correct value."""
        assert AccountBook.PERSONAL.value == "PERSONAL"

    def test_account_book_prop_firm_value(self):
        """Test AccountBook.PROP_FIRM has correct value."""
        assert AccountBook.PROP_FIRM.value == "PROP_FIRM"

    def test_account_book_from_string(self):
        """Test AccountBook can be created from string."""
        assert AccountBook("PERSONAL") == AccountBook.PERSONAL
        assert AccountBook("PROP_FIRM") == AccountBook.PROP_FIRM


class TestBotRegistryAccountBook:
    """Tests for BotRegistry with AccountBook fields."""

    def test_registry_stores_account_fields(self):
        """Test that registry preserves account book fields."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "test_registry.json")
            registry = BotRegistry(storage_path=registry_path)

            manifest = BotManifest(
                bot_id="account_bot",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.MEDIUM,
                account_book_type=AccountBook.PROP_FIRM,
                prop_firm_name="Blue Wave",
                max_drawdown_pct=10.0,
                tags=["@primal"],
            )

            # Register
            registry.register(manifest)

            # Retrieve
            retrieved = registry.get("account_bot")

            assert retrieved is not None
            assert retrieved.account_book_type == AccountBook.PROP_FIRM
            assert retrieved.prop_firm_name == "Blue Wave"
            assert retrieved.max_drawdown_pct == 10.0

    def test_list_personal_vs_prop_firm_bots(self):
        """Test filtering bots by account book type."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "test_registry.json")
            registry = BotRegistry(storage_path=registry_path)

            # Register personal account bots
            for i in range(2):
                manifest = BotManifest(
                    bot_id=f"personal_{i}",
                    strategy_type=StrategyType.STRUCTURAL,
                    frequency=TradeFrequency.MEDIUM,
                    account_book_type=AccountBook.PERSONAL,
                    tags=["@personal"],
                )
                registry.register(manifest)

            # Register prop firm bots
            for i in range(3):
                manifest = BotManifest(
                    bot_id=f"prop_{i}",
                    strategy_type=StrategyType.STRUCTURAL,
                    frequency=TradeFrequency.MEDIUM,
                    account_book_type=AccountBook.PROP_FIRM,
                    prop_firm_name="Blue Wave",
                    tags=["@prop_firm"],
                )
                registry.register(manifest)

            # Get all bots
            all_bots = registry.list_all()

            personal_count = sum(
                1 for b in all_bots if b.account_book_type == AccountBook.PERSONAL
            )
            prop_firm_count = sum(
                1 for b in all_bots if b.account_book_type == AccountBook.PROP_FIRM
            )

            assert personal_count == 2
            assert prop_firm_count == 3
