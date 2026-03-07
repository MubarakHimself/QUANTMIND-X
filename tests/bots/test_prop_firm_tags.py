"""
Tests for Prop Firm Tags in BotManifest.

Tests for:
- Prop firm tags in bot metadata
- Firm configuration support
- Tag operations (add, remove, has)
- Valid tag validation
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models.base import Base, TradingMode
from src.database.models.bots import BotManifest


TEST_DB_PATH = "test_prop_firm_tags.db"


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine."""
    engine = create_engine(f"sqlite:///{TEST_DB_PATH}")
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a test database session."""
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    yield session
    session.close()


class TestPropFirmTags:
    """Test prop firm tags functionality in BotManifest."""

    def test_default_prop_firm_tags_empty(self, test_session: Session):
        """Test that default prop_firm_tags is empty when no metadata."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        test_session.add(bot)
        test_session.commit()

        assert bot.prop_firm_tags == []
        assert bot.firm_config == {}

    def test_set_prop_firm_tags(self, test_session: Session):
        """Test setting prop firm tags."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        bot.prop_firm_tags = ["@fundednext", "@challenge_active"]
        test_session.add(bot)
        test_session.commit()
        test_session.refresh(bot)

        assert bot.prop_firm_tags == ["@fundednext", "@challenge_active"]

    def test_add_prop_firm_tag(self, test_session: Session):
        """Test adding a prop firm tag."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        test_session.add(bot)
        test_session.commit()

        bot.add_prop_firm_tag("@fundednext")
        test_session.commit()
        test_session.refresh(bot)

        assert "@fundednext" in bot.prop_firm_tags

    def test_add_duplicate_tag_no_change(self, test_session: Session):
        """Test that adding duplicate tag doesn't create duplicates."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        bot.prop_firm_tags = ["@fundednext"]
        test_session.add(bot)
        test_session.commit()

        bot.add_prop_firm_tag("@fundednext")
        test_session.commit()
        test_session.refresh(bot)

        assert bot.prop_firm_tags == ["@fundednext"]

    def test_remove_prop_firm_tag(self, test_session: Session):
        """Test removing a prop firm tag."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        bot.prop_firm_tags = ["@fundednext", "@challenge_active", "@funded"]
        test_session.add(bot)
        test_session.commit()

        bot.remove_prop_firm_tag("@challenge_active")
        test_session.commit()
        test_session.refresh(bot)

        assert "@challenge_active" not in bot.prop_firm_tags
        assert "@fundednext" in bot.prop_firm_tags
        assert "@funded" in bot.prop_firm_tags

    def test_has_prop_firm_tag(self, test_session: Session):
        """Test checking if bot has a prop firm tag."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        bot.prop_firm_tags = ["@fundednext", "@funded"]
        test_session.add(bot)
        test_session.commit()

        assert bot.has_prop_firm_tag("@fundednext") is True
        assert bot.has_prop_firm_tag("@challenge_active") is False
        assert bot.has_prop_firm_tag("@funded") is True

    def test_valid_prop_firm_tags(self):
        """Test that valid prop firm tags are returned correctly."""
        valid_tags = BotManifest.get_valid_prop_firm_tags()
        assert "@fundednext" in valid_tags
        assert "@challenge_active" in valid_tags
        assert "@funded" in valid_tags
        assert len(valid_tags) == 3


class TestFirmConfig:
    """Test firm configuration functionality in BotManifest."""

    def test_default_firm_config_empty(self, test_session: Session):
        """Test that default firm_config is empty when no metadata."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        test_session.add(bot)
        test_session.commit()

        assert bot.firm_config == {}

    def test_set_firm_config(self, test_session: Session):
        """Test setting firm configuration."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        config = {
            "firm_name": "FundedNext",
            "account_id": "FN12345",
            "max_drawdown_pct": 10.0,
            "daily_loss_limit_pct": 5.0
        }
        bot.firm_config = config
        test_session.add(bot)
        test_session.commit()
        test_session.refresh(bot)

        assert bot.firm_config == config
        assert bot.firm_config["firm_name"] == "FundedNext"
        assert bot.firm_config["account_id"] == "FN12345"

    def test_firm_config_persists_with_metadata(self, test_session: Session):
        """Test that firm_config persists alongside other metadata."""
        bot = BotManifest(
            bot_name="TestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD",
            bot_metadata={"custom_field": "custom_value"}
        )
        config = {
            "firm_name": "MyForexFunds",
            "account_id": "MFF67890"
        }
        bot.firm_config = config
        test_session.add(bot)
        test_session.commit()
        test_session.refresh(bot)

        # Verify both custom metadata and firm_config are preserved
        assert bot.bot_metadata["custom_field"] == "custom_value"
        assert bot.firm_config["firm_name"] == "MyForexFunds"
        assert bot.firm_config["account_id"] == "MFF67890"


class TestBotManifestWithPropFirm:
    """Test BotManifest with prop firm integration."""

    def test_create_bot_with_prop_firm_data(self, test_session: Session):
        """Test creating a bot with full prop firm data."""
        bot = BotManifest(
            bot_name="PropFirmBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="PROP_FIRM",
            required_margin=500.0,
            max_drawdown_pct=8.0,
            min_win_rate=0.60
        )
        # Set prop firm tags
        bot.prop_firm_tags = ["@challenge_active"]
        # Set firm config
        bot.firm_config = {
            "firm_name": "FundedNext",
            "account_id": "FN_CHALLENGE_123",
            "challenge_type": "express",
            "starting_balance": 10000.0
        }

        test_session.add(bot)
        test_session.commit()
        test_session.refresh(bot)

        # Verify all data
        assert bot.bot_name == "PropFirmBot"
        assert bot.broker_type == "PROP_FIRM"
        assert bot.has_prop_firm_tag("@challenge_active") is True
        assert bot.has_prop_firm_tag("@funded") is False
        assert bot.firm_config["firm_name"] == "FundedNext"
        assert bot.firm_config["challenge_type"] == "express"

    def test_tag_lifecycle_promotion(self, test_session: Session):
        """Test bot tag lifecycle - challenge to funded."""
        bot = BotManifest(
            bot_name="PromotionBot",
            bot_type="structural",
            strategy_type="mean_reversion",
            broker_type="PROP_FIRM"
        )
        bot.prop_firm_tags = ["@challenge_active"]
        test_session.add(bot)
        test_session.commit()

        # Bot passes challenge - add funded tag
        bot.add_prop_firm_tag("@funded")
        test_session.commit()
        test_session.refresh(bot)

        assert bot.has_prop_firm_tag("@challenge_active") is True
        assert bot.has_prop_firm_tag("@funded") is True

    def test_to_dict_includes_metadata(self, test_session: Session):
        """Test that to_dict includes prop firm data in metadata."""
        bot = BotManifest(
            bot_name="DictTestBot",
            bot_type="scalper",
            strategy_type="momentum",
            broker_type="STANDARD"
        )
        bot.prop_firm_tags = ["@fundednext", "@funded"]
        bot.firm_config = {"firm_name": "TestFirm", "account_id": "T123"}

        result = bot.to_dict()

        assert result["metadata"]["prop_firm_tags"] == ["@fundednext", "@funded"]
        assert result["metadata"]["firm_config"]["firm_name"] == "TestFirm"
        assert result["metadata"]["firm_config"]["account_id"] == "T123"
