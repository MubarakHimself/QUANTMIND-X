"""
Tests for AccountType Enum and Field (MUB-31)

Tests:
- AccountType enum values
- account_type field on PropFirmAccount model
- Default value behavior
- Query by account_type

Reference: MUB-31 Task 1
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base, PropFirmAccount, AccountType, TradingMode


TEST_DB_PATH = "test_account_type.db"


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine."""
    engine = create_engine(f"sqlite:///{TEST_DB_PATH}")
    Base.metadata.create_all(bind=engine)
    yield engine
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


class TestAccountTypeEnum:
    """Test AccountType enum values and behavior."""

    def test_account_type_enum_values(self):
        """Test that AccountType enum has correct values."""
        assert AccountType.PERSONAL.value == "personal"
        assert AccountType.PROP_FIRM.value == "prop_firm"

    def test_account_type_enum_members(self):
        """Test that AccountType has exactly two members."""
        assert len(AccountType) == 2
        assert AccountType.PERSONAL in AccountType
        assert AccountType.PROP_FIRM in AccountType


class TestPropFirmAccountType:
    """Test account_type field on PropFirmAccount model."""

    def test_default_account_type_is_prop_firm(self, test_session: Session):
        """Test that default account_type is PROP_FIRM."""
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="99999",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        assert account.account_type == AccountType.PROP_FIRM

    def test_set_account_type_to_personal(self, test_session: Session):
        """Test creating an account with PERSONAL type."""
        account = PropFirmAccount(
            firm_name="PersonalTrading",
            account_id="88888",
            daily_loss_limit_pct=2.0,
            hard_stop_buffer_pct=0.5,
            target_profit_pct=5.0,
            min_trading_days=3,
            account_type=AccountType.PERSONAL
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        assert account.account_type == AccountType.PERSONAL
        assert account.firm_name == "PersonalTrading"

    def test_query_by_account_type_prop_firm(self, test_session: Session):
        """Test querying accounts by PROP_FIRM account_type."""
        # Create accounts
        prop_account = PropFirmAccount(
            firm_name="PropFirm1",
            account_id="11111",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5,
            account_type=AccountType.PROP_FIRM
        )
        personal_account = PropFirmAccount(
            firm_name="MyPersonal",
            account_id="22222",
            daily_loss_limit_pct=2.0,
            hard_stop_buffer_pct=0.5,
            target_profit_pct=5.0,
            min_trading_days=3,
            account_type=AccountType.PERSONAL
        )
        test_session.add_all([prop_account, personal_account])
        test_session.commit()

        # Query by PROP_FIRM
        prop_firm_accounts = test_session.query(PropFirmAccount).filter(
            PropFirmAccount.account_type == AccountType.PROP_FIRM
        ).all()

        assert len(prop_firm_accounts) == 1
        assert prop_firm_accounts[0].firm_name == "PropFirm1"

    def test_query_by_account_type_personal(self, test_session: Session):
        """Test querying accounts by PERSONAL account_type."""
        # Create accounts
        prop_account = PropFirmAccount(
            firm_name="PropFirm1",
            account_id="33333",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5,
            account_type=AccountType.PROP_FIRM
        )
        personal_account = PropFirmAccount(
            firm_name="MyPersonal",
            account_id="44444",
            daily_loss_limit_pct=2.0,
            hard_stop_buffer_pct=0.5,
            target_profit_pct=5.0,
            min_trading_days=3,
            account_type=AccountType.PERSONAL
        )
        test_session.add_all([prop_account, personal_account])
        test_session.commit()

        # Query by PERSONAL
        personal_accounts = test_session.query(PropFirmAccount).filter(
            PropFirmAccount.account_type == AccountType.PERSONAL
        ).all()

        assert len(personal_accounts) == 1
        assert personal_accounts[0].firm_name == "MyPersonal"

    def test_account_type_indexed(self, test_session: Session):
        """Test that account_type column exists and is queryable."""
        account = PropFirmAccount(
            firm_name="IndexedFirm",
            account_id="55555",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5
        )
        test_session.add(account)
        test_session.commit()

        # Query using the indexed column
        result = test_session.query(PropFirmAccount).filter(
            PropFirmAccount.account_type == AccountType.PROP_FIRM
        ).first()

        assert result is not None
        assert result.firm_name == "IndexedFirm"
