import pytest
from unittest.mock import MagicMock, patch, contextmanager
from src.router.bot_cloner import BotCloner, CloneCandidate
from src.database.db_manager import DBManager


@pytest.fixture
def mock_db():
    """Create a mock database that properly mocks get_session() context manager."""
    db = MagicMock()
    
    # Create a mock session
    mock_session = MagicMock()
    
    # Mock session query chain for StrategyPerformance
    mock_query = MagicMock()
    mock_query.filter_by.return_value.first.return_value = None
    mock_session.query.return_value = mock_query
    
    # Mock get_session() as a context manager that yields mock_session
    @contextmanager
    def mock_get_session():
        yield mock_session
    
    db.get_session = mock_get_session
    
    return db, mock_session


def test_is_clone_eligible(mock_db):
    db, session = mock_db
    cloner = BotCloner(db)
    with patch.object(cloner, '_get_bot_metrics', return_value={
        'sharpe': 2.5, 'win_rate': 0.6, 'total_trades': 150, 'days_active': 45
    }):
        assert cloner.is_clone_eligible('test_bot')


def test_get_similar_symbols():
    cloner = BotCloner()
    symbols = cloner.get_similar_symbols('EURUSD')
    assert 'GBPUSD' in symbols


def test_clone_bot(mock_db):
    db, session = mock_db
    cloner = BotCloner(db)
    
    # Mock the _get_bot_metrics method to return valid metrics
    with patch.object(cloner, '_get_bot_metrics', return_value={
        'sharpe': 2.5, 'win_rate': 0.6, 'total_trades': 150, 'days_active': 45
    }):
        # Mock bot_registry to return a valid manifest
        mock_manifest = MagicMock()
        mock_manifest.symbols = ['EURUSD']
        mock_manifest.strategy_type = MagicMock()
        mock_manifest.frequency = MagicMock()
        mock_manifest.prop_firm_safe = True
        mock_manifest.preferred_conditions = None
        mock_manifest.preferred_broker_type = MagicMock()
        mock_manifest.min_capital_req = 100.0
        mock_manifest.tags = ['@test']
        mock_manifest.total_trades = 100
        mock_manifest.win_rate = 0.6
        mock_manifest.preferred_timeframe = MagicMock()
        mock_manifest.use_multi_timeframe = False
        mock_manifest.secondary_timeframes = []
        mock_manifest.max_positions = 1
        mock_manifest.max_daily_trades = 10
        
        cloner.bot_registry = MagicMock()
        cloner.bot_registry.get.return_value = mock_manifest
        cloner.bot_registry.register = MagicMock()
        
        # Clone bot - session.add and session.commit are already mocked
        clones = cloner.clone_bot('test_bot', ['GBPUSD'])
        
        # Verify clone was created
        assert len(clones) == 1
        
        # Verify session.add was called with BotCloneHistory
        assert session.add.called
        
        # Verify session.commit was called to persist the history
        assert session.commit.called
