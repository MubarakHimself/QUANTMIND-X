import pytest
from unittest.mock import MagicMock, patch
from src.router.bot_cloner import BotCloner, CloneCandidate
from src.database.db_manager import DBManager

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Mock session query chain for StrategyPerformance
    mock_query = MagicMock()
    mock_query.filter_by.return_value.first.return_value = None
    db.session.query.return_value = mock_query
    return db

def test_is_clone_eligible(mock_db):
    cloner = BotCloner(mock_db)
    with patch.object(cloner, '_get_bot_metrics', return_value={
        'sharpe': 2.5, 'win_rate': 0.6, 'total_trades': 150, 'days_active': 45
    }):
        assert cloner.is_clone_eligible('test_bot')

def test_get_similar_symbols():
    cloner = BotCloner()
    symbols = cloner.get_similar_symbols('EURUSD')
    assert 'GBPUSD' in symbols

def test_clone_bot(mock_db):
    cloner = BotCloner(mock_db)
    # Mock the _get_bot_metrics method to return valid metrics
    with patch.object(cloner, '_get_bot_metrics', return_value={
        'sharpe': 2.5, 'win_rate': 0.6, 'total_trades': 150, 'days_active': 45
    }):
        # Mock session.add and session.commit for recording clone history
        with patch.object(cloner.db.session, 'add'), patch.object(cloner.db.session, 'commit'):
            clones = cloner.clone_bot('test_bot', ['GBPUSD'])
            assert len(clones) == 1
