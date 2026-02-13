import pytest
from unittest.mock import MagicMock
from src.agents.paper_trading_validator import PaperTradingValidator
from src.agents.quantcode import paper_deployment_node  # etc.

@pytest.fixture
def validator():
    return PaperTradingValidator()

def test_meets_promotion_criteria(validator):
    metrics = {'sharpe': 1.8, 'win_rate': 0.6}
    assert validator.meets_promotion_criteria(metrics)

def test_paper_deployment_node():
    from src.agents.state import QuantCodeState
    state = QuantCodeState(messages=[], strategy_plan='test', code_implementation='test')
    result = paper_deployment_node(state)
    assert 'paper_agent_id' in result
