import pytest
from src.router.dynamic_bot_limits import DynamicBotLimiter

def test_get_max_bots():
    assert DynamicBotLimiter.get_max_bots(300) == 3
    assert DynamicBotLimiter.get_max_bots(700) == 5
    assert DynamicBotLimiter.get_max_bots(2000) == 10
    assert DynamicBotLimiter.get_max_bots(15000) == 30

def test_get_recommended_risk_per_bot():
    risk = DynamicBotLimiter.get_recommended_risk_per_bot(1000)
    assert risk == 0.6  # 3%/5 bots = 0.6%
