"""
pytest configuration and fixtures for Enhanced Kelly tests.

Provides fixtures and test utilities for the Enhanced Kelly position sizing test suite.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.position_sizing.enhanced_kelly import (
    EnhancedKellyCalculator,
    KellyResult,
    EnhancedKellyConfig
)
from src.position_sizing.kelly_analyzer import (
    KellyStatisticsAnalyzer,
    KellyParameters
)
from src.position_sizing.portfolio_kelly import (
    PortfolioKellyScaler,
    PortfolioStatus
)


@pytest.fixture
def standard_config():
    """Standard Enhanced Kelly configuration for testing."""
    return EnhancedKellyConfig(
        kelly_fraction=0.50,
        max_risk_pct=0.02,
        high_vol_threshold=1.3,
        low_vol_threshold=0.7,
        low_vol_boost=1.2,
        min_trade_history=30,
        min_lot_size=0.01,
        lot_step=0.01,
        max_lot_size=100.0,
        allow_zero_position=False,
        fallback_risk_pct=0.01
    )


@pytest.fixture
def ftmo_config():
    """FTMO Challenge preset configuration."""
    from src.position_sizing.kelly_config import PropFirmPresets
    return PropFirmPresets.ftmo_challenge()


@pytest.fixture
def the5ers_config():
    """The5%ers preset configuration."""
    from src.position_sizing.kelly_config import PropFirmPresets
    return PropFirmPresets.the5ers()


@pytest.fixture
def sample_trade_history() -> List[Dict]:
    """Sample trade history for testing."""
    return [
        {"profit": 500.0},   # Win
        {"profit": -200.0},  # Loss
        {"profit": 600.0},   # Win
        {"profit": -150.0},  # Loss
        {"profit": 450.0},   # Win
        {"profit": -180.0},  # Loss
        {"profit": 550.0},   # Win
        {"profit": -220.0},  # Loss
        {"profit": 700.0},   # Win
        {"profit": -190.0},  # Loss
        {"profit": 480.0},   # Win
        {"profit": -210.0},  # Loss
        {"profit": 520.0},   # Win
        {"profit": -170.0},  # Loss
        {"profit": 650.0},   # Win
        {"profit": -160.0},  # Loss
        {"profit": 580.0},   # Win
        {"profit": -200.0},  # Loss
        {"profit": 620.0},   # Win
        {"profit": -185.0},  # Loss
        {"profit": 510.0},   # Win
        {"profit": -195.0},  # Loss
        {"profit": 590.0},   # Win
        {"profit": -175.0},  # Loss
        {"profit": 680.0},   # Win
        {"profit": -205.0},  # Loss
        {"profit": 530.0},   # Win
        {"profit": -165.0},  # Loss
        {"profit": 610.0},   # Win
        {"profit": -215.0},  # Loss
        {"profit": 500.0},   # Win
        {"profit": -180.0},  # Loss
        {"profit": 570.0},   # Win
        {"profit": -200.0},  # Loss
        {"profit": 630.0},   # Win
        {"profit": -190.0},  # Loss
        {"profit": 540.0},   # Win
        {"profit": -170.0},  # Loss
        {"profit": 670.0},   # Win
        {"profit": -210.0},  # Loss
        {"profit": 520.0},   # Win
        {"profit": -185.0},  # Loss
    ]


@pytest.fixture
def winning_trade_history() -> List[Dict]:
    """Winning strategy trade history (60% win rate, 2:1 R:R)."""
    wins = [{"profit": 400.0} for _ in range(30)]  # 30 wins
    losses = [{"profit": -200.0} for _ in range(20)]  # 20 losses
    import random
    combined = wins + losses
    random.shuffle(combined)
    return combined


@pytest.fixture
def losing_trade_history() -> List[Dict]:
    """Losing strategy trade history (40% win rate, 1:1 R:R)."""
    wins = [{"profit": 200.0} for _ in range(20)]  # 20 wins
    losses = [{"profit": -200.0} for _ in range(30)]  # 30 losses
    import random
    combined = wins + losses
    random.shuffle(combined)
    return combined


@pytest.fixture
def insufficient_trade_history() -> List[Dict]:
    """Insufficient trade history (< 10 trades)."""
    return [
        {"profit": 500.0},
        {"profit": -200.0},
        {"profit": 600.0},
        {"profit": -150.0},
        {"profit": 450.0},
        {"profit": -180.0},
        {"profit": 550.0},
        {"profit": -220.0},
    ]


@pytest.fixture
def empty_trade_history() -> List[Dict]:
    """Empty trade history."""
    return []


@pytest.fixture
def sample_market_state():
    """Sample market state for testing."""
    return {
        "current_atr": 0.0012,  # 12 pips
        "average_atr": 0.0010,  # 10 pips (ATR ratio = 1.2)
        "volatility": 0.008,    # 0.8%
    }


@pytest.fixture
def high_volatility_state():
    """High volatility market state."""
    return {
        "current_atr": 0.0020,  # 20 pips
        "average_atr": 0.0010,  # 10 pips (ATR ratio = 2.0)
        "volatility": 0.025,    # 2.5%
    }


@pytest.fixture
def low_volatility_state():
    """Low volatility market state."""
    return {
        "current_atr": 0.0005,  # 5 pips
        "average_atr": 0.0010,  # 10 pips (ATR ratio = 0.5)
        "volatility": 0.003,    # 0.3%
    }


@pytest.fixture
def kelly_calculator(standard_config):
    """Enhanced Kelly calculator instance."""
    return EnhancedKellyCalculator(standard_config)


@pytest.fixture
def kelly_analyzer():
    """Kelly statistics analyzer instance."""
    return KellyStatisticsAnalyzer(min_trades=30)


@pytest.fixture
def portfolio_scaler():
    """Portfolio Kelly scaler instance."""
    return PortfolioKellyScaler(
        max_portfolio_risk_pct=0.03,
        correlation_adjustment=1.5
    )


@pytest.fixture
def account_balance():
    """Sample account balance."""
    return 10000.0


@pytest.fixture
def stop_loss_pips():
    """Sample stop loss in pips."""
    return 20.0


@pytest.fixture
def pip_value():
    """Sample pip value per standard lot."""
    return 10.0


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "kelly: mark test as Kelly calculation related"
    )
    config.addinivalue_line(
        "markers", "analyzer: mark test as statistics analyzer related"
    )
    config.addinivalue_line(
        "markers", "portfolio: mark test as portfolio scaling related"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case handling"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
