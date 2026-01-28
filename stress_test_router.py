"""
Strategy Router Stress Test
Simulates diverse market behaviors to verify physics-aware adaptability.
"""

import numpy as np
import logging
from src.router.engine import StrategyRouter
from src.library.base_bot import BaseBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TestBot(BaseBot):
    def get_signal(self, data: dict) -> float:
        return 1.0 # Always bullish for this test

def run_scenario(name: str, prices: list, initial_balance: float = 100000.0):
    print(f"\n>>> Scenario: {name}")
    router = StrategyRouter()
    bot = TestBot("StressBot", 9999)
    router.register_bot(bot)
    
    current_balance = initial_balance
    
    for i, p in enumerate(prices):
        # Every 10 ticks, simulate a small profit/loss to update account balance
        if i > 0 and i % 10 == 0:
            current_balance *= (1 + np.random.normal(0, 0.001))
            
        res = router.process_tick("BTCPERT", p, account_data={"current_balance": current_balance})
        
        # Log every 20 ticks or if regime changes
        if i % 25 == 0 or res['quality'] < 0.5:
            qual = res['quality']
            scalar = res['mandate'].allocation_scalar
            print(f"Tick {i:3d} | Price: {p:8.2f} | Qual: {qual:.2f} | Risk Scalar: {scalar:.2f} | {res['regime']}")

def main():
    # 1. Steady Trend
    steady_trend = np.linspace(50000, 55000, 100)
    run_scenario("Steady Trend", steady_trend.tolist())
    
    # 2. Flash Crash with Chaos
    flash_crash = np.concatenate([
        np.linspace(55000, 55000, 50),      # Flat
        np.linspace(55000, 45000, 10),      # CRASH
        np.random.normal(45000, 500, 20),   # Chaos/Volatility
        np.linspace(45000, 46000, 20)       # Slow Recovery
    ])
    run_scenario("Flash Crash & Chaos", flash_crash.tolist())
    
    # 3. High-Frequency Noise (White Noise)
    noise = np.random.normal(50000, 50, 100)
    run_scenario("High Noise (Range)", noise.tolist())

if __name__ == "__main__":
    main()
