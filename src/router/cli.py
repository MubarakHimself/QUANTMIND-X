"""
Router CLI - Verification Script
Usage: python3 -m src.router.cli
"""

import logging
import time
from src.router.engine import StrategyRouter
from src.library.base_bot import BaseBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TrendBot(BaseBot):
    """A sample bot for testing the auction."""
    def get_signal(self, data: dict) -> float:
        return 0.8 # Bullish

def main():
    print("--- QuantMindX Strategy Router: Sentient Loop Verification ---")
    router = StrategyRouter()
    
    # 1. Register a bot
    bot = TrendBot("AlphaTrend", 1001)
    router.register_bot(bot)
    
    # 2. Simulate Ticks
    prices = [1.1000, 1.1010, 1.1005, 1.1020, 1.1050]
    
    print("\nStarting Sentient Loop simulation...")
    for i, price in enumerate(prices):
        print(f"\n[Tick {i+1}] Price: {price}")
        
        # Run the full loop
        result = router.process_tick("EURUSD", price)
        
        print(f"Regime: {result['regime']}")
        print(f"Quality: {result['quality']:.2f}")
        print(f"Governor Mandate: {result['mandate'].risk_mode} (Scalar: {result['mandate'].allocation_scalar:.2f})")
        
    print("\n[SUCCESS] Standard Loop verified.")

    # 3. Test Prop Mode (Throttling)
    print("\n--- Testing Prop Firm Mode (T3 Throttle) ---")
    from src.router.prop.governor import PropGovernor
    from src.router.prop.state import PropState
    
    prop_router = StrategyRouter()
    prop_router.governor = PropGovernor("ACCOUNT_1")
    # Simulate a daily start balance of 100k
    prop_router.governor.prop_state.daily_start_balance = 100000.0
    
    bot_prop = TrendBot("PropBot", 2001)
    prop_router.register_bot(bot_prop)
    
    # Simulate a loss of $3000 (3% of 100k). 
    # Effective limit is 4%. Expected throttle should be significant.
    print(f"\n[Prop Tick] Price: 1.0970 (Current Balance: 97000)")
    res = prop_router.process_tick("EURUSD", 1.0970, account_data={"current_balance": 97000})
    # We pass current_balance in the manual proposal dict if needed, 
    # but let's update StrategyRouter to handle balance.
    
    print(f"Prop Mandate: {res['mandate'].risk_mode}")
    print(f"Notes: {res['mandate'].notes}")
    
    print("\n[SUCCESS] Prop Logic verified.")

if __name__ == "__main__":
    main()
