"""
Routing Matrix: Automatic Bot-to-Account Assignment

Routes bots to appropriate accounts based on their manifest properties.

From PDF: 
- Account A ($200): "Machine Gun" - HFT/Scalpers on RoboForex Prime
- Account B ($200): "Sniper" - ICT Macros, AMD, ORB on Exness Raw
- Prop Firm: Only prop_firm_safe bots

**Validates: PDF Requirements - Routing Matrix Dispatcher**
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from src.router.bot_manifest import (
    BotManifest, 
    BotRegistry, 
    StrategyType, 
    TradeFrequency, 
    BrokerType
)

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Trading account types."""
    MACHINE_GUN = "machine_gun"     # HFT/Scalper account (RoboForex)
    SNIPER = "sniper"               # Structural/ICT account (Exness)
    PROP_FIRM = "prop_firm"         # Prop firm challenge account
    CRYPTO = "crypto"               # Crypto exchange (Binance)
    DEMO = "demo"                   # Demo/paper trading


@dataclass
class AccountConfig:
    """Configuration for a trading account."""
    account_id: str
    account_type: AccountType
    broker_name: str
    account_number: str
    max_positions: int = 10
    max_daily_trades: int = 500
    capital_allocation: float = 200.0
    is_active: bool = True
    is_demo: bool = False
    
    # Acceptance criteria
    accepts_strategies: List[StrategyType] = field(default_factory=list)
    accepts_frequencies: List[TradeFrequency] = field(default_factory=list)
    requires_prop_safe: bool = False
    
    # Current state
    current_positions: int = 0
    daily_trades: int = 0
    current_pnl: float = 0.0


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    bot_id: str
    assigned_account: Optional[str]
    is_approved: bool
    rejection_reason: Optional[str] = None
    priority_score: float = 0.0


class RoutingMatrix:
    """
    Routes bots to appropriate accounts based on manifest properties.
    
    Two-Front War Strategy from PDF:
    - Account A: "Machine Gun" - 300+ trades/day, HFT scalpers
    - Account B: "Sniper" - 5 trades/day, structural bots
    """
    
    def __init__(self, bot_registry: Optional[BotRegistry] = None):
        self.bot_registry = bot_registry or BotRegistry()
        self._accounts: Dict[str, AccountConfig] = {}
        self._routing_cache: Dict[str, str] = {}  # bot_id -> account_id
        
        # Initialize default accounts from PDF
        self._init_default_accounts()
    
    def _init_default_accounts(self) -> None:
        """Initialize accounts from PDF configuration."""
        
        # Account A: Machine Gun (RoboForex Prime)
        self.register_account(AccountConfig(
            account_id="account_a_machine_gun",
            account_type=AccountType.MACHINE_GUN,
            broker_name="RoboForex",
            account_number="PRIME_001",
            max_positions=20,
            max_daily_trades=500,
            capital_allocation=200.0,
            accepts_strategies=[StrategyType.SCALPER, StrategyType.HFT],
            accepts_frequencies=[TradeFrequency.HFT, TradeFrequency.HIGH]
        ))
        
        # Account B: Sniper (Exness Raw)
        self.register_account(AccountConfig(
            account_id="account_b_sniper",
            account_type=AccountType.SNIPER,
            broker_name="Exness",
            account_number="RAW_001",
            max_positions=10,
            max_daily_trades=20,
            capital_allocation=200.0,
            accepts_strategies=[StrategyType.STRUCTURAL, StrategyType.SWING],
            accepts_frequencies=[TradeFrequency.LOW, TradeFrequency.MEDIUM]
        ))
        
        # Demo Account (for testing)
        self.register_account(AccountConfig(
            account_id="demo_account",
            account_type=AccountType.DEMO,
            broker_name="Demo",
            account_number="DEMO_001",
            max_positions=50,
            max_daily_trades=1000,
            capital_allocation=10000.0,
            is_demo=True,
            accepts_strategies=list(StrategyType),
            accepts_frequencies=list(TradeFrequency)
        ))
    
    def register_account(self, config: AccountConfig) -> None:
        """Register a trading account."""
        self._accounts[config.account_id] = config
        logger.info(f"Registered account: {config.account_id} ({config.broker_name})")
    
    def get_account(self, account_id: str) -> Optional[AccountConfig]:
        """Get account by ID."""
        return self._accounts.get(account_id)
    
    def list_accounts(self) -> List[AccountConfig]:
        """List all registered accounts."""
        return list(self._accounts.values())
    
    def route_bot(self, manifest: BotManifest) -> RoutingDecision:
        """
        Route a bot to the appropriate account.
        
        Routing logic from PDF:
        1. HFT/Scalpers → Machine Gun account
        2. Structural/ICT → Sniper account
        3. Check prop_firm_safe flag for prop firm accounts
        4. Match broker preference (RAW_ECN vs STANDARD)
        """
        
        best_account = None
        best_score = -1
        rejection_reasons = []
        
        for account in self._accounts.values():
            if not account.is_active:
                continue
            
            score, reason = self._calculate_compatibility_score(manifest, account)
            
            if reason:
                rejection_reasons.append(f"{account.account_id}: {reason}")
                continue
            
            if score > best_score:
                best_score = score
                best_account = account
        
        if best_account:
            decision = RoutingDecision(
                bot_id=manifest.bot_id,
                assigned_account=best_account.account_id,
                is_approved=True,
                priority_score=best_score
            )
            self._routing_cache[manifest.bot_id] = best_account.account_id
            logger.info(f"Routed {manifest.bot_id} → {best_account.account_id} (score: {best_score:.2f})")
        else:
            decision = RoutingDecision(
                bot_id=manifest.bot_id,
                assigned_account=None,
                is_approved=False,
                rejection_reason="; ".join(rejection_reasons)
            )
            logger.warning(f"No account found for {manifest.bot_id}: {decision.rejection_reason}")
        
        return decision
    
    def _calculate_compatibility_score(
        self, 
        manifest: BotManifest, 
        account: AccountConfig
    ) -> tuple[float, Optional[str]]:
        """
        Calculate compatibility score between bot and account.
        
        Returns (score, rejection_reason).
        Score 0-100, higher is better.
        """
        score = 50.0  # Base score
        
        # Check strategy type compatibility
        if manifest.strategy_type not in account.accepts_strategies:
            return 0, f"Strategy {manifest.strategy_type.value} not accepted"
        score += 20
        
        # Check frequency compatibility
        if manifest.frequency not in account.accepts_frequencies:
            return 0, f"Frequency {manifest.frequency.value} not accepted"
        score += 15
        
        # Check prop firm safety
        if account.requires_prop_safe and not manifest.prop_firm_safe:
            return 0, "Not prop firm safe"
        
        # Check capacity
        if account.current_positions >= account.max_positions:
            return 0, "Max positions reached"
        
        if account.daily_trades >= account.max_daily_trades:
            return 0, "Max daily trades reached"
        
        # Check capital requirements
        if manifest.min_capital_req > account.capital_allocation:
            return 0, f"Insufficient capital (needs ${manifest.min_capital_req})"
        
        # Broker preference bonus
        if manifest.preferred_broker_type == BrokerType.RAW_ECN:
            if "roboforex" in account.broker_name.lower():
                score += 10
        elif manifest.preferred_broker_type == BrokerType.STANDARD:
            if "exness" in account.broker_name.lower():
                score += 10
        
        # Demo penalty (prefer live accounts)
        if account.is_demo:
            score -= 30
        
        return score, None
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary of current routing state."""
        summary = {
            "total_accounts": len(self._accounts),
            "active_accounts": sum(1 for a in self._accounts.values() if a.is_active),
            "cached_routes": len(self._routing_cache),
            "accounts": {}
        }
        
        for account_id, account in self._accounts.items():
            bots = [
                bid for bid, aid in self._routing_cache.items() 
                if aid == account_id
            ]
            summary["accounts"][account_id] = {
                "type": account.account_type.value,
                "broker": account.broker_name,
                "positions": f"{account.current_positions}/{account.max_positions}",
                "daily_trades": f"{account.daily_trades}/{account.max_daily_trades}",
                "assigned_bots": bots,
                "is_demo": account.is_demo
            }
        
        return summary
    
    def route_all_bots(self) -> List[RoutingDecision]:
        """Route all registered bots to accounts."""
        decisions = []
        
        for manifest in self.bot_registry.list_all():
            decision = self.route_bot(manifest)
            decisions.append(decision)
        
        approved = sum(1 for d in decisions if d.is_approved)
        logger.info(f"Routed {approved}/{len(decisions)} bots successfully")
        
        return decisions
    
    def get_account_for_bot(self, manifest: "BotManifest") -> Optional[str]:
        """
        Get the assigned account for a bot.
        
        Convenience method for Strategy Router during dispatch.
        Uses cache if available, otherwise routes the bot.
        
        Args:
            manifest: Bot manifest
            
        Returns:
            Account ID if routed successfully, None otherwise
        """
        # Check cache first
        if manifest.bot_id in self._routing_cache:
            return self._routing_cache[manifest.bot_id]
        
        # Route the bot
        decision = self.route_bot(manifest)
        return decision.assigned_account if decision.is_approved else None


# Global routing matrix instance
_global_routing_matrix: Optional[RoutingMatrix] = None


def get_routing_matrix() -> RoutingMatrix:
    """Get or create the global routing matrix instance."""
    global _global_routing_matrix
    if _global_routing_matrix is None:
        _global_routing_matrix = RoutingMatrix()
    return _global_routing_matrix
