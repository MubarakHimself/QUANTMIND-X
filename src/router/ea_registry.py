"""
EA Registry Module

Manages Expert Advisor (EA) registration, configuration, and mode tracking.
Supports demo/live mode distinction with virtual balance tracking for demo EAs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EAMode(Enum):
    """EA trading mode enum."""
    DEMO = "demo"
    LIVE = "live"


@dataclass
class EAConfig:
    """
    EA Configuration dataclass.
    
    Attributes:
        ea_id: Unique EA identifier
        name: Human-readable EA name
        symbol: Primary trading symbol
        timeframe: Primary timeframe
        magic_number: MT5 magic number for order identification
        mode: Trading mode (demo or live)
        virtual_balance: Virtual balance for demo mode (default 1000.0)
        preferred_regime: Preferred market regime (optional)
        preferred_volatility: Preferred volatility level (optional)
        max_lot_size: Maximum lot size
        max_daily_loss_pct: Maximum daily loss percentage
        tags: List of tags (auto-adds '@demo' for demo mode EAs)
    """
    ea_id: str
    name: str
    symbol: str
    timeframe: str
    magic_number: int
    mode: EAMode = EAMode.LIVE
    virtual_balance: float = 1000.0
    preferred_regime: Optional[str] = None
    preferred_volatility: Optional[str] = None
    max_lot_size: float = 1.0
    max_daily_loss_pct: float = 5.0
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Auto-add @demo tag for demo mode EAs."""
        if self.mode == EAMode.DEMO and '@demo' not in self.tags:
            self.tags.append('@demo')
        elif self.mode == EAMode.LIVE and '@live' not in self.tags:
            self.tags.append('@live')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ea_id': self.ea_id,
            'name': self.name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'magic_number': self.magic_number,
            'mode': self.mode.value,
            'virtual_balance': self.virtual_balance,
            'preferred_regime': self.preferred_regime,
            'preferred_volatility': self.preferred_volatility,
            'max_lot_size': self.max_lot_size,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'tags': self.tags
        }


class EARegistry:
    """
    EA Registry for managing EA configurations and mode tracking.
    
    Features:
    - Register EAs with demo/live mode
    - Filter EAs by mode
    - Track virtual balances for demo EAs
    - Support EA promotion from demo to live
    """
    
    def __init__(self):
        """Initialize EA Registry."""
        self._eas: Dict[str, EAConfig] = {}
        logger.info("EARegistry initialized")
    
    def register(self, config: EAConfig) -> bool:
        """
        Register a new EA.
        
        Args:
            config: EA configuration
            
        Returns:
            True if registration successful
        """
        if config.ea_id in self._eas:
            logger.warning(f"EA {config.ea_id} already registered - updating")
        
        self._eas[config.ea_id] = config
        logger.info(f"Registered EA: {config.ea_id} (mode={config.mode.value}, symbol={config.symbol})")
        return True
    
    def unregister(self, ea_id: str) -> bool:
        """
        Unregister an EA.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            True if unregistration successful
        """
        if ea_id in self._eas:
            del self._eas[ea_id]
            logger.info(f"Unregistered EA: {ea_id}")
            return True
        return False
    
    def get(self, ea_id: str) -> Optional[EAConfig]:
        """
        Get EA configuration by ID.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            EAConfig if found, None otherwise
        """
        return self._eas.get(ea_id)
    
    def get_by_mode(self, mode: EAMode) -> List[EAConfig]:
        """
        Get all EAs filtered by mode.
        
        Args:
            mode: Trading mode to filter by
            
        Returns:
            List of EA configurations matching the mode
        """
        return [ea for ea in self._eas.values() if ea.mode == mode]
    
    def get_demo_eas(self) -> List[EAConfig]:
        """
        Get all demo mode EAs.
        
        Returns:
            List of demo EA configurations
        """
        return self.get_by_mode(EAMode.DEMO)
    
    def get_live_eas(self) -> List[EAConfig]:
        """
        Get all live mode EAs.
        
        Returns:
            List of live EA configurations
        """
        return self.get_by_mode(EAMode.LIVE)
    
    def promote_to_live(self, ea_id: str) -> bool:
        """
        Promote an EA from demo to live mode.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            True if promotion successful
        """
        ea = self.get(ea_id)
        if ea is None:
            logger.warning(f"Cannot promote EA {ea_id} - not found")
            return False
        
        if ea.mode == EAMode.LIVE:
            logger.info(f"EA {ea_id} is already in live mode")
            return True
        
        # Update mode
        ea.mode = EAMode.LIVE
        ea.tags = [t for t in ea.tags if t != '@demo']
        if '@live' not in ea.tags:
            ea.tags.append('@live')
        
        logger.info(f"Promoted EA {ea_id} from demo to live mode")
        return True
    
    def demote_to_demo(self, ea_id: str, virtual_balance: float = 1000.0) -> bool:
        """
        Demote an EA from live to demo mode.
        
        Args:
            ea_id: EA identifier
            virtual_balance: Initial virtual balance for demo mode
            
        Returns:
            True if demotion successful
        """
        ea = self.get(ea_id)
        if ea is None:
            logger.warning(f"Cannot demote EA {ea_id} - not found")
            return False
        
        # Update mode
        ea.mode = EAMode.DEMO
        ea.virtual_balance = virtual_balance
        ea.tags = [t for t in ea.tags if t != '@live']
        if '@demo' not in ea.tags:
            ea.tags.append('@demo')
        
        logger.info(f"Demoted EA {ea_id} from live to demo mode with virtual balance {virtual_balance}")
        return True
    
    def list_all(self) -> List[EAConfig]:
        """
        List all registered EAs.
        
        Returns:
            List of all EA configurations
        """
        return list(self._eas.values())
    
    def count(self) -> Dict[str, int]:
        """
        Count EAs by mode.

        Returns:
            Dictionary with counts by mode
        """
        return {
            'total': len(self._eas),
            'demo': len(self.get_demo_eas()),
            'live': len(self.get_live_eas())
        }

    def add_tag(self, ea_id: str, tag: str) -> bool:
        """
        Add a tag to an EA.

        Args:
            ea_id: EA identifier
            tag: Tag to add

        Returns:
            True if tag was added
        """
        ea = self._eas.get(ea_id)
        if ea:
            if tag not in ea.tags:
                ea.tags.append(tag)
                logger.info(f"Added tag '{tag}' to EA {ea_id}")
            return True
        return False

    def remove_tag(self, ea_id: str, tag: str) -> bool:
        """
        Remove a tag from an EA.

        Args:
            ea_id: EA identifier
            tag: Tag to remove

        Returns:
            True if tag was removed
        """
        ea = self._eas.get(ea_id)
        if ea and ea.tags:
            if tag in ea.tags:
                ea.tags.remove(tag)
                logger.info(f"Removed tag '{tag}' from EA {ea_id}")
                return True
        return False

    def get_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all EAs with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of EA dictionaries with the tag
        """
        return [
            {"ea_id": ea.ea_id, "name": ea.name, "tags": ea.tags}
            for ea in self._eas.values() if ea.tags and tag in ea.tags
        ]


# Global registry instance
_global_registry: Optional[EARegistry] = None


def get_ea_registry() -> EARegistry:
    """Get or create the global EA registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = EARegistry()
    return _global_registry