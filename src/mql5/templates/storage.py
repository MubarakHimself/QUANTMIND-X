"""
Template Storage and Retrieval

Handles persistence and querying of strategy templates.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.mql5.templates.schema import (
    StrategyTemplate,
    TemplateMatchResult,
    EventType,
    StrategyTypeTemplate,
    RiskProfile,
)

logger = logging.getLogger(__name__)

# Template storage directory
TEMPLATE_DIR = Path(".quantmind/templates")


class TemplateStorage:
    """
    Storage and retrieval system for strategy templates.

    Uses JSON file storage for simplicity - in production could be SQLite.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or TEMPLATE_DIR
        self.index_file = self.storage_path / "template_index.json"
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_template_file(self, template_id: str) -> Path:
        """Get file path for a template."""
        return self.storage_path / f"{template_id}.json"

    def save(self, template: StrategyTemplate) -> StrategyTemplate:
        """Save a template to storage."""
        template.updated_at = datetime.now()

        template_file = self._get_template_file(template.id)
        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Update index
        self._update_index(template, add=True)

        logger.info(f"Template saved: {template.id} - {template.name}")
        return template

    def get(self, template_id: str) -> Optional[StrategyTemplate]:
        """Retrieve a template by ID."""
        template_file = self._get_template_file(template_id)

        if not template_file.exists():
            return None

        with open(template_file, "r") as f:
            data = json.load(f)

        return StrategyTemplate.from_dict(data)

    def get_all(self) -> List[StrategyTemplate]:
        """Retrieve all templates."""
        templates = []

        for template_file in self.storage_path.glob("*.json"):
            if template_file.name == "template_index.json":
                continue

            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    templates.append(StrategyTemplate.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")

        return sorted(templates, key=lambda t: t.created_at, reverse=True)

    def get_active(self) -> List[StrategyTemplate]:
        """Retrieve all active templates."""
        return [t for t in self.get_all() if t.is_active]

    def get_by_event_type(self, event_type: str) -> List[StrategyTemplate]:
        """Get templates applicable to a specific event type."""
        return [
            t for t in self.get_active()
            if event_type in t.applicable_events
        ]

    def get_by_symbol(self, symbol: str) -> List[StrategyTemplate]:
        """Get templates applicable to a specific symbol."""
        return [
            t for t in self.get_active()
            if symbol in t.applicable_symbols
        ]

    def delete(self, template_id: str) -> bool:
        """Delete a template by ID."""
        template = self.get(template_id)
        if not template:
            return False

        template_file = self._get_template_file(template_id)
        template_file.unlink()

        # Update index
        self._update_index(template, add=False)

        logger.info(f"Template deleted: {template_id}")
        return True

    def _update_index(self, template: StrategyTemplate, add: bool = True) -> None:
        """Update the template index file."""
        index = self._load_index()

        if add:
            index["templates"][template.id] = {
                "name": template.name,
                "strategy_type": template.strategy_type.value if template.strategy_type else None,
                "applicable_events": template.applicable_events,
                "created_at": template.created_at.isoformat() if template.created_at else None,
            }
        else:
            index["templates"].pop(template.id, None)

        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _load_index(self) -> Dict[str, Any]:
        """Load the template index."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)

        return {"templates": {}}

    def seed_default_templates(self) -> List[StrategyTemplate]:
        """Seed the database with default templates if empty."""
        existing = self.get_all()

        if len(existing) >= 3:
            return existing  # Already seeded

        default_templates = [
            # Template 1: News Event Breakout
            StrategyTemplate(
                name="News Event Breakout",
                strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
                applicable_events=[
                    EventType.HIGH_IMPACT_NEWS.value,
                    EventType.CENTRAL_BANK.value,
                    EventType.GEOPOLITICAL.value,
                ],
                applicable_symbols=["EURUSD", "GBPUSD", "USDJPY"],
                risk_profile=RiskProfile.CONSERVATIVE,
                avg_deployment_time=11,
                parameters={
                    "breakout_threshold_pips": 20,
                    "ma_fast": 20,
                    "ma_slow": 50,
                    "atr_period": 14,
                    "atr_multiplier": 2.0,
                    "session_mask": "UK,US",
                    "force_close_hour": 21,
                    "overnight_hold": False,
                    "max_spread_entry": 30,
                    "trailing_stop": True,
                    "trailing_distance": 30,
                },
                lot_sizing_multiplier=0.5,
                auto_expiry_hours=24,
                is_islamic_compliant=True,
            ),
            # Template 2: Range Expansion
            StrategyTemplate(
                name="Range Expansion Strategy",
                strategy_type=StrategyTypeTemplate.RANGE_EXPANSION,
                applicable_events=[
                    EventType.ECONOMIC_DATA.value,
                    EventType.CENTRAL_BANK.value,
                ],
                applicable_symbols=["EURUSD", "GBPUSD", "AUDUSD"],
                risk_profile=RiskProfile.CONSERVATIVE,
                avg_deployment_time=12,
                parameters={
                    "range_period_bars": 50,
                    "breakout_threshold": 0.5,  # % of ATR
                    "ma_period": 20,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "session_mask": "UK,US,Asia",
                    "force_close_hour": 21,
                    "overnight_hold": False,
                    "max_spread_entry": 25,
                    "use_trailing_stop": True,
                    "trailing_distance": 25,
                },
                lot_sizing_multiplier=0.5,
                auto_expiry_hours=24,
                is_islamic_compliant=True,
            ),
            # Template 3: Volatility Spike
            StrategyTemplate(
                name="Volatility Spike Capture",
                strategy_type=StrategyTypeTemplate.VOLATILITY_SPIKE,
                applicable_events=[
                    EventType.MARKET_SHOCK.value,
                    EventType.HIGH_IMPACT_NEWS.value,
                    EventType.GEOPOLITICAL.value,
                ],
                applicable_symbols=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
                risk_profile=RiskProfile.MODERATE,
                avg_deployment_time=10,
                parameters={
                    "atr_period": 14,
                    "volatility_threshold": 2.5,  # ATR multiplier
                    "fast_ema": 5,
                    "slow_ema": 20,
                    "grid_levels": 5,
                    "grid_spacing_pips": 30,
                    "session_mask": "UK,US",
                    "force_close_hour": 20,
                    "overnight_hold": False,
                    "max_spread_entry": 40,
                    "use_trailing_stop": True,
                    "trailing_distance": 40,
                },
                lot_sizing_multiplier=0.5,
                auto_expiry_hours=24,
                is_islamic_compliant=True,
            ),
        ]

        saved_templates = []
        for template in default_templates:
            saved = self.save(template)
            saved_templates.append(saved)

        logger.info(f"Seeded {len(saved_templates)} default templates")
        return saved_templates


# Module-level storage instance
_storage: Optional[TemplateStorage] = None


def get_template_storage() -> TemplateStorage:
    """Get the global template storage instance."""
    global _storage
    if _storage is None:
        _storage = TemplateStorage()
    return _storage