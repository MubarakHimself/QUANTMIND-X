"""
Strategy Template Storage and Service

Handles template CRUD operations and storage for the fast-track
event workflow (Story 8.3).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.strategy.template_models import (
    StrategyTemplate,
    StrategyType,
    EventType,
    RiskProfile,
    DEFAULT_PARAMS,
    DEFAULT_FAST_TRACK_CONFIG,
)

logger = logging.getLogger(__name__)

# Template storage path
TEMPLATE_DIR = Path("./data/strategy_templates")
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


class TemplateStorage:
    """Storage layer for strategy templates."""

    @staticmethod
    def _get_template_path(template_id: str) -> Path:
        return TEMPLATE_DIR / f"{template_id}.json"

    @staticmethod
    def _get_index_path() -> Path:
        return TEMPLATE_DIR / "index.json"

    def save(self, template: StrategyTemplate) -> StrategyTemplate:
        """Save template to storage."""
        path = self._get_template_path(template.id)

        # Update timestamp
        template.updated_at = datetime.now()

        with open(path, "w") as f:
            json.dump(template.model_dump(mode="json"), f, indent=2, default=str)

        # Update index
        self._update_index(template, add=True)

        logger.info(f"Saved template: {template.name} ({template.id})")
        return template

    def get(self, template_id: str) -> Optional[StrategyTemplate]:
        """Get template by ID."""
        path = self._get_template_path(template_id)

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        return StrategyTemplate(**data)

    def get_all(self) -> List[StrategyTemplate]:
        """Get all templates."""
        templates = []

        for path in TEMPLATE_DIR.glob("*.json"):
            if path.name == "index.json":
                continue

            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    templates.append(StrategyTemplate(**data))
            except Exception as e:
                logger.warning(f"Failed to load template {path}: {e}")

        return sorted(templates, key=lambda t: t.created_at)

    def delete(self, template_id: str) -> bool:
        """Delete template."""
        path = self._get_template_path(template_id)

        if not path.exists():
            return False

        path.unlink()
        self._update_index(template_id, remove=True)

        logger.info(f"Deleted template: {template_id}")
        return True

    def search(
        self,
        event_type: Optional[EventType] = None,
        strategy_type: Optional[StrategyType] = None,
        risk_profile: Optional[RiskProfile] = None,
    ) -> List[StrategyTemplate]:
        """Search templates by filters."""
        templates = self.get_all()

        filtered = []

        for template in templates:
            if not template.is_active:
                continue

            if event_type and event_type not in template.applicable_events:
                continue

            if strategy_type and template.strategy_type != strategy_type:
                continue

            if risk_profile and template.risk_profile != risk_profile:
                continue

            filtered.append(template)

        return filtered

    def _update_index(self, template: StrategyTemplate, add: bool = True, remove: bool = False):
        """Update template index."""
        index_path = self._get_index_path()

        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"templates": []}

        if add:
            if template.id not in [t["id"] for t in index["templates"]]:
                index["templates"].append({
                    "id": template.id,
                    "name": template.name,
                    "strategy_type": template.strategy_type.value,
                    "created_at": template.created_at.isoformat(),
                })

        if remove:
            index["templates"] = [t for t in index["templates"] if t["id"] != template.id]

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)


class TemplateService:
    """Service layer for template operations."""

    def __init__(self):
        self.storage = TemplateStorage()

    def get_all_templates(self) -> List[StrategyTemplate]:
        """Get all active templates."""
        return self.storage.get_all()

    def get_template(self, template_id: str) -> Optional[StrategyTemplate]:
        """Get template by ID."""
        return self.storage.get(template_id)

    def create_template(
        self,
        name: str,
        description: str,
        strategy_type: StrategyType,
        applicable_events: List[EventType],
        risk_profile: RiskProfile,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StrategyTemplate:
        """Create a new template with defaults."""
        # Use default parameters for strategy type if not provided
        if parameters is None:
            parameters = DEFAULT_PARAMS.get(strategy_type, {})

        # Use default fast-track config for strategy type
        fast_track_config = DEFAULT_FAST_TRACK_CONFIG.get(strategy_type, {})

        template = StrategyTemplate(
            name=name,
            description=description,
            strategy_type=strategy_type,
            applicable_events=applicable_events,
            risk_profile=risk_profile,
            avg_deployment_time=fast_track_config.get("max_deployment_time_minutes", 15),
            parameters=parameters,
            symbols=symbols or ["EURUSD", "GBPUSD", "USDJPY"],
            timeframes=timeframes or ["M15", "H1", "H4"],
            fast_track_config=fast_track_config,
        )

        return self.storage.save(template)

    def update_template(self, template_id: str, **updates) -> Optional[StrategyTemplate]:
        """Update template fields."""
        template = self.storage.get(template_id)

        if not template:
            return None

        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)

        return self.storage.save(template)

    def delete_template(self, template_id: str) -> bool:
        """Delete template."""
        return self.storage.delete(template_id)

    def search_templates(
        self,
        event_type: Optional[EventType] = None,
        strategy_type: Optional[StrategyType] = None,
        risk_profile: Optional[RiskProfile] = None,
    ) -> List[StrategyTemplate]:
        """Search templates with filters."""
        return self.storage.search(event_type, strategy_type, risk_profile)

    def initialize_default_templates(self):
        """Initialize default templates if none exist."""
        existing = self.storage.get_all()

        if existing:
            logger.info(f"Found {len(existing)} existing templates")
            return

        logger.info("Creating default template library...")

        # Template 1: News Event Breakout
        self.create_template(
            name="News Event Breakout",
            description="Strategy for trading breakouts immediately after high-impact news events",
            strategy_type=StrategyType.NEWS_EVENT_BREAKOUT,
            applicable_events=[EventType.HIGH_IMPACT_NEWS, EventType.CENTRAL_BANK],
            risk_profile=RiskProfile.CONSERVATIVE,
        )

        # Template 2: Range Expansion
        self.create_template(
            name="Range Expansion",
            description="Strategy for trading range expansion after volatility normalization",
            strategy_type=StrategyType.RANGE_EXPANSION,
            applicable_events=[EventType.ECONOMIC_DATA, EventType.GEOPOLITICAL],
            risk_profile=RiskProfile.MODERATE,
        )

        # Template 3: Volatility Spike
        self.create_template(
            name="Volatility Spike",
            description="Strategy for capturing momentum during high volatility periods",
            strategy_type=StrategyType.VOLATILITY_SPIKE,
            applicable_events=[EventType.HIGH_IMPACT_NEWS, EventType.CENTRAL_BANK],
            risk_profile=RiskProfile.AGGRESSIVE,
        )

        logger.info("Default template library initialized with 3 templates")


# Singleton instance
_template_service: Optional[TemplateService] = None


def get_template_service() -> TemplateService:
    """Get template service singleton."""
    global _template_service
    if _template_service is None:
        _template_service = TemplateService()
        # Initialize default templates on first access
        _template_service.initialize_default_templates()
    return _template_service