"""
MQL5 EA Templates Module
"""
from src.mql5.templates.ea_base_template import EA_TEMPLATE
from src.mql5.templates.schema import (
    StrategyTemplate,
    TemplateMatchResult,
    EventType,
    StrategyTypeTemplate,
    RiskProfile,
    TemplateParameter,
)
from src.mql5.templates.storage import TemplateStorage, get_template_storage
from src.mql5.templates.matcher import TemplateMatcher, get_template_matcher

__all__ = [
    "EA_TEMPLATE",
    "StrategyTemplate",
    "TemplateMatchResult",
    "EventType",
    "StrategyTypeTemplate",
    "RiskProfile",
    "TemplateParameter",
    "TemplateStorage",
    "get_template_storage",
    "TemplateMatcher",
    "get_template_matcher",
]
