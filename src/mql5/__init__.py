"""
MQL5 EA Code Generation Module

Generates MQL5 Expert Advisor code from TRD documents.
"""
from src.mql5.generator import MQL5Generator
from src.mql5.templates.ea_base_template import EA_TEMPLATE

__all__ = ["MQL5Generator", "EA_TEMPLATE"]
