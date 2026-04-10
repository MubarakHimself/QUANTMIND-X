"""
QuantMindLib V1 — RegistryRecord Schema
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.library.core.types.enums import BotTier, RegistryStatus


class RegistryRecord(BaseModel):
    """Bot registration record in the library's own registry."""

    bot_id: str
    bot_spec_id: str
    status: RegistryStatus
    tier: BotTier
    registered_at_ms: int
    last_updated_ms: int
    owner: str
    variant_ids: List[str]
    deployed_at: Optional[str] = None

    model_config = BaseModel.model_config

    __all__ = ["RegistryRecord"]
