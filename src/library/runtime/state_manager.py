"""
QuantMindLib V1 -- BotStateManager
Manages runtime state: cached FeatureVectors and MarketContext per bot.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, PrivateAttr

from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.domain.market_context import MarketContext


class BotStateManager(BaseModel):
    """
    Thread-safe runtime state manager for one or more bots.
    Caches FeatureVector per bot and maintains current MarketContext.
    """

    _feature_vectors: Dict[str, FeatureVector] = PrivateAttr(default_factory=dict)
    _market_contexts: Dict[str, MarketContext] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    model_config = BaseModel.model_config

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_feature_vectors", {})
        object.__setattr__(self, "_market_contexts", {})
        object.__setattr__(self, "_lock", threading.RLock())

    def update_feature_vector(self, bot_id: str, fv: FeatureVector) -> None:
        """Update the cached FeatureVector for a bot."""
        with self._lock:
            self._feature_vectors[bot_id] = fv

    def get_feature_vector(self, bot_id: str) -> Optional[FeatureVector]:
        """Get the cached FeatureVector for a bot."""
        with self._lock:
            return self._feature_vectors.get(bot_id)

    def update_market_context(self, bot_id: str, ctx: MarketContext) -> None:
        """Update the cached MarketContext for a bot."""
        with self._lock:
            self._market_contexts[bot_id] = ctx

    def get_market_context(self, bot_id: str) -> Optional[MarketContext]:
        """Get the cached MarketContext for a bot."""
        with self._lock:
            return self._market_contexts.get(bot_id)

    def get_combined_state(
        self, bot_id: str
    ) -> Tuple[Optional[FeatureVector], Optional[MarketContext]]:
        """Get both FeatureVector and MarketContext for a bot."""
        with self._lock:
            return (
                self._feature_vectors.get(bot_id),
                self._market_contexts.get(bot_id),
            )

    def tick_update(
        self, bot_id: str, bar_data: Dict[str, Any], market_ctx: MarketContext
    ) -> Optional[FeatureVector]:
        """
        Called on each tick/bar update. Updates market context and returns
        the cached FeatureVector (or None if not yet computed).
        """
        with self._lock:
            self._market_contexts[bot_id] = market_ctx
            return self._feature_vectors.get(bot_id)

    def reset(self, bot_id: str) -> None:
        """Clear cached state for a bot."""
        with self._lock:
            self._feature_vectors.pop(bot_id, None)
            self._market_contexts.pop(bot_id, None)

    def reset_all(self) -> None:
        """Clear all cached state."""
        with self._lock:
            self._feature_vectors.clear()
            self._market_contexts.clear()

    def list_bots(self) -> List[str]:
        """List all bots with cached state."""
        with self._lock:
            return list(self._feature_vectors.keys())

    def has_stale_context(self, bot_id: str, threshold_ms: int = 5000) -> bool:
        """True if the market context for a bot is stale or absent."""
        ctx = self.get_market_context(bot_id)
        if ctx is None:
            return True
        return not ctx.is_fresh(threshold_ms)
