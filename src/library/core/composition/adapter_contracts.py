"""
QuantMindLib V1 — Adapter Interface Contracts

Defines structural protocol interfaces for adapters using Python's typing.Protocol.
Any class implementing the required methods satisfies the protocol (structural subtyping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.library.core.domain.feature_vector import FeatureVector
    from src.library.core.domain.market_context import MarketContext
    from src.library.core.domain.order_flow_signal import OrderFlowSignal
    from src.library.core.domain.pattern_signal import PatternSignal
    from src.library.core.domain.trade_intent import TradeIntent
    from src.library.core.domain.execution_directive import ExecutionDirective


@runtime_checkable
class IMarketDataAdapter(Protocol):
    """
    Structural interface for market data feed adapters.
    Any class with these methods satisfies IMarketDataAdapter.

    Methods:
        get_feature_vector: Retrieve the feature vector for a given symbol.
        get_market_context: Retrieve the current market context snapshot.
        get_order_flow: Retrieve the order flow signal for a given symbol.
        get_pattern: Retrieve the pattern signal for a given symbol.
        is_connected: Return True if the data feed is currently connected.
    """

    def get_feature_vector(self, symbol: str) -> FeatureVector: ...

    def get_market_context(self) -> MarketContext: ...

    def get_order_flow(self, symbol: str) -> OrderFlowSignal: ...

    def get_pattern(self, symbol: str) -> PatternSignal: ...

    def is_connected(self) -> bool: ...


@runtime_checkable
class IExecutionAdapter(Protocol):
    """
    Structural interface for execution adapters.

    Methods:
        submit_trade: Submit a trade intent and return an execution directive.
        cancel_order: Cancel an order by its ID. Returns True on success.
        get_position: Retrieve the current position size for a symbol.
        is_connected: Return True if the execution channel is currently connected.
    """

    def submit_trade(self, intent: TradeIntent) -> ExecutionDirective: ...

    def cancel_order(self, order_id: str) -> bool: ...

    def get_position(self, symbol: str) -> float: ...

    def is_connected(self) -> bool: ...


@runtime_checkable
class IRiskAdapter(Protocol):
    """
    Structural interface for risk management adapters.

    Methods:
        evaluate_risk: Evaluate whether a trade intent is permissible given the
            current market context. Returns True if the risk check passes.
        get_risk_envelope: Retrieve the risk envelope for a given bot ID.
            Returns the risk envelope (approximated as Any at the Protocol level
            to avoid circular imports with the domain layer).
        is_connected: Return True if the risk engine is currently connected.
    """

    def evaluate_risk(self, intent: TradeIntent, context: MarketContext) -> bool: ...

    def get_risk_envelope(self, bot_id: str) -> Any: ...

    def is_connected(self) -> bool: ...


__all__ = ["IMarketDataAdapter", "IExecutionAdapter", "IRiskAdapter"]
