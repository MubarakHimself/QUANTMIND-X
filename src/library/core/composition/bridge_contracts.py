"""
QuantMindLib V1 — Bridge Interface Contracts
CONTRACT-023

Defines structural protocol interfaces for bridges using Python's typing.Protocol.
Bridges are the glue between adapters and domain objects: they transform adapter
output into domain objects and route domain intents to adapters.

Any class implementing the required methods satisfies the protocol
(structural subtyping via @runtime_checkable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.library.core.domain.market_context import MarketContext
    from src.library.core.domain.sentinel_state import SentinelState, HMMState
    from src.library.core.domain.feature_vector import FeatureVector
    from src.library.core.domain.order_flow_signal import OrderFlowSignal
    from src.library.core.domain.pattern_signal import PatternSignal
    from src.library.core.domain.trade_intent import TradeIntent
    from src.library.core.domain.execution_directive import ExecutionDirective


@runtime_checkable
class ISentinelBridge(Protocol):
    """
    Bridge from market data adapters to Sentinel domain state.
    Transforms raw adapter output into SentinelState for the regime system.

    Methods:
        to_sentinel_state: Transform market context + order flow + pattern into
            a SentinelState object.
        to_hmm_state: Derive the HMM hidden state from a feature vector.
        get_current_regime: Return the current regime name string.
        is_healthy: Return True if the Sentinel bridge is operational.
    """

    def to_sentinel_state(
        self,
        market_context: MarketContext,
        order_flow: OrderFlowSignal,
        pattern: PatternSignal,
    ) -> SentinelState: ...

    def to_hmm_state(self, features: FeatureVector) -> HMMState: ...

    def get_current_regime(self) -> str: ...

    def is_healthy(self) -> bool: ...


@runtime_checkable
class IExecutionBridge(Protocol):
    """
    Bridge from domain trade intents to execution adapters.
    Transforms TradeIntent -> ExecutionDirective via the execution pipeline.

    Methods:
        to_directive: Convert a trade intent + market context into an
            execution directive.
        validate_directive: Verify that an execution directive is well-formed
            and permissible.
        route_directive: Send a directive to a specific execution target and
            return the routing result.
        is_healthy: Return True if the execution bridge is operational.
    """

    def to_directive(
        self,
        intent: TradeIntent,
        context: MarketContext,
    ) -> ExecutionDirective: ...

    def validate_directive(self, directive: ExecutionDirective) -> bool: ...

    def route_directive(
        self,
        directive: ExecutionDirective,
        target: str,
    ) -> dict[str, Any]: ...

    def is_healthy(self) -> bool: ...


@runtime_checkable
class IFeatureBridge(Protocol):
    """
    Bridge from market data adapters to feature extraction.
    Transforms raw market data into normalized FeatureVectors.

    Methods:
        to_feature_vector: Convert market context + raw adapter data into a
            structured FeatureVector.
        get_feature_schema: Return the list of feature names the bridge
            produces (schema declaration).
        is_healthy: Return True if the feature bridge is operational.
    """

    def to_feature_vector(
        self,
        market_context: MarketContext,
        raw_data: dict[str, Any],
    ) -> FeatureVector: ...

    def get_feature_schema(self) -> list[str]: ...

    def is_healthy(self) -> bool: ...


@runtime_checkable
class IRiskBridge(Protocol):
    """
    Bridge from trade intents to risk adapters.
    Evaluates intent against current regime and routes to IRiskAdapter.

    Methods:
        evaluate_intent: Return True if the trade intent passes risk checks
            given the current market context.
        get_envelope: Retrieve the risk envelope dict for a bot given its
            intent.  Returns a dict (approximated to avoid circular imports
            with the domain layer at the Protocol level).
        is_healthy: Return True if the risk bridge is operational.
    """

    def evaluate_intent(
        self,
        intent: TradeIntent,
        context: MarketContext,
    ) -> bool: ...

    def get_envelope(
        self,
        bot_id: str,
        intent: TradeIntent,
    ) -> dict[str, Any]: ...

    def is_healthy(self) -> bool: ...


__all__ = [
    "ISentinelBridge",
    "IExecutionBridge",
    "IFeatureBridge",
    "IRiskBridge",
]
