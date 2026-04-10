"""
QuantMindLib V1 — Feature Registry Bootstrap

Singleton registry containing all V1 feature modules.
Lazy initialization on first access.
"""
from __future__ import annotations

import threading

from src.library.features.registry import FeatureRegistry

# ---------------------------------------------------------------------------
# Module-level singleton with double-check locking
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_registry: FeatureRegistry | None = None


def _build_registry() -> FeatureRegistry:
    """Build and populate the feature registry with all V1 feature modules."""
    reg = FeatureRegistry()

    # === Indicators ===
    from src.library.features.indicators import (
        RSIFeature,
        ATRFeature,
        MACDFeature,
        VWAPFeature,
    )

    reg.register(RSIFeature(), family="indicators")       # feature_id: "indicators/rsi_14"
    reg.register(ATRFeature(), family="indicators")      # feature_id: "indicators/atr_14"
    reg.register(MACDFeature(), family="indicators")     # feature_id: "indicators/macd"
    reg.register(VWAPFeature(), family="indicators")     # feature_id: "indicators/vwap"

    # === Volume ===
    from src.library.features.volume import (
        RVOLFeature,
        MFIFeature,
        VolumeProfileFeature,
    )

    reg.register(RVOLFeature(), family="volume")         # feature_id: "volume/rvol"
    reg.register(MFIFeature(), family="volume")          # feature_id: "volume/mfi_14"
    reg.register(VolumeProfileFeature(), family="volume")  # feature_id: "volume/profile"

    # === OrderFlow ===
    from src.library.features.orderflow import (
        SpreadBehaviorFeature,
        DOMPressureFeature,
        DepthThinningFeature,
    )

    reg.register(SpreadBehaviorFeature(), family="orderflow")  # feature_id: "orderflow/spread_behavior"
    reg.register(DOMPressureFeature(), family="orderflow")    # feature_id: "orderflow/dom_pressure"
    reg.register(DepthThinningFeature(), family="orderflow")  # feature_id: "orderflow/depth_thinning"

    # === Session ===
    from src.library.features.session import (
        SessionDetectorFeature,
        SessionBlackoutFeature,
    )

    reg.register(SessionDetectorFeature(), family="session")  # feature_id: "session/detector"
    reg.register(SessionBlackoutFeature(), family="session")  # feature_id: "session/blackout"

    # === Transforms ===
    from src.library.features.transforms import (
        NormalizeTransform,
        RollingWindowTransform,
        ResampleTransform,
    )

    reg.register(NormalizeTransform(), family="transforms")     # feature_id: "transforms/normalize_value"
    reg.register(RollingWindowTransform(), family="transforms")  # feature_id: "transforms/rolling_value_20"
    reg.register(ResampleTransform(), family="transforms")     # feature_id: "transforms/resample_M1_to_M5"

    # === Microstructure ===
    from src.library.features.microstructure import (
        SpreadStateFeature,
        TopOfBookPressureFeature,
        AggressionProxyFeature,
        VolumeImbalanceFeature,
        TickActivityFeature,
        MultiLevelDepthFeature,
        AbsorptionProxyFeature,
        BreakoutPressureProxyFeature,
        LiquidityStressProxyFeature,
    )

    reg.register(SpreadStateFeature(), family="microstructure")           # feature_id: "microstructure/spread_state"
    reg.register(TopOfBookPressureFeature(), family="microstructure")    # feature_id: "microstructure/tob_pressure"
    reg.register(AggressionProxyFeature(), family="microstructure")      # feature_id: "microstructure/aggression"
    reg.register(VolumeImbalanceFeature(), family="microstructure")      # feature_id: "microstructure/volume_imbalance"
    reg.register(TickActivityFeature(), family="microstructure")         # feature_id: "microstructure/tick_activity"
    reg.register(MultiLevelDepthFeature(), family="microstructure")      # feature_id: "microstructure/multi_level_depth"
    reg.register(AbsorptionProxyFeature(), family="microstructure")      # feature_id: "microstructure/absorption_proxy"
    reg.register(BreakoutPressureProxyFeature(), family="microstructure")  # feature_id: "microstructure/breakout_pressure_proxy"
    reg.register(LiquidityStressProxyFeature(), family="microstructure")   # feature_id: "microstructure/liquidity_stress_proxy"

    return reg


def get_default_registry() -> FeatureRegistry:
    """
    Return the lazy-initialized singleton FeatureRegistry.

    Thread-safe: uses RLock to handle concurrent first access.
    The singleton is created on first call. Subsequent calls return
    the same instance.
    """
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = _build_registry()
    return _registry