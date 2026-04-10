"""
Self-Verification Tests for Packet 11A: MicrostructureFeature ABC.
5 tests verifying abstract base class behavior.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.microstructure_base import MicrostructureFeature
from src.library.features.microstructure.volume_imbalance import VolumeImbalanceFeature
from src.library.features.microstructure.tick_activity import TickActivityFeature
from src.library.features.microstructure.depth import MultiLevelDepthFeature


class TestMicrostructureBaseAbstract:
    def test_cannot_instantiate_directly(self):
        """MicrostructureFeature is abstract — cannot be instantiated."""
        with pytest.raises(TypeError):
            MicrostructureFeature()

    def test_volume_imbalance_inherits_from_microstructure(self):
        """VolumeImbalanceFeature is a subclass of MicrostructureFeature."""
        feat = VolumeImbalanceFeature()
        assert isinstance(feat, MicrostructureFeature)

    def test_tick_activity_inherits_from_microstructure(self):
        """TickActivityFeature is a subclass of MicrostructureFeature."""
        feat = TickActivityFeature()
        assert isinstance(feat, MicrostructureFeature)

    def test_multi_level_depth_inherits_from_microstructure(self):
        """MultiLevelDepthFeature is a subclass of MicrostructureFeature."""
        feat = MultiLevelDepthFeature()
        assert isinstance(feat, MicrostructureFeature)

    def test_subclasses_implement_compute_and_batch(self):
        """All subclasses implement both compute() and compute_batch()."""
        # VolumeImbalanceFeature
        vif = VolumeImbalanceFeature()
        bar = {"open": 1.0850, "close": 1.0858, "volume": 1000.0}
        assert isinstance(vif.compute(bar), float)
        assert isinstance(vif.compute_batch([bar]), list)

        # TickActivityFeature
        taf = TickActivityFeature()
        bar2 = {"tick_count": 50}
        assert isinstance(taf.compute(bar2), float)
        assert isinstance(taf.compute_batch([bar2]), list)

        # MultiLevelDepthFeature
        mld = MultiLevelDepthFeature()
        bar3 = {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0}
        assert isinstance(mld.compute(bar3), dict)
        assert isinstance(mld.compute_batch([bar3]), list)
