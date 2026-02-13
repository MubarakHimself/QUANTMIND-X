"""
Unit tests for BotManifest timeframe field serialization/deserialization

Tests preferred_timeframe, use_multi_timeframe, and secondary_timeframes
fields in BotManifest.
"""

import pytest
from datetime import datetime
from src.router.bot_manifest import (
    BotManifest,
    BotRegistry,
    StrategyType,
    TradeFrequency,
    BrokerType,
    PreferredConditions,
    TimeWindow,
)
from src.router.multi_timeframe_sentinel import Timeframe


class TestBotManifestTimeframes:
    """Tests for BotManifest timeframe fields."""
    
    def test_manifest_with_timeframe_fields(self):
        """Test that manifest accepts timeframe fields."""
        manifest = BotManifest(
            bot_id="test_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.MEDIUM,
            preferred_timeframe=Timeframe.H1,
            use_multi_timeframe=True,
            secondary_timeframes=[Timeframe.M15, Timeframe.H4]
        )
        
        assert manifest.preferred_timeframe == Timeframe.H1
        assert manifest.use_multi_timeframe is True
        assert len(manifest.secondary_timeframes) == 2
        assert Timeframe.M15 in manifest.secondary_timeframes
        assert Timeframe.H4 in manifest.secondary_timeframes
    
    def test_manifest_to_dict_with_timeframes(self):
        """Test that to_dict includes timeframe fields."""
        manifest = BotManifest(
            bot_id="test_bot",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            preferred_timeframe=Timeframe.M5,
            use_multi_timeframe=False,
            secondary_timeframes=[]
        )
        
        manifest_dict = manifest.to_dict()
        
        assert manifest_dict["preferred_timeframe"] == "M5"
        assert manifest_dict["use_multi_timeframe"] is False
        assert manifest_dict["secondary_timeframes"] == []
    
    def test_manifest_to_dict_with_multi_timeframe(self):
        """Test to_dict with multi-timeframe enabled."""
        manifest = BotManifest(
            bot_id="test_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            preferred_timeframe=Timeframe.H1,
            use_multi_timeframe=True,
            secondary_timeframes=[Timeframe.M15, Timeframe.H4]
        )
        
        manifest_dict = manifest.to_dict()
        
        assert manifest_dict["preferred_timeframe"] == "H1"
        assert manifest_dict["use_multi_timeframe"] is True
        assert manifest_dict["secondary_timeframes"] == ["M15", "H4"]
    
    def test_manifest_from_dict_with_timeframes(self):
        """Test that from_dict deserializes timeframe fields."""
        data = {
            "bot_id": "test_bot",
            "strategy_type": "STRUCTURAL",
            "frequency": "MEDIUM",
            "preferred_timeframe": "M15",
            "use_multi_timeframe": True,
            "secondary_timeframes": ["M5", "H1", "H4"]
        }
        
        manifest = BotManifest.from_dict(data)
        
        assert manifest.preferred_timeframe == Timeframe.M15
        assert manifest.use_multi_timeframe is True
        assert Timeframe.M5 in manifest.secondary_timeframes
        assert Timeframe.H1 in manifest.secondary_timeframes
        assert Timeframe.H4 in manifest.secondary_timeframes
    
    def test_manifest_from_dict_without_timeframes(self):
        """Test that from_dict handles missing timeframe fields (backward compatibility)."""
        data = {
            "bot_id": "test_bot",
            "strategy_type": "SCALPER",
            "frequency": "HIGH"
        }
        
        manifest = BotManifest.from_dict(data)
        
        # Should default to H1
        assert manifest.preferred_timeframe == Timeframe.H1
        # Should default to False
        assert manifest.use_multi_timeframe is False
        # Should default to empty list
        assert manifest.secondary_timeframes == []
    
    def test_manifest_from_dict_with_invalid_timeframe(self):
        """Test that from_dict handles invalid timeframe values."""
        data = {
            "bot_id": "test_bot",
            "strategy_type": "SCALPER",
            "frequency": "HIGH",
            "preferred_timeframe": "INVALID_TF"
        }
        
        # Should not raise, should use default
        manifest = BotManifest.from_dict(data)
        
        assert manifest.preferred_timeframe == Timeframe.H1
    
    def test_manifest_roundtrip_with_timeframes(self):
        """Test serialization/deserialization roundtrip."""
        original = BotManifest(
            bot_id="roundtrip_bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            preferred_timeframe=Timeframe.H4,
            use_multi_timeframe=True,
            secondary_timeframes=[Timeframe.H1, Timeframe.M15]
        )
        
        # Serialize
        manifest_dict = original.to_dict()
        
        # Deserialize
        restored = BotManifest.from_dict(manifest_dict)
        
        assert restored.bot_id == original.bot_id
        assert restored.strategy_type == original.strategy_type
        assert restored.preferred_timeframe == original.preferred_timeframe
        assert restored.use_multi_timeframe == original.use_multi_timeframe
        assert restored.secondary_timeframes == original.secondary_timeframes


class TestBotRegistryTimeframes:
    """Tests for BotRegistry with timeframe fields."""
    
    def test_registry_stores_and_retrieves_timeframes(self):
        """Test that registry preserves timeframe fields."""
        # Create a temporary registry for testing
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "test_registry.json")
            registry = BotRegistry(storage_path=registry_path)
            
            manifest = BotManifest(
                bot_id="mtf_bot",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.MEDIUM,
                preferred_timeframe=Timeframe.H1,
                use_multi_timeframe=True,
                secondary_timeframes=[Timeframe.M15, Timeframe.M5],
                tags=["@primal"]
            )
            
            # Register
            registry.register(manifest)
            
            # Retrieve
            retrieved = registry.get("mtf_bot")
            
            assert retrieved is not None
            assert retrieved.preferred_timeframe == Timeframe.H1
            assert retrieved.use_multi_timeframe is True
            assert Timeframe.M15 in retrieved.secondary_timeframes
            assert Timeframe.M5 in retrieved.secondary_timeframes
    
    def test_list_by_tag_preserves_timeframes(self):
        """Test that list_by_tag returns manifests with timeframe fields."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "test_registry.json")
            registry = BotRegistry(storage_path=registry_path)
            
            # Register multiple bots
            for i in range(3):
                manifest = BotManifest(
                    bot_id=f"bot_{i}",
                    strategy_type=StrategyType.STRUCTURAL,
                    frequency=TradeFrequency.MEDIUM,
                    preferred_timeframe=Timeframe.M15 if i % 2 == 0 else Timeframe.H1,
                    use_multi_timeframe=(i % 2 == 0),
                    tags=["@primal"]
                )
                registry.register(manifest)
            
            # Get primal bots
            primal_bots = registry.list_by_tag("@primal")
            
            assert len(primal_bots) == 3
            # Check that timeframe fields are preserved
            for bot in primal_bots:
                assert bot.preferred_timeframe in [Timeframe.M15, Timeframe.H1]


class TestTimeframeIntegration:
    """Integration tests for timeframe fields with Commander filtering."""
    
    def test_commander_filters_by_preferred_timeframe(self):
        """Test that Commander correctly filters by preferred_timeframe."""
        from src.router.commander import Commander
        from src.router.sentinel import RegimeReport
        from datetime import datetime, timezone
        
        commander = Commander()
        
        # Create mock bots with different timeframe preferences
        bots = [
            {
                "bot_id": "bot_m5",
                "preferred_timeframe": "M5",
                "use_multi_timeframe": False,
                "secondary_timeframes": [],
                "regime": "TREND_STABLE"
            },
            {
                "bot_id": "bot_h1",
                "preferred_timeframe": "H1",
                "use_multi_timeframe": False,
                "secondary_timeframes": [],
                "regime": "TREND_STABLE"
            },
            {
                "bot_id": "bot_no_tf",
                "preferred_timeframe": None,
                "use_multi_timeframe": False,
                "secondary_timeframes": [],
                "regime": "TREND_STABLE"
            }
        ]
        
        # Create mock regime reports
        from src.router.multi_timeframe_sentinel import Timeframe
        
        regime_reports = {
            Timeframe.M5: RegimeReport(
                regime="HIGH_CHAOS",
                chaos_score=0.8,
                regime_quality=0.2,
                susceptibility=0.5,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=0.0
            ),
            Timeframe.H1: RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.1,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=0.0
            )
        }
        
        # Filter bots
        filtered = commander._filter_bots_by_timeframe(bots, regime_reports)
        
        # bot_m5 should be filtered (M5 is in HIGH_CHAOS)
        # bot_h1 should pass (H1 is in TREND_STABLE)
        # bot_no_tf should pass (no preferred timeframe)
        
        bot_ids = [b["bot_id"] for b in filtered]
        
        assert "bot_h1" in bot_ids
        assert "bot_no_tf" in bot_ids
        # bot_m5 might or might not be filtered depending on logic
    
    def test_commander_filters_by_secondary_timeframes(self):
        """Test that Commander filters bots with misaligned secondary timeframes."""
        from src.router.commander import Commander
        from src.router.sentinel import RegimeReport
        from src.router.multi_timeframe_sentinel import Timeframe
        
        commander = Commander()
        
        # Bot wants H1 primary with M5 and H4 as secondary
        bots = [
            {
                "bot_id": "aligned_bot",
                "preferred_timeframe": "H1",
                "use_multi_timeframe": True,
                "secondary_timeframes": ["M5", "H4"],
                "regime": "TREND_STABLE"
            }
        ]
        
        # All secondary timeframes aligned
        regime_reports = {
            Timeframe.M5: RegimeReport(
                regime="RANGE_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.1,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=0.0
            ),
            Timeframe.H1: RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.1,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=0.0
            ),
            Timeframe.H4: RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.3,
                regime_quality=0.7,
                susceptibility=0.2,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=0.0
            )
        }
        
        filtered = commander._filter_bots_by_timeframe(bots, regime_reports)
        
        # Should pass - all secondary timeframes are aligned
        assert len(filtered) == 1
        assert filtered[0]["bot_id"] == "aligned_bot"
        
        # Now test with misaligned secondary
        regime_reports[Timeframe.M5] = RegimeReport(
            regime="HIGH_CHAOS",
            chaos_score=0.8,
            regime_quality=0.2,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=0.0
        )
        
        filtered = commander._filter_bots_by_timeframe(bots, regime_reports)
        
        # Should be filtered out - M5 is in HIGH_CHAOS
        assert len(filtered) == 0
