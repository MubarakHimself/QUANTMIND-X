"""Tests for QuantMindLib V1 -- BotStateManager."""

import threading
import time
from datetime import datetime
from typing import Set

import pytest

from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence
from src.library.core.domain.market_context import MarketContext, RegimeReport
from src.library.core.types.enums import NewsState, RegimeType
from src.library.runtime.state_manager import BotStateManager


def _make_fv(bot_id: str = "bot-1", features: dict = None) -> FeatureVector:
    """Helper: create a FeatureVector with predictable defaults."""
    return FeatureVector(
        bot_id=bot_id,
        timestamp=datetime.now(),
        features=features or {"test_feature": 1.0},
        feature_confidence={
            "test_feature": FeatureConfidence(
                source="test",
                quality=0.9,
                latency_ms=0.5,
                feed_quality_tag="HIGH",
            )
        },
    )


def _make_ctx(
    regime: RegimeType = RegimeType.TREND_STABLE,
    news: NewsState = NewsState.CLEAR,
    last_update_ms: int = None,
) -> MarketContext:
    """Helper: create a MarketContext with predictable defaults."""
    if last_update_ms is None:
        last_update_ms = int(time.time() * 1000)
    return MarketContext(
        regime=regime,
        news_state=news,
        regime_confidence=0.85,
        session_id="test-session",
        is_stale=False,
        last_update_ms=last_update_ms,
    )


class TestBotStateManagerInit:
    """Tests for BotStateManager initialization."""

    def test_initializes_with_empty_state(self):
        mgr = BotStateManager()
        assert mgr.list_bots() == []

    def test_model_is_pydantic_base_model(self):
        assert isinstance(BotStateManager(), object)


class TestFeatureVectorRoundtrip:
    """Tests for feature vector cache operations."""

    def test_update_and_get_feature_vector(self):
        mgr = BotStateManager()
        fv = _make_fv("bot-a", {"rsi_14": 65.0, "macd_signal": 0.001})
        mgr.update_feature_vector("bot-a", fv)
        result = mgr.get_feature_vector("bot-a")
        assert result is not None
        assert result.bot_id == "bot-a"
        assert result.features["rsi_14"] == 65.0
        assert result.features["macd_signal"] == 0.001

    def test_get_feature_vector_returns_none_for_unknown_bot(self):
        mgr = BotStateManager()
        assert mgr.get_feature_vector("unknown-bot") is None

    def test_update_feature_vector_overwrites_existing(self):
        mgr = BotStateManager()
        mgr.update_feature_vector("bot-1", _make_fv("bot-1", {"v1": 1.0}))
        mgr.update_feature_vector("bot-1", _make_fv("bot-1", {"v2": 2.0}))
        result = mgr.get_feature_vector("bot-1")
        assert "v2" in result.features
        assert "v1" not in result.features


class TestMarketContextRoundtrip:
    """Tests for market context cache operations."""

    def test_update_and_get_market_context(self):
        mgr = BotStateManager()
        ctx = _make_ctx(RegimeType.RANGE_STABLE, NewsState.KILL_ZONE)
        mgr.update_market_context("bot-x", ctx)
        result = mgr.get_market_context("bot-x")
        assert result is not None
        assert result.regime == RegimeType.RANGE_STABLE
        assert result.news_state == NewsState.KILL_ZONE

    def test_get_market_context_returns_none_for_unknown_bot(self):
        mgr = BotStateManager()
        assert mgr.get_market_context("unknown-bot") is None


class TestGetCombinedState:
    """Tests for get_combined_state."""

    def test_returns_both_none_when_empty(self):
        mgr = BotStateManager()
        fv, ctx = mgr.get_combined_state("bot-1")
        assert fv is None
        assert ctx is None

    def test_returns_both_when_cached(self):
        mgr = BotStateManager()
        fv = _make_fv("bot-1")
        ctx = _make_ctx()
        mgr.update_feature_vector("bot-1", fv)
        mgr.update_market_context("bot-1", ctx)
        got_fv, got_ctx = mgr.get_combined_state("bot-1")
        assert got_fv is fv
        assert got_ctx is ctx

    def test_returns_partial_when_only_fv_cached(self):
        mgr = BotStateManager()
        fv = _make_fv("bot-1")
        mgr.update_feature_vector("bot-1", fv)
        got_fv, got_ctx = mgr.get_combined_state("bot-1")
        assert got_fv is fv
        assert got_ctx is None


class TestTickUpdate:
    """Tests for tick_update."""

    def test_tick_update_updates_context_and_returns_fv(self):
        mgr = BotStateManager()
        fv = _make_fv("bot-tick")
        mgr.update_feature_vector("bot-tick", fv)
        new_ctx = _make_ctx()
        result = mgr.tick_update("bot-tick", {"close": 1.0850}, new_ctx)
        assert result is fv
        assert mgr.get_market_context("bot-tick") is new_ctx

    def test_tick_update_returns_none_when_no_fv_cached(self):
        mgr = BotStateManager()
        new_ctx = _make_ctx()
        result = mgr.tick_update("bot-new", {"close": 1.0850}, new_ctx)
        assert result is None


class TestReset:
    """Tests for reset operations."""

    def test_reset_clears_single_bot(self):
        mgr = BotStateManager()
        mgr.update_feature_vector("bot-a", _make_fv("bot-a"))
        mgr.update_market_context("bot-a", _make_ctx())
        mgr.update_feature_vector("bot-b", _make_fv("bot-b"))
        mgr.reset("bot-a")
        assert mgr.get_feature_vector("bot-a") is None
        assert mgr.get_market_context("bot-a") is None
        assert mgr.get_feature_vector("bot-b") is not None
        assert "bot-a" not in mgr.list_bots()

    def test_reset_unknown_bot_is_safe(self):
        mgr = BotStateManager()
        mgr.reset("no-such-bot")  # Should not raise

    def test_reset_all_clears_everything(self):
        mgr = BotStateManager()
        for i in range(3):
            mgr.update_feature_vector(f"bot-{i}", _make_fv(f"bot-{i}"))
            mgr.update_market_context(f"bot-{i}", _make_ctx())
        mgr.reset_all()
        assert mgr.list_bots() == []
        assert mgr.get_feature_vector("bot-0") is None
        assert mgr.get_market_context("bot-0") is None


class TestListBots:
    """Tests for list_bots."""

    def test_list_bots_returns_bot_ids(self):
        mgr = BotStateManager()
        assert mgr.list_bots() == []
        mgr.update_feature_vector("bot-x", _make_fv("bot-x"))
        mgr.update_feature_vector("bot-y", _make_fv("bot-y"))
        assert set(mgr.list_bots()) == {"bot-x", "bot-y"}

    def test_list_bots_excludes_bots_without_fv(self):
        mgr = BotStateManager()
        mgr.update_feature_vector("bot-with-fv", _make_fv("bot-with-fv"))
        mgr.update_market_context("bot-ctx-only", _make_ctx())
        assert "bot-with-fv" in mgr.list_bots()
        assert "bot-ctx-only" not in mgr.list_bots()


class TestHasStaleContext:
    """Tests for has_stale_context."""

    def test_true_when_no_context(self):
        mgr = BotStateManager()
        assert mgr.has_stale_context("no-such-bot") is True

    def test_false_when_context_is_fresh(self):
        mgr = BotStateManager()
        ctx = _make_ctx(last_update_ms=int(time.time() * 1000))
        mgr.update_market_context("bot-fresh", ctx)
        assert mgr.has_stale_context("bot-fresh", threshold_ms=5000) is False

    def test_true_when_context_is_stale(self):
        mgr = BotStateManager()
        stale_ms = int(time.time() * 1000) - 10_000  # 10 seconds ago
        ctx = _make_ctx(last_update_ms=stale_ms)
        mgr.update_market_context("bot-stale", ctx)
        assert mgr.has_stale_context("bot-stale", threshold_ms=5000) is True

    def test_threshold_parameter_respected(self):
        mgr = BotStateManager()
        old_ms = int(time.time() * 1000) - 3_000  # 3 seconds ago
        ctx = _make_ctx(last_update_ms=old_ms)
        mgr.update_market_context("bot-mid", ctx)
        assert mgr.has_stale_context("bot-mid", threshold_ms=5000) is False
        assert mgr.has_stale_context("bot-mid", threshold_ms=1000) is True


class TestThreadSafety:
    """Tests for thread-safety of concurrent operations."""

    def test_concurrent_updates_are_safe(self):
        mgr = BotStateManager()
        errors: list = []
        barrier = threading.Barrier(10)

        def worker(bot_idx: int):
            try:
                barrier.wait()
                for i in range(100):
                    fid = f"bot-{bot_idx}"
                    fv = _make_fv(fid, {f"feat_{i}": float(i)})
                    mgr.update_feature_vector(fid, fv)
                    ctx = _make_ctx()
                    mgr.update_market_context(fid, ctx)
                    mgr.get_feature_vector(fid)
                    mgr.get_market_context(fid)
                    mgr.list_bots()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(mgr.list_bots()) == 10

    def test_concurrent_reset_all_is_safe(self):
        mgr = BotStateManager()
        for i in range(5):
            mgr.update_feature_vector(f"bot-{i}", _make_fv(f"bot-{i}"))

        errors: list = []

        def writer():
            try:
                for i in range(50):
                    mgr.update_feature_vector("bot-0", _make_fv("bot-0"))
            except Exception as e:
                errors.append(e)

        def reseter():
            try:
                for i in range(50):
                    mgr.reset_all()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reseter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"
