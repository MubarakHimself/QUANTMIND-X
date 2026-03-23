"""
Tests for Strategy Template Library
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.mql5.templates.schema import (
    StrategyTemplate,
    TemplateMatchResult,
    EventType,
    StrategyTypeTemplate,
    RiskProfile,
    TemplateParameter,
)
from src.mql5.templates.storage import TemplateStorage
from src.mql5.templates.matcher import TemplateMatcher


class TestStrategyTemplate:
    """Tests for StrategyTemplate schema."""

    def test_create_template(self):
        """Test creating a strategy template."""
        template = StrategyTemplate(
            name="Test Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            applicable_events=[EventType.HIGH_IMPACT_NEWS.value],
            applicable_symbols=["EURUSD", "GBPUSD"],
            risk_profile=RiskProfile.CONSERVATIVE,
            avg_deployment_time=11,
            parameters={"test_param": "value"},
        )

        assert template.name == "Test Template"
        assert template.strategy_type == StrategyTypeTemplate.NEWS_EVENT_BREAKOUT
        assert template.risk_profile == RiskProfile.CONSERVATIVE
        assert template.avg_deployment_time == 11

    def test_template_to_dict(self):
        """Test template serialization to dict."""
        template = StrategyTemplate(
            name="Test Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            applicable_events=[EventType.HIGH_IMPACT_NEWS.value],
            applicable_symbols=["EURUSD"],
        )

        data = template.to_dict()

        assert data["name"] == "Test Template"
        assert data["strategy_type"] == "news_event_breakout"
        assert "id" in data
        assert "created_at" in data

    def test_template_from_dict(self):
        """Test template deserialization from dict."""
        data = {
            "id": "test-id-123",
            "name": "Test Template",
            "strategy_type": "news_event_breakout",
            "applicable_events": ["HIGH_IMPACT_NEWS"],
            "applicable_symbols": ["EURUSD"],
            "risk_profile": "conservative",
            "avg_deployment_time": 11,
            "parameters": {},
            "lot_sizing_multiplier": 0.5,
            "auto_expiry_hours": 24,
            "is_islamic_compliant": True,
            "is_active": True,
            "created_at": "2026-03-20T00:00:00",
            "updated_at": "2026-03-20T00:00:00",
            "author": "Test",
        }

        template = StrategyTemplate.from_dict(data)

        assert template.id == "test-id-123"
        assert template.name == "Test Template"
        assert template.risk_profile == RiskProfile.CONSERVATIVE


class TestTemplateStorage:
    """Tests for TemplateStorage."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        temp_dir = Path(tempfile.mkdtemp())
        storage = TemplateStorage(storage_path=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    def test_save_and_retrieve_template(self, temp_storage):
        """Test saving and retrieving a template."""
        template = StrategyTemplate(
            name="Test Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            applicable_events=[EventType.HIGH_IMPACT_NEWS.value],
            applicable_symbols=["EURUSD"],
        )

        saved = temp_storage.save(template)
        retrieved = temp_storage.get(saved.id)

        assert retrieved is not None
        assert retrieved.name == "Test Template"
        assert retrieved.id == saved.id

    def test_get_all_templates(self, temp_storage):
        """Test retrieving all templates."""
        # Save multiple templates
        for i in range(3):
            template = StrategyTemplate(
                name=f"Template {i}",
                strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            )
            temp_storage.save(template)

        templates = temp_storage.get_all()
        assert len(templates) == 3

    def test_delete_template(self, temp_storage):
        """Test deleting a template."""
        template = StrategyTemplate(
            name="Test Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
        )
        saved = temp_storage.save(template)

        success = temp_storage.delete(saved.id)
        assert success is True

        retrieved = temp_storage.get(saved.id)
        assert retrieved is None

    def test_seed_default_templates(self, temp_storage):
        """Test seeding default templates."""
        templates = temp_storage.seed_default_templates()

        assert len(templates) >= 3

        # Check template names
        names = [t.name for t in templates]
        assert "News Event Breakout" in names
        assert "Range Expansion Strategy" in names
        assert "Volatility Spike Capture" in names

    def test_get_active_templates(self, temp_storage):
        """Test filtering active templates."""
        template = StrategyTemplate(
            name="Active Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            is_active=True,
        )
        temp_storage.save(template)

        template_inactive = StrategyTemplate(
            name="Inactive Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
            is_active=False,
        )
        temp_storage.save(template_inactive)

        active = temp_storage.get_active()
        assert all(t.is_active for t in active)


class TestTemplateMatcher:
    """Tests for TemplateMatcher."""

    @pytest.fixture
    def storage_with_templates(self, tmp_path):
        """Create storage with seeded templates."""
        storage = TemplateStorage(storage_path=tmp_path)
        storage.seed_default_templates()
        return storage

    def test_match_high_impact_event(self, storage_with_templates, monkeypatch):
        """Test matching against HIGH impact news event."""
        # Mock storage
        monkeypatch.setattr(
            "src.mql5.templates.matcher.get_template_storage",
            lambda: storage_with_templates
        )

        matcher = TemplateMatcher()
        results = matcher.match_event(
            event_type=EventType.HIGH_IMPACT_NEWS.value,
            affected_symbols=["EURUSD", "GBPUSD"],
            impact_tier="HIGH",
        )

        assert len(results) > 0
        assert all(isinstance(r, TemplateMatchResult) for r in results)

    def test_match_central_bank_event(self, storage_with_templates, monkeypatch):
        """Test matching against central bank event."""
        monkeypatch.setattr(
            "src.mql5.templates.matcher.get_template_storage",
            lambda: storage_with_templates
        )

        matcher = TemplateMatcher()
        results = matcher.match_event(
            event_type=EventType.CENTRAL_BANK.value,
            affected_symbols=["EURUSD"],
            impact_tier="HIGH",
        )

        # Should find templates applicable to central bank events
        assert len(results) > 0

    def test_confidence_scoring(self, storage_with_templates, monkeypatch):
        """Test that confidence scoring works."""
        monkeypatch.setattr(
            "src.mql5.templates.matcher.get_template_storage",
            lambda: storage_with_templates
        )

        matcher = TemplateMatcher()
        results = matcher.match_event(
            event_type=EventType.HIGH_IMPACT_NEWS.value,
            affected_symbols=["EURUSD"],
            impact_tier="HIGH",
        )

        # Results should be sorted by confidence
        if len(results) > 1:
            assert results[0].confidence_score >= results[1].confidence_score

    def test_top_matches_with_threshold(self, storage_with_templates, monkeypatch):
        """Test filtering top matches by confidence threshold."""
        monkeypatch.setattr(
            "src.mql5.templates.matcher.get_template_storage",
            lambda: storage_with_templates
        )

        matcher = TemplateMatcher()
        results = matcher.get_top_matches(
            event_type=EventType.HIGH_IMPACT_NEWS.value,
            affected_symbols=["EURUSD"],
            impact_tier="HIGH",
            min_confidence=0.3,
            limit=2,
        )

        assert len(results) <= 2
        assert all(r.confidence_score >= 0.3 for r in results)


class TestTemplateMatchResult:
    """Tests for TemplateMatchResult."""

    def test_match_result_to_dict(self):
        """Test match result serialization."""
        template = StrategyTemplate(
            name="Test Template",
            strategy_type=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
        )

        result = TemplateMatchResult(
            template=template,
            confidence_score=0.85,
            match_factors={"event_match": 1.0, "symbol_match": 0.7},
            estimated_deployment_time=11,
        )

        data = result.to_dict()

        assert data["template_name"] == "Test Template"
        assert data["confidence_score"] == 0.85
        assert data["estimated_deployment_time"] == 11
        assert "template" in data