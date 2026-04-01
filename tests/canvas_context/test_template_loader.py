"""Test CanvasContextTemplate loader."""
import pytest
from pathlib import Path

from src.canvas_context.loader import (
    load_template,
    get_all_templates,
    get_canvas_list,
    clear_cache,
    normalize_canvas_name,
    SUPPORTED_CANVASES,
)
from src.canvas_context.types import CanvasContextTemplate, SkillIndexEntry


class TestTemplateLoader:
    """Tests for CanvasContextTemplate loading."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_load_template_risk(self):
        """Test loading risk canvas template."""
        template = load_template("risk")

        assert isinstance(template, CanvasContextTemplate)
        assert template.canvas == "risk"
        assert template.canvas_display_name == "Risk Management"
        assert "risk.*" in template.memory_scope
        assert template.department_mailbox == "risk_dept_mail"
        assert len(template.skill_index) > 0
        assert "risk_calculator" in template.required_tools

    def test_load_template_live_trading(self):
        """Test loading live_trading canvas template."""
        template = load_template("live_trading")

        assert isinstance(template, CanvasContextTemplate)
        assert template.canvas == "live_trading"
        assert template.canvas_display_name == "Live Trading"
        assert "trading.*" in template.memory_scope

    def test_load_template_accepts_live_trading_alias(self):
        """Test loading live-trading alias resolves to live_trading template."""
        template = load_template("live-trading")

        assert isinstance(template, CanvasContextTemplate)
        assert template.canvas == "live_trading"

    def test_load_template_accepts_shared_assets_alias(self):
        """Test loading shared-assets alias resolves to shared_assets template."""
        template = load_template("shared-assets")

        assert isinstance(template, CanvasContextTemplate)
        assert template.canvas == "shared_assets"

    def test_load_template_workshop(self):
        """Test loading workshop canvas template (default)."""
        template = load_template("workshop")

        assert isinstance(template, CanvasContextTemplate)
        assert template.canvas == "workshop"
        assert template.canvas_display_name == "Workshop"
        # Workshop should have wildcard memory scope
        assert "*" in template.memory_scope

    def test_load_template_unknown_canvas(self):
        """Test loading unknown canvas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown canvas"):
            load_template("unknown_canvas")

    def test_template_caching(self):
        """Test that templates are cached."""
        # Load twice
        template1 = load_template("risk")
        template2 = load_template("risk")

        # Same object due to caching
        assert template1 is template2

    def test_clear_cache(self):
        """Test cache clearing."""
        # Load and cache
        load_template("risk")
        clear_cache()

        # Should reload (different object)
        template = load_template("risk")
        assert isinstance(template, CanvasContextTemplate)

    def test_get_all_templates(self):
        """Test loading all templates."""
        templates = get_all_templates()

        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert "risk" in templates
        assert "workshop" in templates

    def test_get_canvas_list(self):
        """Test getting canvas list."""
        canvases = get_canvas_list()

        assert isinstance(canvases, list)
        assert len(canvases) > 0
        # Check structure
        for canvas in canvases:
            assert "id" in canvas
            assert "name" in canvas


class TestTemplateSchema:
    """Tests for CanvasContextTemplate schema validation."""

    def test_skill_index_parsing(self):
        """Test skill index is properly parsed."""
        template = load_template("risk")

        assert len(template.skill_index) > 0
        skill = template.skill_index[0]
        assert isinstance(skill, SkillIndexEntry)
        assert skill.id
        assert skill.path
        assert skill.trigger

    def test_memory_scope_list(self):
        """Test memory_scope is a list."""
        template = load_template("risk")

        assert isinstance(template.memory_scope, list)
        assert len(template.memory_scope) > 0

    def test_required_tools_list(self):
        """Test required_tools is a list."""
        template = load_template("risk")

        assert isinstance(template.required_tools, list)
        assert len(template.required_tools) > 0

    def test_max_identifiers_default(self):
        """Test max_identifiers has default value."""
        template = load_template("risk")

        assert template.max_identifiers > 0
        assert template.max_identifiers == 50  # Default from schema


class TestSupportedCanvases:
    """Tests for supported canvases configuration."""

    def test_supported_canvases_contains_all(self):
        """Test all expected canvases are supported."""
        expected = {
            "live_trading",
            "risk",
            "portfolio",
            "research",
            "development",
            "trading",
            "workshop",
            "flowforge",
            "shared_assets",
        }

        assert SUPPORTED_CANVASES == expected

    def test_all_templates_exist(self):
        """Test all supported canvases have template files."""
        for canvas in SUPPORTED_CANVASES:
            template = load_template(canvas)
            assert template.canvas == canvas

    def test_normalize_canvas_name_maps_frontend_aliases(self):
        """Test frontend kebab-case ids normalize to canonical template ids."""
        assert normalize_canvas_name("live-trading") == "live_trading"
        assert normalize_canvas_name("shared-assets") == "shared_assets"
        assert normalize_canvas_name("risk") == "risk"
