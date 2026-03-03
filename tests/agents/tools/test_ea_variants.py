"""
Tests for EA Variant Creation (Vanilla and Spiced EA)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.agents.tools.ea_lifecycle import EALifecycleManager


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    output_path = Path(temp_dir) / "output" / "expert_advisors"
    output_path.mkdir(parents=True)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def ea_manager(temp_workspace):
    """Create EALifecycleManager instance with temp workspace."""
    return EALifecycleManager(base_path=temp_workspace)


class TestEAVariants:
    """Test EA variant creation from TRD."""

    def test_create_both_ea_variants(self, ea_manager):
        """Should create both vanilla and spiced EA from TRD."""
        trd_variants = {
            "vanilla": {"strategy_name": "TestVanilla", "entry_rules": ["rule1"]},
            "spiced": {"strategy_name": "TestSpiced", "entry_rules": ["rule1", "rule2"], "articles": []}
        }
        result = ea_manager.create_ea_from_trd(trd_variants, create_variants="both")
        assert "vanilla" in result
        assert "spiced" in result
        assert result["vanilla"]["ea_type"] == "vanilla"
        assert result["spiced"]["ea_type"] == "spiced"

    def test_create_vanilla_only(self, ea_manager):
        """Should create only vanilla EA when specified."""
        trd_variants = {
            "vanilla": {"strategy_name": "VanillaOnly", "entry_rules": ["rule1"]}
        }
        result = ea_manager.create_ea_from_trd(trd_variants, create_variants="vanilla")
        assert "vanilla" in result
        assert "spiced" not in result

    def test_create_spiced_only(self, ea_manager):
        """Should create only spiced EA when specified."""
        trd_variants = {
            "spiced": {"strategy_name": "SpicedOnly", "entry_rules": ["rule1"], "articles": ["article1"]}
        }
        result = ea_manager.create_ea_from_trd(trd_variants, create_variants="spiced")
        assert "spiced" in result
        assert "vanilla" not in result

    def test_vanilla_ea_has_required_sections(self, ea_manager):
        """Vanilla EA should have basic trading sections."""
        trd_variants = {
            "vanilla": {"strategy_name": "TestVanilla", "entry_rules": ["rule1"]}
        }
        result = ea_manager.create_ea_from_trd(trd_variants, create_variants="vanilla")
        ea_code = result["vanilla"]["code"]
        assert "OnInit" in ea_code
        assert "OnTick" in ea_code
        assert "OnDeinit" in ea_code

    def test_spiced_ea_has_required_sections(self, ea_manager):
        """Spiced EA should have advanced sections."""
        trd_variants = {
            "spiced": {"strategy_name": "TestSpiced", "entry_rules": ["rule1"], "articles": ["article1"]}
        }
        result = ea_manager.create_ea_from_trd(trd_variants, create_variants="spiced")
        ea_code = result["spiced"]["code"]
        assert "OnInit" in ea_code
        assert "OnTick" in ea_code

    def test_generate_mq5_creates_valid_code(self, ea_manager):
        """_generate_mq5 should create valid MQ5 code."""
        strategy_config = {
            "strategy_name": "MQ5Test",
            "entry_rules": ["rule1", "rule2"],
            "exit_rules": ["exit1"],
            "parameters": {"risk_percent": 2.0, "lot_size": 0.1}
        }
        ea_code = ea_manager._generate_mq5(strategy_config, ea_type="vanilla")
        assert "#property strict" in ea_code
        assert "MQ5Test" in ea_code

    def test_create_vanilla_ea(self, ea_manager):
        """_create_vanilla_ea should return proper structure."""
        config = {"strategy_name": "VanillaTest", "entry_rules": ["rule1"]}
        result = ea_manager._create_vanilla_ea(config)
        assert result["ea_type"] == "vanilla"
        assert result["strategy_name"] == "VanillaTest"
        assert "code" in result

    def test_create_spiced_ea(self, ea_manager):
        """_create_spiced_ea should return proper structure with articles."""
        config = {
            "strategy_name": "SpicedTest",
            "entry_rules": ["rule1"],
            "articles": ["article1", "article2"]
        }
        result = ea_manager._create_spiced_ea(config)
        assert result["ea_type"] == "spiced"
        assert result["strategy_name"] == "SpicedTest"
        assert "code" in result
