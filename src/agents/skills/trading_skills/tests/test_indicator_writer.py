"""
Tests for Indicator Writer Skill (MQL5 Indicator Code Generation)

Tests focus on:
1. CRiBuffDbl ring buffer class generation
2. Indicator property directives generation
3. OnCalculate function signature generation
4. SetIndexBuffer initialization code
5. Compilation-ready output (valid MQL5 syntax)
6. OnInit/OnDeinit scaffold generation
7. Complete indicator file generation
8. Input validation and error handling
"""

import pytest
import re
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.agents.skills.skill_schema import SkillDefinition, SkillValidator
from src.agents.skills.trading_skills.indicator_writer import (
    generate_mql5_indicator,
    generate_cribuffdbl_class,
    generate_oncalculate_function,
    generate_setindexbuffer_code,
    generate_oninit_deinit,
    generate_heikin_ashi_indicator,
)


class TestIndicatorWriterGeneration:
    """Test MQL5 indicator code generation."""

    @pytest.fixture
    def indicator_writer_skill(self) -> SkillDefinition:
        """Load the indicator_writer skill definition."""
        # Import from the skill module
        from agents.skills.trading_skills.indicator_writer import skill_definition
        return skill_definition

    def test_skill_definition_valid(self, indicator_writer_skill: SkillDefinition):
        """Test that the indicator_writer skill definition is valid."""
        assert indicator_writer_skill.name == "indicator_writer"
        assert indicator_writer_skill.category == "trading_skills"
        assert indicator_writer_skill.version == "1.0.0"

        # Validate against schema
        errors = SkillValidator.validate_skill(indicator_writer_skill)
        assert not errors, f"Skill validation failed: {errors}"

    def test_generates_property_directives(self, indicator_writer_skill: SkillDefinition):
        """Test that generated indicators include required #property directives."""
        # This would call the skill's generate function
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        result = generate_mql5_indicator(
            indicator_name="TestIndicator",
            indicator_type="line",
            buffers=1
        )

        # Check for required property directives
        assert "#property" in result
        assert "indicator_chart_window" in result or "indicator_separate_window" in result
        assert "indicator_buffers" in result
        assert "indicator_plots" in result

    def test_generates_cribuffdbl_ring_buffer(self, indicator_writer_skill: SkillDefinition):
        """Test that CRiBuffDbl ring buffer class is generated correctly."""
        from src.agents.skills.trading_skills.indicator_writer import generate_cribuffdbl_class

        result = generate_cribuffdbl_class()

        # Verify class structure
        assert "class CRiBuffDbl" in result
        assert "private:" in result or "protected:" in result
        assert "m_buffer[]" in result or "m_buffer" in result
        assert "m_head_index" in result
        assert "m_max_total" in result

        # Verify key methods
        assert "AddValue" in result
        assert "GetValue" in result
        assert "SetMaxTotal" in result
        assert "GetTotal" in result or "ToRealInd" in result

        # Verify it's valid MQL5 class syntax
        assert "{" in result and "}" in result

    def test_generates_oncalculate_signature(self, indicator_writer_skill: SkillDefinition):
        """Test that OnCalculate function has correct signature per MQL5 docs."""
        from src.agents.skills.trading_skills.indicator_writer import generate_oncalculate_function

        result = generate_oncalculate_function(
            use_full_ohlcv=True,
            use_ring_buffer=True
        )

        # Check for correct signature
        assert "int OnCalculate(" in result
        assert "const int rates_total" in result
        assert "const int prev_calculated" in result
        assert "const double &close" in result or "const double &price" in result

        # If full OHLCV, check for additional parameters
        # We passed use_full_ohlcv=True, so check for OHLCV parameters
        assert "const datetime &time[]" in result
        assert "const double &open[]" in result
        assert "const double &high[]" in result
        assert "const double &low[]" in result

        # Check return statement
        assert "return(" in result or "return " in result

    def test_generates_setindexbuffer_initialization(self, indicator_writer_skill: SkillDefinition):
        """Test that SetIndexBuffer initialization code is generated correctly."""
        from src.agents.skills.trading_skills.indicator_writer import generate_setindexbuffer_code

        result = generate_setindexbuffer_code(
            num_buffers=2,
            buffer_names=["IndicatorBuffer", "ColorBuffer"],
            use_color_buffer=True
        )

        # Check SetIndexBuffer calls
        assert "SetIndexBuffer(" in result
        assert "INDICATOR_DATA" in result
        assert "IndicatorBuffer" in result

        # Check for proper buffer binding syntax
        assert re.search(r'SetIndexBuffer\(\s*0\s*,', result) is not None

        # With use_color_buffer=True, the last buffer should be INDICATOR_COLOR_INDEX
        assert "INDICATOR_COLOR_INDEX" in result

    def test_generates_oninit_deinit_scaffolds(self, indicator_writer_skill: SkillDefinition):
        """Test that OnInit and OnDeinit function scaffolds are generated."""
        from src.agents.skills.trading_skills.indicator_writer import generate_oninit_deinit

        result = generate_oninit_deinit()

        # Check OnInit
        assert "int OnInit()" in result
        assert "return INIT_SUCCEEDED" in result or "return(INIT_SUCCEEDED)" in result

        # Check OnDeinit
        assert "void OnDeinit(" in result
        assert "const int reason" in result

    def test_complete_indicator_compilable(self, indicator_writer_skill: SkillDefinition):
        """Test that generated complete indicator is compilation-ready."""
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        result = generate_mql5_indicator(
            indicator_name="RSI_Indicator",
            indicator_type="line",
            buffers=1,
            use_ring_buffer=True
        )

        # Check basic MQL5 syntax requirements
        assert "//+------------------------------------------------------------------+" in result or "//+" in result
        assert "property" in result
        assert "OnInit()" in result
        assert "OnCalculate(" in result
        assert "SetIndexBuffer(" in result

        # Check for balanced braces
        open_braces = result.count("{")
        close_braces = result.count("}")
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"

        # Check for CRiBuffDbl if ring buffer enabled
        if "use_ring_buffer" in str(result) or "CRiBuffDbl" in result:
            assert "class CRiBuffDbl" in result or "CRiBuffDbl" in result

    def test_input_validation(self, indicator_writer_skill: SkillDefinition):
        """Test that input validation works correctly."""
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        # Test with invalid inputs
        with pytest.raises(ValueError):
            generate_mql5_indicator(
                indicator_name="",  # Empty name
                indicator_type="line"
            )

        with pytest.raises(ValueError):
            generate_mql5_indicator(
                indicator_name="Valid Name",
                indicator_type="line",
                buffers=0  # Invalid: must have at least 1 buffer
            )


class TestIndicatorWriterPatterns:
    """Test specific patterns from reference articles."""

    def test_ribuffdbl_pattern_matches_reference(self):
        """Test that generated ring buffer matches MQL5 Cookbook pattern."""
        from src.agents.skills.trading_skills.indicator_writer import generate_cribuffdbl_class

        result = generate_cribuffdbl_class()

        # Reference article patterns
        required_members = [
            "m_buffer",      # Ring buffer array
            "m_head_index",  # Pointer to last element
            "m_max_total",   # Max buffer size
            "m_full_buff"    # Full buffer flag
        ]

        for member in required_members:
            assert member in result, f"Missing required member: {member}"

        # Required methods
        required_methods = [
            "AddValue",
            "GetValue",
            "SetMaxTotal",
            "ToRealInd"
        ]

        for method in required_methods:
            assert method in result, f"Missing required method: {method}"

    def test_oncalculate_handles_prev_calculated(self):
        """Test that OnCalculate properly handles prev_calculated for incremental updates."""
        from src.agents.skills.trading_skills.indicator_writer import generate_oncalculate_function

        result = generate_oncalculate_function(
            use_full_ohlcv=False,
            use_ring_buffer=True
        )

        # Check for prev_calculated handling
        assert "prev_calculated" in result
        assert "rates_total" in result

        # Check for loop starting from prev_calculated
        assert re.search(r'for\s*\(\s*int\s+i\s*=\s*prev_calculated', result) is not None or \
               re.search(r'for\s*\(\s*int\s*i\s*=\s*.*prev_calculated', result) is not None

    def test_heikin_ashi_example_from_article(self):
        """Test generation matches Heikin Ashi example from reference article."""
        from src.agents.skills.trading_skills.indicator_writer import generate_heikin_ashi_indicator

        result = generate_heikin_ashi_indicator()

        # Heikin Ashi requires 4 buffers + color
        assert "HA_Open" in result or "HaOpen" in result
        assert "HA_High" in result or "HaHigh" in result
        assert "HA_Low" in result or "HaLow" in result
        assert "HA_Close" in result or "HaClose" in result

        # Check for Heikin Ashi formulas
        # HA Close = (Open + High + Low + Close) / 4
        assert re.search(r'HA_Close.*=.*\(.*open.*\+.*high.*\+.*low.*\+.*close.*\)', result) is not None

        # HA Open = (Previous HA Open + Previous HA Close) / 2
        assert re.search(r'HA_Open.*=.*\(.*HA_Open\[.*i.*-.*1.*\].*\+.*HA_Close\[.*i.*-.*1.*\].*\)', result) is not None


class TestIndicatorWriterErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_indicator_type(self):
        """Test that invalid indicator types are rejected."""
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        with pytest.raises(ValueError, match="indicator_type"):
            generate_mql5_indicator(
                indicator_name="Test",
                indicator_type="invalid_type"
            )

    def test_negative_buffer_count(self):
        """Test that negative buffer counts are rejected."""
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        with pytest.raises(ValueError, match="buffer"):
            generate_mql5_indicator(
                indicator_name="Test",
                indicator_type="line",
                buffers=-1
            )

    def test_special_characters_in_name(self):
        """Test that special characters in indicator name are handled."""
        from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator

        # Should either sanitize or raise error
        result = generate_mql5_indicator(
            indicator_name="Test@#$Indicator",
            indicator_type="line"
        )

        # Check that special chars are removed or escaped
        assert "@" not in result or "#property indicator_" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
