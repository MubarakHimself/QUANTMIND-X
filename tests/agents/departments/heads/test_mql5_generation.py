"""
Tests for MQL5 EA Code Generation

Story 7.2: Development Department — Real MQL5 EA Code Generation
"""
import pytest
import os
import tempfile


class TestMQL5Generator:
    """Test MQL5 code generator."""

    @pytest.fixture
    def generator(self):
        """Create MQL5Generator instance."""
        from src.mql5.generator import MQL5Generator
        return MQL5Generator()

    @pytest.fixture
    def sample_trd(self):
        """Create sample TRD document."""
        from src.trd.schema import TRDDocument, PositionSizing, PositionSizingMethod

        return TRDDocument(
            strategy_id="test_ea_001",
            strategy_name="Test Moving Average EA",
            version=1,
            symbol="EURUSD",
            timeframe="H4",
            strategy_type="trend",
            description="Moving average crossover EA",
            entry_conditions=[
                "Fast MA crosses above Slow MA - Buy",
                "Fast MA crosses below Slow MA - Sell",
            ],
            exit_conditions=[
                "Opposite crossover signal",
                "Take profit or stop loss hit",
            ],
            position_sizing=PositionSizing(
                method=PositionSizingMethod.FIXED_LOT,
                risk_percent=1.0,
                max_lots=1.0,
                fixed_lot_size=0.01,
            ),
            parameters={
                "session_mask": "UK",
                "force_close_hour": 22,
                "overnight_hold": False,
                "daily_loss_cap": 2.0,
                "spread_filter": 30,
                "slippage": 3,
                "magic_number": 123456,
                "max_orders": 5,
                "trailing_stop": True,
                "trailing_distance": 50,
            },
        )

    def test_generate_ea(self, generator, sample_trd):
        """Test generating MQL5 EA code."""
        code = generator.generate(sample_trd)

        assert code is not None
        assert len(code) > 0
        assert "OnInit" in code
        assert "OnTick" in code
        assert "OnDeinit" in code

    def test_generate_ea_includes_parameters(self, generator, sample_trd):
        """Test generated EA includes TRD parameters."""
        code = generator.generate(sample_trd)

        # Check parameters are mapped
        assert "InpMagicNumber" in code
        assert "123456" in code
        assert "InpMaxSpread" in code
        assert "InpSessionMask" in code

    def test_generate_ea_includes_header(self, generator, sample_trd):
        """Test generated EA has proper header."""
        code = generator.generate(sample_trd)

        # MQL5 uses // @version=5 comment format
        assert "// @version=5" in code
        assert "copyright" in code.lower()
        # Strategy name is converted to underscores for filename
        assert "Test_Moving_Average_EA" in code

    def test_syntax_validation(self, generator, sample_trd):
        """Test MQL5 syntax validation."""
        code = generator.generate(sample_trd)
        is_valid, error = generator.validate_mql5_syntax(code)

        assert is_valid is True
        assert error == ""

    def test_template_variables(self, generator, sample_trd):
        """Test template variables are properly substituted."""
        code = generator.generate(sample_trd)

        # Should not have unreplaced placeholders
        assert "{{strategy_name}}" not in code
        assert "{{symbol}}" not in code

    def test_save_to_file(self, generator, sample_trd):
        """Test saving generated EA to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = generator.generate(sample_trd)
            file_path = os.path.join(tmpdir, "test_ea.mq5")

            generator.save_to_file(code, file_path)

            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                saved_code = f.read()

            assert saved_code == code


class TestMQL5Template:
    """Test MQL5 template."""

    def test_template_exists(self):
        """Test EA template exists."""
        from src.mql5.templates import EA_TEMPLATE

        assert EA_TEMPLATE is not None
        assert len(EA_TEMPLATE) > 0

    def test_template_has_required_functions(self):
        """Test template has required MQL5 functions."""
        from src.mql5.templates import EA_TEMPLATE

        assert "OnInit()" in EA_TEMPLATE
        assert "OnTick()" in EA_TEMPLATE
        assert "OnDeinit" in EA_TEMPLATE

    def test_template_has_input_group(self):
        """Test template has input parameter group."""
        from src.mql5.templates import EA_TEMPLATE

        assert "input group" in EA_TEMPLATE.lower()

    def test_parameter_mapping(self):
        """Test TRD to MQL5 parameter mapping."""
        from src.mql5.templates.ea_base_template import TRD_TO_MQL5_MAPPING

        assert "session_mask" in TRD_TO_MQL5_MAPPING
        assert "force_close_hour" in TRD_TO_MQL5_MAPPING
        assert "daily_loss_cap" in TRD_TO_MQL5_MAPPING


class TestEAOutputStorage:
    """Test EA output storage."""

    @pytest.fixture
    def storage(self):
        """Create EAOutputStorage instance."""
        from src.strategy.output import EAOutputStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = EAOutputStorage(data_dir=tmpdir)
            yield storage

    def test_save_ea(self, storage):
        """Test saving EA to storage."""
        mql5_code = "// Test EA\nvoid OnInit() {}\nvoid OnTick() {}"
        trd_snapshot = {
            "strategy_id": "test_001",
            "strategy_name": "Test EA",
            "symbol": "EURUSD",
        }

        output = storage.save_ea(
            strategy_id="test_001",
            strategy_name="TestEA",
            mql5_code=mql5_code,
            trd_snapshot=trd_snapshot,
        )

        assert output.strategy_id == "test_001"
        assert output.version == 1
        assert os.path.exists(output.file_path)

    def test_version_increment(self, storage):
        """Test version numbering increments."""
        mql5_code = "// Test EA"

        # First save
        output1 = storage.save_ea(
            strategy_id="ver_test",
            strategy_name="VerTest",
            mql5_code=mql5_code,
            trd_snapshot={},
        )
        assert output1.version == 1

        # Second save - same strategy
        output2 = storage.save_ea(
            strategy_id="ver_test",
            strategy_name="VerTest",
            mql5_code=mql5_code,
            trd_snapshot={},
        )
        assert output2.version == 2

    def test_get_ea(self, storage):
        """Test retrieving saved EA."""
        mql5_code = "// Test EA"
        trd_snapshot = {"strategy_id": "retrieve_test"}

        output = storage.save_ea(
            strategy_id="retrieve_test",
            strategy_name="RetrieveTest",
            mql5_code=mql5_code,
            trd_snapshot=trd_snapshot,
        )

        # Retrieve
        retrieved = storage.get_ea("retrieve_test")

        assert retrieved is not None
        assert retrieved.strategy_id == "retrieve_test"
        assert retrieved.version == output.version

    def test_get_all_versions(self, storage):
        """Test retrieving all versions of a strategy."""
        mql5_code = "// Test EA"

        # Create multiple versions
        for _ in range(3):
            storage.save_ea(
                strategy_id="multi_ver",
                strategy_name="MultiVer",
                mql5_code=mql5_code,
                trd_snapshot={},
            )

        versions = storage.get_all_versions("multi_ver")

        assert len(versions) == 3


class TestIntegration:
    """Integration tests for full EA generation pipeline."""

    def test_full_generation_pipeline(self):
        """Test complete TRD to EA generation pipeline."""
        from src.trd.parser import TRDParser
        from src.trd.validator import TRDValidator
        from src.mql5.generator import MQL5Generator
        from src.strategy.output import EAOutputStorage

        # TRD data
        trd_data = {
            "strategy_id": "integration_test",
            "strategy_name": "Integration Test EA",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "entry_conditions": ["Test entry"],
            "parameters": {
                "session_mask": "UK",
                "force_close_hour": 22,
                "magic_number": 999999,
            },
        }

        # Parse
        parser = TRDParser()
        trd = parser.parse_dict(trd_data)

        # Validate
        validator = TRDValidator()
        result = validator.validate(trd)

        assert result.is_valid is True

        # Generate
        generator = MQL5Generator()
        code = generator.generate(trd)

        assert "OnInit" in code

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = EAOutputStorage(data_dir=tmpdir)
            output = storage.save_ea(
                strategy_id=trd.strategy_id,
                strategy_name=trd.strategy_name,
                mql5_code=code,
                trd_snapshot=trd.to_dict(),
            )

            assert output.version == 1
            assert os.path.exists(output.file_path)
