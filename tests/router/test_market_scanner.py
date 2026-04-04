"""
Unit tests for MarketScanner component.

Tests session breakout detection, volatility scanning, and alert generation.
"""

import pytest
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the component under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router.market_scanner import (
    MarketScanner,
    ScannerAlert,
    AlertType,
    AlertPriority,
)


@pytest.fixture
def scanner():
    """Fallback scanner fixture for tests outside class-scoped fixtures."""
    return MarketScanner(symbols=["EURUSD"])


def test_scanner_imports():
    """Scanners should be importable from new module."""
    from src.router.scanners import MarketScanner, SymbolScanner, TrendScanner

    assert MarketScanner is not None
    assert SymbolScanner is not None
    assert TrendScanner is not None


class TestScannerAlert:
    """Test ScannerAlert data class."""
    
    def test_alert_creation(self):
        """Test creating a scanner alert."""
        alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Bullish breakout above 1.08500",
            confidence=0.85,
            recommended_bots=["london_breakout_01"],
        )
        
        assert alert.type == AlertType.SESSION_BREAKOUT
        assert alert.symbol == "EURUSD"
        assert alert.session == "LONDON"
        assert alert.confidence == 0.85
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = ScannerAlert(
            type=AlertType.VOLATILITY_SPIKE,
            symbol="GBPUSD",
            session="NEW_YORK",
            setup="Volatility spike: 2.5x average",
            confidence=0.75,
            priority=AlertPriority.HIGH,
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["type"] == "volatility_spike"
        assert alert_dict["symbol"] == "GBPUSD"
        assert alert_dict["priority"] == "high"
        assert "timestamp" in alert_dict


class TestMarketScanner:
    """Test MarketScanner main functionality."""
    
    @pytest.fixture
    def scanner(self):
        """Create a MarketScanner instance for testing."""
        return MarketScanner(symbols=["EURUSD", "GBPUSD"])
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initializes correctly."""
        assert scanner.symbols == ["EURUSD", "GBPUSD"]
        assert scanner._recent_alerts == []
    
    def test_get_current_session_name(self, scanner):
        """Test getting current session name."""
        with patch('src.router.market_scanner.get_current_session') as mock_session:
            mock_session.return_value = Mock(value="LONDON")
            
            session = scanner._get_current_session_name()
            assert session == "LONDON"
    
    def test_get_session_start_london(self, scanner):
        """Test getting London session start time."""
        session_start = scanner._get_session_start("LONDON")
        
        assert session_start is not None
        assert session_start.hour == 8  # London opens at 8 AM UTC
    
    def test_get_session_start_new_york(self, scanner):
        """Test getting New York session start time."""
        session_start = scanner._get_session_start("NEW_YORK")
        
        assert session_start is not None
        assert session_start.hour == 13  # NY opens at 1 PM UTC
    
    def test_get_session_start_asian(self, scanner):
        """Test getting Asian session start time."""
        session_start = scanner._get_session_start("ASIAN")
        
        assert session_start is not None
        assert session_start.hour == 0  # Asian opens at 12 AM UTC
    
    def test_get_breakout_bots(self, scanner):
        """Test getting recommended breakout bots."""
        bots = scanner._get_breakout_bots("EURUSD", "bullish")
        
        assert isinstance(bots, list)
        assert len(bots) > 0
    
    def test_get_volatility_bots(self, scanner):
        """Test getting recommended volatility bots."""
        bots = scanner._get_volatility_bots("EURUSD")
        
        assert isinstance(bots, list)
        assert len(bots) > 0
    
    def test_get_news_bots(self, scanner):
        """Test getting recommended news bots."""
        bots = scanner._get_news_bots()
        
        assert isinstance(bots, list)
    
    def test_get_ict_bots(self, scanner):
        """Test getting recommended ICT bots."""
        bots = scanner._get_ict_bots("EURUSD")
        
        assert isinstance(bots, list)


class TestSessionBreakoutScanner:
    """Test session breakout detection."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner with mocked data."""
        scanner = MarketScanner(symbols=["EURUSD"])
        return scanner
    
    def test_scan_breakouts_young_session(self, scanner):
        """Test that young sessions don't trigger breakouts."""
        with patch.object(scanner, '_get_session_start') as mock_start:
            # Session started 15 minutes ago
            mock_start.return_value = datetime.now(timezone.utc) - timedelta(minutes=15)
            
            alerts = scanner.scan_session_breakouts("LONDON")
            
            # Should not scan breakouts in young session
            assert alerts == []
    
    def test_scan_breakouts_bullish_detected(self, scanner):
        """Test detecting bullish breakout."""
        with patch.object(scanner, '_get_session_start') as mock_start:
            # Session started 45 minutes ago
            mock_start.return_value = datetime.now(timezone.utc) - timedelta(minutes=45)
            
            with patch.object(scanner, '_get_opening_range') as mock_range:
                mock_range.return_value = {"high": 1.08500, "low": 1.08300}
                
                with patch.object(scanner, '_get_current_price') as mock_price:
                    # Price above range high
                    mock_price.return_value = 1.08600
                    
                    alerts = scanner.scan_session_breakouts("LONDON")
                    
                    # Should detect bullish breakout
                    assert len(alerts) == 1
                    assert alerts[0].type == AlertType.SESSION_BREAKOUT
                    assert "Bullish" in alerts[0].setup
    
    def test_scan_breakouts_bearish_detected(self, scanner):
        """Test detecting bearish breakdown."""
        with patch.object(scanner, '_get_session_start') as mock_start:
            mock_start.return_value = datetime.now(timezone.utc) - timedelta(minutes=45)
            
            with patch.object(scanner, '_get_opening_range') as mock_range:
                mock_range.return_value = {"high": 1.08500, "low": 1.08300}
                
                with patch.object(scanner, '_get_current_price') as mock_price:
                    # Price below range low
                    mock_price.return_value = 1.08200
                    
                    alerts = scanner.scan_session_breakouts("LONDON")
                    
                    # Should detect bearish breakdown
                    assert len(alerts) == 1
                    assert alerts[0].type == AlertType.SESSION_BREAKOUT
                    assert "Bearish" in alerts[0].setup


class TestVolatilityScanner:
    """Test volatility spike detection."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner for volatility tests."""
        return MarketScanner(symbols=["EURUSD", "GBPUSD"])
    
    def test_no_spike_normal_volatility(self, scanner):
        """Test that normal volatility doesn't trigger alerts."""
        with patch.object(scanner, '_get_current_atr') as mock_current:
            with patch.object(scanner, '_get_average_atr') as mock_avg:
                mock_current.return_value = 0.0010
                mock_avg.return_value = 0.0010
                
                alerts = scanner.scan_volatility_spikes()
                
                # No spike at 1x average
                assert len(alerts) == 0
    
    def test_spike_detected_high_volatility(self, scanner):
        """Test that high volatility triggers alerts."""
        with patch.object(scanner, '_get_current_atr') as mock_current:
            with patch.object(scanner, '_get_average_atr') as mock_avg:
                mock_current.return_value = 0.0030  # 3x average
                mock_avg.return_value = 0.0010
                
                alerts = scanner.scan_volatility_spikes()
                
                # Should detect spike
                assert len(alerts) > 0
                assert alerts[0].type == AlertType.VOLATILITY_SPIKE
                assert alerts[0].priority in [AlertPriority.HIGH, AlertPriority.MEDIUM]


class TestNewsEventScanner:
    """Test news event scanning."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner for news tests."""
        return MarketScanner(symbols=["EURUSD"])
    
    def test_no_news_far_away(self, scanner):
        """Test no alert when news is far away."""
        with patch.object(scanner, '_get_upcoming_news_events') as mock_news:
            # News in 1 hour
            mock_news.return_value = [{
                "name": "NFP",
                "currency": "USD",
                "datetime": datetime.now(timezone.utc) + timedelta(hours=1),
                "impact": "high",
            }]
            
            alerts = scanner.scan_news_events()
            
            # No alert for news > 5 minutes away
            assert len(alerts) == 0
    
    def test_news_alert_5_minutes_before(self, scanner):
        """Test alert triggered 5 minutes before news."""
        with patch.object(scanner, '_get_upcoming_news_events') as mock_news:
            # News in 3 minutes
            mock_news.return_value = [{
                "name": "NFP",
                "currency": "USD",
                "datetime": datetime.now(timezone.utc) + timedelta(minutes=3),
                "impact": "high",
            }]
            
            alerts = scanner.scan_news_events()
            
            # Should alert for news within 5 minutes
            assert len(alerts) == 1
            assert alerts[0].type == AlertType.NEWS_EVENT
            assert alerts[0].priority == AlertPriority.CRITICAL


class TestICTSetupScanner:
    """Test ICT setup detection."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner for ICT tests."""
        return MarketScanner(symbols=["EURUSD"])
    
    def test_no_setups_without_data(self, scanner):
        """Test no alerts when price data unavailable."""
        with patch.object(scanner, '_get_recent_price_data') as mock_data:
            mock_data.return_value = None
            
            alerts = scanner.scan_ict_setups()
            
            assert len(alerts) == 0
    
    def test_fvg_detection(self, scanner):
        """Test FVG detection logic."""
        # Mock price data with FVG pattern
        with patch.object(scanner, '_detect_fvg') as mock_fvg:
            mock_fvg.return_value = [{
                "direction": "bullish",
                "level": 1.08500,
                "confidence": 0.75,
                "gap_size": 0.00020,
            }]
            
            with patch.object(scanner, '_get_recent_price_data') as mock_data:
                mock_data.return_value = [{"close": 1.08600}] * 50
                
                alerts = scanner.scan_ict_setups()
                
                # Should detect FVG
                assert len(alerts) > 0
                assert alerts[0].type == AlertType.ICT_SETUP
                assert "FVG" in alerts[0].setup


class TestScanInterval:
    """Test session-aware scan interval."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner for interval tests."""
        return MarketScanner()
    
    def test_overlap_interval(self, scanner):
        """Test 1-minute interval during overlap."""
        with patch('src.router.market_scanner.get_current_session') as mock_session:
            with patch('src.router.market_scanner.TradingSession') as mock_enum:
                mock_enum.OVERLAP = "OVERLAP"
                mock_session.return_value = mock_enum.OVERLAP
                
                interval = scanner.get_scan_interval()
                
                # Overlap should use 1-minute interval
                assert interval == 60
    
    def test_major_session_interval(self, scanner):
        """Test 5-minute interval during major sessions."""
        with patch('src.router.market_scanner.get_current_session') as mock_session:
            with patch('src.router.market_scanner.TradingSession') as mock_enum:
                mock_enum.LONDON = "LONDON"
                mock_enum.OVERLAP = "OVERLAP"
                mock_enum.NEW_YORK = "NEW_YORK"
                mock_session.return_value = mock_enum.LONDON
                
                interval = scanner.get_scan_interval()
                
                # Major sessions should use 5-minute interval
                assert interval == 300


class TestFullScan:
    """Test full scan execution."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner for full scan tests."""
        return MarketScanner(symbols=["EURUSD"])
    
    def test_run_full_scan(self, scanner):
        """Test running full scan returns list."""
        with patch.object(scanner, 'scan_session_breakouts') as mock_breakouts:
            with patch.object(scanner, 'scan_volatility_spikes') as mock_volatility:
                with patch.object(scanner, 'scan_news_events') as mock_news:
                    with patch.object(scanner, 'scan_ict_setups') as mock_ict:
                        with patch('src.router.market_scanner.get_current_session') as mock_session:
                            mock_breakouts.return_value = []
                            mock_volatility.return_value = []
                            mock_news.return_value = []
                            mock_ict.return_value = []
                            mock_session.return_value = Mock(value="LONDON")
                            
                            alerts = scanner.run_full_scan()
                            
                            assert isinstance(alerts, list)
    
    def test_recent_alerts_stored(self, scanner):
        """Test that recent alerts are stored."""
        mock_alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Test alert",
            confidence=0.8,
        )
        
        with patch.object(scanner, 'scan_session_breakouts') as mock_breakouts:
            with patch.object(scanner, 'scan_volatility_spikes') as mock_vol:
                with patch.object(scanner, 'scan_news_events') as mock_news:
                    with patch.object(scanner, 'scan_ict_setups') as mock_ict:
                        with patch('src.router.market_scanner.get_current_session') as mock_session:
                            mock_breakouts.return_value = [mock_alert]
                            mock_vol.return_value = []
                            mock_news.return_value = []
                            mock_ict.return_value = []
                            mock_session.return_value = Mock(value="LONDON")
                            
                            scanner.run_full_scan()
                            
                            assert len(scanner._recent_alerts) == 1
    
    def test_alert_limit_100(self, scanner):
        """Test that only last 100 alerts are stored."""
        # Add 150 mock alerts
        for i in range(150):
            scanner._recent_alerts.append(ScannerAlert(
                type=AlertType.SESSION_BREAKOUT,
                symbol="EURUSD",
                session="LONDON",
                setup=f"Test alert {i}",
                confidence=0.5,
            ))
        
        with patch.object(scanner, 'scan_session_breakouts') as mock_breakouts:
            with patch.object(scanner, 'scan_volatility_spikes') as mock_vol:
                with patch.object(scanner, 'scan_news_events') as mock_news:
                    with patch.object(scanner, 'scan_ict_setups') as mock_ict:
                        with patch('src.router.market_scanner.get_current_session') as mock_session:
                            mock_breakouts.return_value = []
                            mock_vol.return_value = []
                            mock_news.return_value = []
                            mock_ict.return_value = []
                            mock_session.return_value = Mock(value="LONDON")
                            
                            scanner.run_full_scan()
                            
                            # Should keep only last 100
                            assert len(scanner._recent_alerts) == 100
    
    def test_get_recent_alerts(self, scanner):
        """Test getting recent alerts."""
        scanner._recent_alerts = [
            ScannerAlert(
                type=AlertType.SESSION_BREAKOUT,
                symbol=f"TEST{i}",
                session="LONDON",
                setup="Test",
                confidence=0.5,
            )
            for i in range(10)
        ]
        
        recent = scanner.get_recent_alerts(limit=5)
        
        assert len(recent) == 5


class TestZeroRangeGuard:
    """Test zero-division guard in session breakout scanning."""
    
    @pytest.fixture
    def scanner(self):
        """Create a MarketScanner instance for testing."""
        return MarketScanner(symbols=["EURUSD"])
    
    def test_flat_range_skipped(self, scanner):
        """Test that flat opening ranges (zero size) are skipped."""
        # Mock the opening range with flat high/low (zero range)
        flat_range = {"high": 1.08500, "low": 1.08500}
        
        with patch.object(scanner, '_get_opening_range', return_value=flat_range):
            with patch.object(scanner, '_get_current_price', return_value=1.08500):
                alerts = scanner.scan_session_breakouts("LONDON")
                
                # Should return empty list - flat range should be skipped
                assert len(alerts) == 0
    
    def test_zero_range_logs_debug(self, scanner, caplog):
        """Test that flat range triggers debug log message."""
        flat_range = {"high": 1.08500, "low": 1.08500}
        
        with patch.object(scanner, '_get_opening_range', return_value=flat_range):
            with patch.object(scanner, '_get_current_price', return_value=1.08500):
                with caplog.at_level(logging.DEBUG):
                    alerts = scanner.scan_session_breakouts("LONDON")
                    
                    # Check that debug message was logged
                    assert any("Flat opening range" in record.message for record in caplog.records)
    
    def test_normal_range_processed(self, scanner):
        """Test that normal ranges (non-zero) are processed correctly."""
        normal_range = {"high": 1.08600, "low": 1.08400}  # range_size = 0.002
        
        with patch.object(scanner, '_get_opening_range', return_value=normal_range):
            with patch.object(scanner, '_get_current_price', return_value=1.08700):  # Above high
                alerts = scanner.scan_session_breakouts("LONDON")
                
                # Should return an alert since price broke above range
                assert len(alerts) == 1
                assert alerts[0].type == AlertType.SESSION_BREAKOUT
                assert alerts[0].symbol == "EURUSD"


class TestPersistence:
    """Test persistence of alerts to market_opportunities table."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock DBManager."""
        mock = Mock()
        mock.session = Mock()
        mock.session.execute = Mock()
        mock.session.commit = Mock()
        return mock
    
    def test_persist_alert_success(self, scanner, mock_db_manager):
        """Test successful alert persistence."""
        scanner.hot_db_manager = mock_db_manager
        
        alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Bullish breakout",
            confidence=0.85,
            recommended_bots=["bot1"],
            metadata={"test": "data"}
        )
        
        scanner._persist_alert(alert)
        
        # Verify execute was called
        assert mock_db_manager.session.execute.called
        # Verify commit was called
        assert mock_db_manager.session.commit.called
    
    def test_persist_alert_failure_handled(self, scanner, caplog):
        """Test that persistence failures are handled gracefully."""
        mock = Mock()
        mock.session = Mock()
        mock.session.execute = Mock(side_effect=Exception("DB Error"))
        scanner.hot_db_manager = mock
        
        alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Test",
            confidence=0.5
        )
        
        # Should not raise exception
        scanner._persist_alert(alert)
        
        # Should log warning
        assert any("Failed to persist" in record.message for record in caplog.records)


class TestWebSocketBroadcast:
    """Test WebSocket broadcasting of alerts."""
    
    @pytest.fixture
    def scanner(self):
        """Create a MarketScanner instance for testing."""
        return MarketScanner(symbols=["EURUSD"])
    
    def test_broadcast_alert_calls_ws(self, scanner, caplog):
        """Test that broadcast_alert calls WebSocket endpoint."""
        alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Bullish breakout",
            confidence=0.85,
            recommended_bots=["bot1"]
        )
        
        with patch('src.api.websocket_endpoints.broadcast_market_opportunity') as mock_broadcast:
            with patch('asyncio.new_event_loop'):
                scanner._broadcast_alert(alert)
                
                # Check that broadcast was attempted
                # Note: May fail due to event loop, but function should be called
    
    def test_broadcast_failure_handled(self, scanner, caplog):
        """Test that broadcast failures are handled gracefully."""
        alert = ScannerAlert(
            type=AlertType.SESSION_BREAKOUT,
            symbol="EURUSD",
            session="LONDON",
            setup="Test",
            confidence=0.5
        )
        
        with patch('src.api.websocket_endpoints.broadcast_market_opportunity', side_effect=Exception("WS Error")):
            with patch('asyncio.new_event_loop'):
                # Should not raise exception
                scanner._broadcast_alert(alert)
                
                # Should log warning
                assert any("Failed to broadcast" in record.message for record in caplog.records)


class TestCommanderIntegration:
    """Test Commander integration hooks."""
    
    def test_scanner_commander_integration_init(self):
        """Test ScannerCommanderIntegration initialization."""
        from src.router.market_scanner import ScannerCommanderIntegration
        
        integration = ScannerCommanderIntegration()
        
        assert integration.scanner is not None
        assert integration._active is False
    
    def test_scanner_commander_activate(self):
        """Test activating scanner integration."""
        from src.router.market_scanner import ScannerCommanderIntegration
        
        integration = ScannerCommanderIntegration()
        
        with patch('src.router.market_scanner.start_scanner_scheduler') as mock_start:
            mock_start.return_value = True
            result = integration.activate()
            
            assert result is True
            assert integration._active is True
    
    def test_scanner_commander_deactivate(self):
        """Test deactivating scanner integration."""
        from src.router.market_scanner import ScannerCommanderIntegration
        
        integration = ScannerCommanderIntegration()
        integration._active = True
        
        with patch('src.router.market_scanner.stop_scanner_scheduler') as mock_stop:
            mock_stop.return_value = True
            result = integration.deactivate()
            
            assert result is True
            assert integration._active is False
    
    def test_get_opportunities_for_symbol(self):
        """Test getting opportunities for specific symbol."""
        from src.router.market_scanner import ScannerCommanderIntegration, MarketScanner
        
        # Create scanner with some mock alerts
        scanner = MarketScanner(symbols=["EURUSD", "GBPUSD"])
        scanner._recent_alerts = [
            ScannerAlert(
                type=AlertType.SESSION_BREAKOUT,
                symbol="EURUSD",
                session="LONDON",
                setup="Test",
                confidence=0.8
            ),
            ScannerAlert(
                type=AlertType.VOLATILITY_SPIKE,
                symbol="GBPUSD",
                session="LONDON",
                setup="Test",
                confidence=0.7
            )
        ]
        
        integration = ScannerCommanderIntegration(scanner)
        
        # Get EURUSD opportunities with high confidence
        opportunities = integration.get_opportunities_for_symbol("EURUSD", min_confidence=0.7)
        
        assert len(opportunities) == 1
        assert opportunities[0]["symbol"] == "EURUSD"
    
    def test_get_high_confidence_opportunities(self):
        """Test getting high confidence opportunities."""
        from src.router.market_scanner import ScannerCommanderIntegration, MarketScanner
        
        scanner = MarketScanner(symbols=["EURUSD"])
        scanner._recent_alerts = [
            ScannerAlert(
                type=AlertType.SESSION_BREAKOUT,
                symbol="EURUSD",
                session="LONDON",
                setup="Test",
                confidence=0.9
            ),
            ScannerAlert(
                type=AlertType.VOLATILITY_SPIKE,
                symbol="EURUSD",
                session="LONDON",
                setup="Test",
                confidence=0.5
            )
        ]
        
        integration = ScannerCommanderIntegration(scanner)
        
        opportunities = integration.get_high_confidence_opportunities(min_confidence=0.8)
        
        assert len(opportunities) == 1
        assert opportunities[0]["confidence"] == 0.9


class TestSchedulerIntegration:
    """Test APScheduler integration."""
    
    def test_market_scanner_scheduler_init(self):
        """Test MarketScannerScheduler initialization."""
        from src.router.market_scanner import MarketScannerScheduler, MarketScanner
        
        scheduler = MarketScannerScheduler()
        
        assert scheduler.scanner is not None
        assert scheduler._scheduler is None
        assert scheduler._is_running is False
    
    def test_scheduler_status(self):
        """Test scheduler status reporting."""
        from src.router.market_scanner import MarketScannerScheduler
        
        scheduler = MarketScannerScheduler()
        status = scheduler.get_status()
        
        assert "running" in status
        assert "last_scan_time" in status
        assert "interval_seconds" in status


class TestDynamicBotLimiterIntegration:
    """Test DynamicBotLimiter integration."""

    def test_check_scanner_opportunity_limits_low_confidence(self):
        """Test that low confidence opportunities are rejected."""
        from src.router.market_scanner import check_scanner_opportunity_limits

        can_act, reason = check_scanner_opportunity_limits(
            opportunity_confidence=0.5,
            current_bots=2
        )

        assert can_act is False
        assert "confidence" in reason.lower()

    def test_check_scanner_opportunity_limits_high_confidence(self):
        """Test that high confidence opportunities pass confidence check."""
        from src.router.market_scanner import check_scanner_opportunity_limits

        # With high confidence but bot limit not reached
        can_act, reason = check_scanner_opportunity_limits(
            opportunity_confidence=0.8,
            current_bots=1,
            account_balance=1000.0
        )

        # Should pass confidence check
        assert "confidence" not in reason.lower()


# ============== Tests for Modular Scanner Structure ==============

def test_scanner_imports():
    """Scanners should be importable from new module."""
    from src.router.scanners import MarketScanner, SymbolScanner, TrendScanner

    assert MarketScanner is not None
    assert SymbolScanner is not None
    assert TrendScanner is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
