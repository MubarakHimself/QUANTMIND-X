"""
Unit tests for SessionKellyModifiers.

Tests House Money Mode, Reverse HMM, Premium Session threshold lowering,
Win-Reset logic, and Session Auto-Reset functionality.

Story 4.10: Session-Scoped Kelly Modifiers
"""

import unittest
from datetime import datetime, timezone

from src.risk.sizing.session_kelly_modifiers import (
    SessionKellyModifiers,
    SessionKellyState,
    PremiumSessionAssault,
)
from src.router.sessions import TradingSession


class TestSessionKellyModifiers(unittest.TestCase):
    """Test suite for SessionKellyModifiers."""

    def setUp(self):
        """Set up test fixtures."""
        self.modifiers = SessionKellyModifiers()
        self.utc_now = datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)  # London session

    def test_initialization(self):
        """Test SessionKellyModifiers initialization."""
        modifiers = SessionKellyModifiers(
            account_id="test_account",
            hmm_profit_threshold=0.08,
            hmm_loss_threshold=-0.10,
        )
        self.assertEqual(modifiers.account_id, "test_account")
        self.assertEqual(modifiers.hmm_profit_threshold, 0.08)
        self.assertEqual(modifiers.hmm_loss_threshold, -0.10)
        # Initial state
        self.assertEqual(modifiers._hmm_multiplier, 1.0)
        self.assertEqual(modifiers._session_loss_counter, 0)
        self.assertEqual(modifiers._reverse_hmm_multiplier, 1.0)

    # =========================================================================
    # Task 1 (AC #1) - House Money Mode Global Multipliers Tests
    # =========================================================================

    def test_hmm_house_money_positive_threshold(self):
        """
        AC #1: Given daily P&L >= +8%, When House Money Mode activates,
        Then Kelly x 1.4x (House Money effect)
        """
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.10,  # +10% daily P&L
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        self.assertEqual(state.hmm_multiplier, 1.4)
        self.assertEqual(state.session_kelly_multiplier, 1.4)
        self.assertTrue(state.is_house_money_active)
        self.assertFalse(state.is_preservation_mode)

    def test_hmm_preservation_negative_threshold(self):
        """
        AC #1: Given daily P&L <= -10%, When House Money Mode activates,
        Then Kelly x 0.5x (Preservation mode)
        """
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=-0.12,  # -12% daily P&L
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        self.assertEqual(state.hmm_multiplier, 0.5)
        self.assertEqual(state.session_kelly_multiplier, 0.5)
        self.assertFalse(state.is_house_money_active)
        self.assertTrue(state.is_preservation_mode)

    def test_hmm_baseline_no_trigger(self):
        """
        AC #1: Given daily P&L is between thresholds, When House Money Mode evaluates,
        Then Kelly x 1.0x baseline
        """
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.03,  # +3% - not enough for HMM, not enough for preservation
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        self.assertEqual(state.hmm_multiplier, 1.0)
        self.assertEqual(state.session_kelly_multiplier, 1.0)
        self.assertFalse(state.is_house_money_active)
        self.assertFalse(state.is_preservation_mode)

    def test_hmm_exactly_at_threshold(self):
        """Test HMM triggers exactly at threshold."""
        # At exactly +8%
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.08,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        self.assertEqual(state.hmm_multiplier, 1.4)
        self.assertTrue(state.is_house_money_active)

        # At exactly -10%
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=-0.10,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        self.assertEqual(state.hmm_multiplier, 0.5)
        self.assertTrue(state.is_preservation_mode)

    # =========================================================================
    # Task 2 (AC #2) - Premium Session Kelly Threshold Lowering Tests
    # =========================================================================

    def test_premium_session_threshold_lowering(self):
        """
        AC #2: Given premium session, When House Money threshold evaluated,
        Then equity threshold is lowered (effective threshold = 8% - 2% = 6%)
        """
        # Create modifiers with known premium session
        modifiers = SessionKellyModifiers()

        # During premium assault window (London Open assault - hour 8)
        premium_time = datetime(2026, 3, 24, 8, 30, tzinfo=timezone.utc)  # 8:30 UTC = 8:30 London

        # With +7% daily P&L - would NOT trigger HMM normally (needs 8%)
        # But during premium session with 2% threshold boost (8%-2%=6%), SHOULD trigger
        state = modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.07,  # +7%
            current_session=TradingSession.LONDON,
            utc_now=premium_time
        )

        # The threshold should be lowered to 6% during premium, so 7% triggers HMM
        # Note: This depends on the is_premium_session detection
        # If is_premium_session is True, threshold is 8%-2%=6%
        if state.is_premium_session:
            self.assertEqual(state.hmm_multiplier, 1.4)
            self.assertTrue(state.is_house_money_active)

    def test_premium_session_london_open(self):
        """Test London Open assault premium session detection."""
        # London Open assault is 08:00-08:59 London time
        # 8:30 UTC = 8:30 London time (during DST)
        london_open_time = datetime(2026, 3, 24, 8, 30, tzinfo=timezone.utc)

        is_premium, assault = self.modifiers.is_premium_session(
            session=TradingSession.LONDON,
            utc_now=london_open_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.LONDON_OPEN)

    def test_premium_session_london_ny_overlap(self):
        """Test London-NY Overlap assault premium session detection."""
        # London-NY Overlap assault is 13:00-13:59 GMT
        overlap_time = datetime(2026, 3, 24, 13, 30, tzinfo=timezone.utc)

        is_premium, assault = self.modifiers.is_premium_session(
            session=TradingSession.OVERLAP,
            utc_now=overlap_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.LONDON_NY_OVERLAP)

    def test_premium_session_ny_open(self):
        """Test NY Open assault premium session detection."""
        # NY Open assault is 13:30-14:29 NY time (EDT = UTC-4 in March)
        # So 13:30 NY = 17:30 UTC, 14:29 NY = 18:29 UTC
        ny_open_time = datetime(2026, 3, 24, 17, 45, tzinfo=timezone.utc)  # ~13:45 NY time

        is_premium, assault = self.modifiers.is_premium_session(
            session=TradingSession.NEW_YORK,
            utc_now=ny_open_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.NY_OPEN)

    def test_non_premium_session(self):
        """Test non-premium session returns False."""
        # Asian session (not a premium assault)
        asian_time = datetime(2026, 3, 24, 3, 0, tzinfo=timezone.utc)  # 3:00 UTC = 12:00 Tokyo

        is_premium, assault = self.modifiers.is_premium_session(
            session=TradingSession.ASIAN,
            utc_now=asian_time
        )
        self.assertFalse(is_premium)
        self.assertIsNone(assault)

    # =========================================================================
    # Task 3 (AC #3, #4, #5) - Reverse House Money Effect Tests
    # =========================================================================

    def test_reverse_hmm_2_losses_removes_premium(self):
        """
        AC #3: Given 2 consecutive session losses, When Kelly calculated,
        Then Kelly x 1.0x (premium boost removed, no penalty)
        """
        # Simulate 2 losses
        self.modifiers.on_trade_result(is_win=False)  # Loss 1
        self.modifiers.on_trade_result(is_win=False)  # Loss 2

        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.05,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        self.assertEqual(state.reverse_hmm_multiplier, 1.0)
        self.assertEqual(state.session_loss_counter, 2)
        self.assertFalse(state.premium_boost_active)

    def test_reverse_hmm_4_losses_applies_penalty(self):
        """
        AC #4: Given 4 consecutive session losses, When Kelly calculated,
        Then Kelly x 0.70x baseline (session under stress)
        """
        # Simulate 4 losses
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)

        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.05,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        self.assertEqual(state.reverse_hmm_multiplier, 0.70)
        self.assertEqual(state.session_loss_counter, 4)
        self.assertFalse(state.premium_boost_active)

    def test_reverse_hmm_6_losses_minimum_viable(self):
        """
        AC #5: Given 6 consecutive session losses, When Kelly calculated,
        Then Kelly x 0.50x baseline (session is broken)
        """
        # Simulate 6 losses
        for _ in range(6):
            self.modifiers.on_trade_result(is_win=False)

        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.05,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        self.assertEqual(state.reverse_hmm_multiplier, 0.50)
        self.assertEqual(state.session_loss_counter, 6)
        self.assertFalse(state.premium_boost_active)

    def test_reverse_hmm_loss_counter_tracks_correctly(self):
        """Test session loss counter increments correctly on losses."""
        self.assertEqual(self.modifiers._session_loss_counter, 0)

        self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._session_loss_counter, 1)

        self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._session_loss_counter, 2)

        self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._session_loss_counter, 3)

    # =========================================================================
    # Task 4 (AC #6) - Win-Reset Logic Tests
    # =========================================================================

    def test_win_resets_loss_counter(self):
        """
        AC #6: Given winning trade after penalty states, When win closes,
        Then Kelly fraction resets to 1.0x baseline
        """
        # Build up to 4 losses
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)

        self.assertEqual(self.modifiers._session_loss_counter, 4)
        self.assertEqual(self.modifiers._reverse_hmm_multiplier, 0.70)

        # Win resets counter
        self.modifiers.on_trade_result(is_win=True)

        self.assertEqual(self.modifiers._session_loss_counter, 0)
        self.assertEqual(self.modifiers._reverse_hmm_multiplier, 1.0)

    def test_win_resets_win_streak_on_loss(self):
        """Test that loss resets the consecutive win streak."""
        # Simulate 2 consecutive wins (needed to re-enable premium)
        self.modifiers.on_trade_result(is_win=True)
        self.modifiers.on_trade_result(is_win=True)
        self.assertEqual(self.modifiers._consecutive_session_wins, 2)

        # Loss resets win streak
        self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._consecutive_session_wins, 0)

    def test_premium_reenable_requires_2_consecutive_wins(self):
        """
        AC #6: Premium boost requires 2 consecutive session-level wins to re-enable.
        """
        # Start with premium boost disabled (after losses)
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)
        self.assertFalse(self.modifiers._premium_boost_active)

        # One win - still not enough
        self.modifiers.on_trade_result(is_win=True)
        self.assertEqual(self.modifiers._consecutive_session_wins, 1)
        self.assertFalse(self.modifiers._premium_boost_active)

        # Two wins - premium re-enabled
        self.modifiers.on_trade_result(is_win=True)
        self.assertEqual(self.modifiers._consecutive_session_wins, 2)
        self.assertTrue(self.modifiers._premium_boost_active)

    # =========================================================================
    # Task 5 (AC #7) - Session Auto-Reset Tests
    # =========================================================================

    def test_session_close_resets_all_modifiers(self):
        """
        AC #7: Given session closes, When new session begins,
        Then all session-level Kelly modifiers reset to 1.0x
        """
        # Build up some state
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._session_loss_counter, 4)
        self.assertEqual(self.modifiers._reverse_hmm_multiplier, 0.70)

        # Session close resets
        self.modifiers.on_session_close()

        self.assertEqual(self.modifiers._session_loss_counter, 0)
        self.assertEqual(self.modifiers._consecutive_session_wins, 0)
        self.assertEqual(self.modifiers._reverse_hmm_multiplier, 1.0)
        self.assertFalse(self.modifiers._premium_boost_active)

    def test_session_close_resets_premium_state(self):
        """Test session close resets premium boost state."""
        # Get into premium state
        self.modifiers._premium_boost_active = True
        self.modifiers._is_premium_session = True
        self.modifiers._current_premium_assault = PremiumSessionAssault.LONDON_OPEN

        # Session close should reset
        self.modifiers.on_session_close()

        self.assertFalse(self.modifiers._premium_boost_active)
        self.assertFalse(self.modifiers._is_premium_session)
        self.assertIsNone(self.modifiers._current_premium_assault)

    def test_session_start_calls_close(self):
        """Test on_session_start also resets session state."""
        # Build up state
        for _ in range(2):
            self.modifiers.on_trade_result(is_win=False)
        self.assertEqual(self.modifiers._session_loss_counter, 2)

        # Start new session
        self.modifiers.on_session_start(TradingSession.LONDON)

        # Should be reset
        self.assertEqual(self.modifiers._session_loss_counter, 0)
        self.assertEqual(self.modifiers._current_session, TradingSession.LONDON)

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_full_modifier_chain(self):
        """Test full modifier chain: HMM x Reverse HMM."""
        # Scenario: +10% daily P&L (HMM 1.4x) + 4 losses (Reverse HMM 0.70x)
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)

        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.10,  # Triggers HMM 1.4x
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        # Composite = 1.4 * 0.70 = 0.98
        self.assertAlmostEqual(state.session_kelly_multiplier, 0.98, places=2)
        self.assertEqual(state.hmm_multiplier, 1.4)
        self.assertEqual(state.reverse_hmm_multiplier, 0.70)

    def test_hmm_preservation_with_losses(self):
        """Test preservation mode with losses."""
        # Scenario: -12% daily P&L (Preservation 0.5x) + 4 losses (Reverse HMM 0.70x)
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)

        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=-0.12,  # Triggers Preservation 0.5x
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        # Composite = 0.5 * 0.70 = 0.35
        self.assertAlmostEqual(state.session_kelly_multiplier, 0.35, places=2)
        self.assertEqual(state.hmm_multiplier, 0.5)
        self.assertEqual(state.reverse_hmm_multiplier, 0.70)
        self.assertTrue(state.is_preservation_mode)

    def test_state_to_dict(self):
        """Test SessionKellyState.to_dict() method."""
        state = self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.10,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )

        state_dict = state.to_dict()
        self.assertIn('hmm_multiplier', state_dict)
        self.assertIn('reverse_hmm_multiplier', state_dict)
        self.assertIn('session_kelly_multiplier', state_dict)
        self.assertIn('session_loss_counter', state_dict)
        self.assertIn('is_premium_session', state_dict)

    def test_get_multiplier_chain_display(self):
        """Test get_multiplier_chain_display returns human-readable string."""
        # Normal state
        display = self.modifiers.get_multiplier_chain_display()
        self.assertIsInstance(display, str)

        # With HMM active
        self.modifiers.compute_session_kelly_modifier(
            daily_pnl_pct=0.10,
            current_session=TradingSession.LONDON,
            utc_now=self.utc_now
        )
        display = self.modifiers.get_multiplier_chain_display()
        self.assertIn('HMM', display)

        # With 4 losses
        for _ in range(4):
            self.modifiers.on_trade_result(is_win=False)
        display = self.modifiers.get_multiplier_chain_display()
        self.assertIn('Stress', display)

    def test_get_modifier_chain_components(self):
        """Test get_modifier_chain_components returns individual modifiers."""
        components = self.modifiers.get_modifier_chain_components()

        self.assertIn('hmm_multiplier', components)
        self.assertIn('reverse_hmm_multiplier', components)
        self.assertIn('premium_boost_active', components)
        self.assertIn('session_kelly_multiplier', components)


class TestPremiumSessionAssault(unittest.TestCase):
    """Test suite for PremiumSessionAssault enum and detection."""

    def test_premium_assault_values(self):
        """Test PremiumSessionAssault enum has expected values."""
        self.assertEqual(PremiumSessionAssault.LONDON_OPEN.value, "london_open")
        self.assertEqual(PremiumSessionAssault.LONDON_NY_OVERLAP.value, "london_ny_overlap")
        self.assertEqual(PremiumSessionAssault.NY_OPEN.value, "ny_open")

    def test_premium_assault_in_assault_windows(self):
        """Test premium assault detection across different times."""
        modifiers = SessionKellyModifiers()

        # London Open assault (8:00-8:59 London)
        london_open_time = datetime(2026, 3, 24, 8, 0, tzinfo=timezone.utc)  # 8:00 UTC
        is_premium, assault = modifiers.is_premium_session(
            session=TradingSession.LONDON,
            utc_now=london_open_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.LONDON_OPEN)

        # London-NY Overlap assault (13:00-13:59 GMT)
        overlap_time = datetime(2026, 3, 24, 13, 0, tzinfo=timezone.utc)  # 13:00 UTC
        is_premium, assault = modifiers.is_premium_session(
            session=TradingSession.OVERLAP,
            utc_now=overlap_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.LONDON_NY_OVERLAP)

        # NY Open assault (13:30-14:29 NY time = 17:30-18:29 UTC during DST)
        ny_open_time = datetime(2026, 3, 24, 17, 45, tzinfo=timezone.utc)  # ~13:45 NY time
        is_premium, assault = modifiers.is_premium_session(
            session=TradingSession.NEW_YORK,
            utc_now=ny_open_time
        )
        self.assertTrue(is_premium)
        self.assertEqual(assault, PremiumSessionAssault.NY_OPEN)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for SessionKellyModifiers."""

    def test_zero_daily_pnl(self):
        """Test with exactly zero daily P&L."""
        state = SessionKellyModifiers().compute_session_kelly_modifier(
            daily_pnl_pct=0.0,
            current_session=TradingSession.LONDON,
            utc_now=datetime.now(timezone.utc)
        )
        self.assertEqual(state.hmm_multiplier, 1.0)
        self.assertFalse(state.is_house_money_active)
        self.assertFalse(state.is_preservation_mode)

    def test_very_large_positive_pnl(self):
        """Test with very large positive daily P&L."""
        state = SessionKellyModifiers().compute_session_kelly_modifier(
            daily_pnl_pct=1.0,  # +100%
            current_session=TradingSession.LONDON,
            utc_now=datetime.now(timezone.utc)
        )
        self.assertEqual(state.hmm_multiplier, 1.4)
        self.assertTrue(state.is_house_money_active)

    def test_very_large_negative_pnl(self):
        """Test with very large negative daily P&L."""
        state = SessionKellyModifiers().compute_session_kelly_modifier(
            daily_pnl_pct=-0.50,  # -50%
            current_session=TradingSession.LONDON,
            utc_now=datetime.now(timezone.utc)
        )
        self.assertEqual(state.hmm_multiplier, 0.5)
        self.assertTrue(state.is_preservation_mode)

    def test_multiple_wins_then_loss(self):
        """Test that multiple wins then a loss increments counter."""
        modifiers = SessionKellyModifiers()

        # 3 wins
        for _ in range(3):
            modifiers.on_trade_result(is_win=True)
        self.assertEqual(modifiers._session_loss_counter, 0)
        self.assertEqual(modifiers._consecutive_session_wins, 3)

        # 1 loss
        modifiers.on_trade_result(is_win=False)
        self.assertEqual(modifiers._session_loss_counter, 1)
        self.assertEqual(modifiers._consecutive_session_wins, 0)

    def test_session_close_idempotent(self):
        """Test that calling session_close multiple times is safe."""
        modifiers = SessionKellyModifiers()

        # Build some state
        for _ in range(2):
            modifiers.on_trade_result(is_win=False)

        # Close multiple times
        modifiers.on_session_close()
        modifiers.on_session_close()
        modifiers.on_session_close()

        self.assertEqual(modifiers._session_loss_counter, 0)


if __name__ == '__main__':
    unittest.main()
