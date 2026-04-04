"""
Tests for NY Hybrid Queue Assembly.

Story 17.2: DPR Queue Tier Remix
Story 16.3: Inter-Session Cooldown

Tests:
- NY hybrid queue assembly per Story 16.3 spec
- Position 1: Best London performer (SESSION_SPECIALIST + TIER_1 recovery)
- Position 2: TIER_1 recovery candidate (if not already position 1)
- Positions 3-N: TIER_3 DPR-ranked bots
- Remaining: TIER_2 fresh candidates (always after TIER_3)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.risk.dpr.queue_manager import DPRQueueManager
from src.risk.dpr.queue_models import Tier, QueueEntry, DPRQueueOutput


class TestNYHybridQueueAssembly:
    """Test NY hybrid queue assembly per Story 16.3."""

    @pytest.fixture
    def manager(self):
        """Create DPR queue manager with mock dependencies."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        return DPRQueueManager(
            scoring_engine=mock_engine,
            db_session=mock_session,
        )

    def test_ny_hybrid_empty(self, manager):
        """Test NY hybrid queue with no active bots."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_repo.return_value.get_active_bots.return_value = []

            queue = manager.assemble_ny_hybrid_queue("NY")

            assert queue.session_id == "NY"
            assert queue.ny_hybrid_override is True
            assert len(queue.bots) == 0

    def test_ny_hybrid_london_specialist_position_1(self, manager):
        """Test London specialist gets position 1 in NY hybrid queue."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "london_specialist"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "tier1_recovery"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 80
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist') as mock_spec:
                    def is_specialist(bid, sess):
                        return bid == "london_specialist" and sess == "LONDON"
                    mock_spec.side_effect = is_specialist

                    def mock_tier(bot_id):
                        if bot_id == "london_specialist":
                            return Tier.TIER_1
                        return Tier.TIER_1

                    with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                        with patch.object(manager, 'get_recovery_step', return_value=2):
                            with patch.object(manager, '_persist_audit_log'):
                                queue = manager.assemble_ny_hybrid_queue("NY")

                                assert queue.ny_hybrid_override is True
                                assert queue.bots[0].bot_id == "london_specialist"
                                assert queue.bots[0].queue_position == 1
                                assert queue.bots[0].specialist_session == "LONDON"

    def test_ny_hybrid_tier1_recovery_position_2(self, manager):
        """Test TIER_1 recovery gets position 2 when not London specialist."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "london_spec"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "tier1_recovery"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 80
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist') as mock_spec:
                    def is_specialist(bid, sess):
                        return False  # No London specialist
                    mock_spec.side_effect = is_specialist

                    def mock_tier(bot_id):
                        return Tier.TIER_1

                    with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                        def mock_recovery(bot_id):
                            if bot_id == "tier1_recovery":
                                return 2
                            return 0
                        with patch.object(manager, 'get_recovery_step', side_effect=mock_recovery):
                            with patch.object(manager, '_persist_audit_log'):
                                queue = manager.assemble_ny_hybrid_queue("NY")

                                assert queue.ny_hybrid_override is True
                                # London specialist not at position 1, so tier1_recovery should be there
                                assert queue.bots[0].bot_id == "tier1_recovery"
                                assert queue.bots[0].queue_position == 1

    def test_ny_hybrid_tier3_after_recovery(self, manager):
        """Test TIER_3 bots come after recovery bots."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "tier1_bot"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "tier3_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 80
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    def mock_tier(bot_id):
                        if bot_id == "tier1_bot":
                            return Tier.TIER_1
                        return Tier.TIER_3

                    with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                        with patch.object(manager, 'get_recovery_step', return_value=2):
                            with patch.object(manager, '_persist_audit_log'):
                                queue = manager.assemble_ny_hybrid_queue("NY")

                                # TIER_3 should come after TIER_1 recovery
                                tier1_pos = next((i for i, b in enumerate(queue.bots) if b.bot_id == "tier1_bot"), -1)
                                tier3_pos = next((i for i, b in enumerate(queue.bots) if b.bot_id == "tier3_bot"), -1)

                                if tier1_pos >= 0 and tier3_pos >= 0:
                                    assert tier1_pos < tier3_pos

    def test_ny_hybrid_tier2_always_after_tier3(self, manager):
        """Test TIER_2 bots always come after TIER_3 regardless of score."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "tier2_bot"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "tier3_bot"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                # TIER_2 has higher score but should still come after TIER_3
                if bot_id == "tier2_bot":
                    mock_dpr.composite_score = 90
                else:
                    mock_dpr.composite_score = 50
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist', return_value=False):
                    def mock_tier(bot_id):
                        if bot_id == "tier2_bot":
                            return Tier.TIER_2
                        return Tier.TIER_3

                    with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                        with patch.object(manager, 'get_recovery_step', return_value=0):
                            with patch.object(manager, '_persist_audit_log'):
                                queue = manager.assemble_ny_hybrid_queue("NY")

                                # TIER_2 should come after TIER_3 despite higher score
                                tier3_pos = next((i for i, b in enumerate(queue.bots) if b.bot_id == "tier3_bot"), -1)
                                tier2_pos = next((i for i, b in enumerate(queue.bots) if b.bot_id == "tier2_bot"), -1)

                                assert tier3_pos >= 0
                                assert tier2_pos >= 0
                                assert tier3_pos < tier2_pos

    def test_ny_hybrid_london_specialist_takes_priority_over_tier1_recovery(self, manager):
        """Test London specialist at position 1 takes priority over TIER_1 recovery."""
        with patch('src.database.repositories.bot_repository.BotRepository') as mock_repo:
            mock_bot1 = MagicMock()
            mock_bot1.bot_name = "london_specialist"
            mock_bot2 = MagicMock()
            mock_bot2.bot_name = "tier1_recovery"

            mock_repo.return_value.get_active_bots.return_value = [mock_bot1, mock_bot2]

            def mock_get_score(bot_id, session):
                mock_dpr = MagicMock()
                mock_dpr.composite_score = 80
                return mock_dpr

            with patch.object(manager.scoring_engine, 'get_dpr_score', side_effect=mock_get_score):
                with patch.object(manager.scoring_engine, '_is_specialist') as mock_spec:
                    def is_specialist(bid, sess):
                        return bid == "london_specialist" and sess == "LONDON"
                    mock_spec.side_effect = is_specialist

                    def mock_tier(bot_id):
                        return Tier.TIER_1

                    with patch.object(manager, 'tier_assignment', side_effect=mock_tier):
                        with patch.object(manager, 'get_recovery_step', return_value=2):
                            with patch.object(manager, '_persist_audit_log'):
                                queue = manager.assemble_ny_hybrid_queue("NY")

                                # London specialist should be at position 1
                                assert queue.bots[0].bot_id == "london_specialist"
                                assert queue.bots[0].specialist_session == "LONDON"
                                # TIER_1 recovery should not appear twice
                                positions = [b.queue_position for b in queue.bots if b.bot_id == "tier1_recovery"]
                                assert len(positions) == 0 or positions[0] > 1
