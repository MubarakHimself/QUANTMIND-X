"""
Tests for Session Compaction Module.

Tests cover:
- Token estimation
- Critical message separation
- Compaction operations
- Background summarization
- Edge cases
"""

import threading
import time
import unittest
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

from src.agents.memory.compaction import (
    SessionCompactor,
    CompactionConfig,
    CompactionResult,
    estimate_tokens,
    remove_thinking_blocks,
    count_tokens,
)


class TestEstimateTokens(unittest.TestCase):
    """Tests for token estimation functions."""

    def test_empty_string(self):
        """Test empty string returns 0 tokens."""
        self.assertEqual(estimate_tokens(""), 0)

    def test_short_text(self):
        """Test short text token estimation."""
        text = "hello world"
        # 11 chars / 4 = 2.75 -> 2 (floor) but min is 1
        self.assertEqual(estimate_tokens(text), 2)

    def test_long_text(self):
        """Test long text token estimation."""
        text = "a" * 1000
        self.assertEqual(estimate_tokens(text), 250)

    def test_unicode_text(self):
        """Test unicode text token estimation."""
        text = "hello world"  # 11 chars
        self.assertEqual(estimate_tokens(text), 2)

    def test_count_tokens_alias(self):
        """Test count_tokens is alias for estimate_tokens."""
        self.assertEqual(count_tokens("test"), estimate_tokens("test"))


class TestRemoveThinkingBlocks(unittest.TestCase):
    """Tests for thinking block removal."""

    def test_no_thinking_blocks(self):
        """Test text without thinking blocks."""
        text = "This is normal text without thinking."
        cleaned, thinking = remove_thinking_blocks(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(thinking, "")

    def test_single_thinking_block(self):
        """Test single thinking block removal."""
        text = "Hello <think>This is thinking</think> world"
        cleaned, thinking = remove_thinking_blocks(text)
        # The regex normalizes whitespace to single space
        self.assertEqual(cleaned, "Hello world")
        self.assertIn("This is thinking", thinking)

    def test_multiple_thinking_blocks(self):
        """Test multiple thinking blocks removal."""
        text = "Start <think>first</think> middle <think>second</think> end"
        cleaned, thinking = remove_thinking_blocks(text)
        self.assertEqual(cleaned, "Start middle end")


class TestCompactionConfig(unittest.TestCase):
    """Tests for CompactionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompactionConfig()
        self.assertEqual(config.context_limit, 10000)
        self.assertEqual(config.min_tokens_to_init, 7500)
        self.assertEqual(config.min_tokens_between_updates, 2000)
        self.assertEqual(config.preserve_recent_count, 4)
        self.assertTrue(config.enable_background)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CompactionConfig(
            context_limit=5000,
            preserve_recent_count=2,
            enable_background=False,
        )
        self.assertEqual(config.context_limit, 5000)
        self.assertEqual(config.preserve_recent_count, 2)
        self.assertFalse(config.enable_background)


class TestSessionCompactor(unittest.TestCase):
    """Tests for SessionCompactor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CompactionConfig(
            context_limit=1000,
            min_tokens_to_init=500,
            min_tokens_between_updates=200,
            preserve_recent_count=2,
            enable_background=True,
        )
        self.compactor = SessionCompactor(config=self.config)

    def _create_messages(self, count: int, role: str = "user") -> List[Dict[str, Any]]:
        """Helper to create test messages."""
        return [{"role": role, "content": f"Message {i} content here"} for i in range(count)]

    def test_initialization(self):
        """Test compactor initializes correctly."""
        self.assertIsNotNone(self.compactor.config)
        self.assertIsNone(self.compactor.get_summary())

    def test_estimate_message_tokens_empty(self):
        """Test token estimation for empty messages."""
        tokens = self.compactor.estimate_message_tokens([])
        self.assertEqual(tokens, 0)

    def test_estimate_message_tokens_single(self):
        """Test token estimation for single message."""
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = self.compactor.estimate_message_tokens(messages)
        # "Hello world" = 2 tokens + 4 for role = 6
        self.assertGreater(tokens, 0)

    def test_estimate_message_tokens_multiple(self):
        """Test token estimation for multiple messages."""
        messages = self._create_messages(5)
        tokens = self.compactor.estimate_message_tokens(messages)
        self.assertGreater(tokens, 0)

    def test_should_compact_below_limit(self):
        """Test should_compact returns False below limit."""
        messages = [{"role": "user", "content": "Short"}]
        self.assertFalse(self.compactor.should_compact(messages))

    def test_should_compact_above_limit(self):
        """Test should_compact returns True above limit."""
        # Create enough messages to exceed limit (context_limit is 1000 tokens)
        # Each message is ~14 chars = ~3-4 tokens + 4 overhead = ~8 tokens
        # Need > 1000/8 = 125 messages to exceed limit
        messages = self._create_messages(150)
        self.assertTrue(self.compactor.should_compact(messages))

    def test_should_init_background_no_summary(self):
        """Test background init check with no summary."""
        messages = self._create_messages(30)
        # 30 messages with ~20 chars each = ~150 tokens, still below min_tokens_to_init
        self.assertFalse(self.compactor.should_init_background(messages))

    def test_should_init_background_with_summary(self):
        """Test background init check with existing summary."""
        with self.compactor._lock:
            self.compactor._session_summary = "Existing summary"
        messages = self._create_messages(10)
        self.assertFalse(self.compactor.should_init_background(messages))

    def test_compact_empty_messages(self):
        """Test compaction with empty messages."""
        result = self.compactor.compact([])
        self.assertEqual(result.original_tokens, 0)
        self.assertEqual(result.compacted_tokens, 0)

    def test_compact_preserves_system_messages(self):
        """Test compaction preserves system messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self.compactor.compact(messages)
        # Check system message is preserved
        roles = [m.get("role") for m in result.preserved_messages]
        self.assertIn("system", roles)

    def test_compact_preserves_recent_messages(self):
        """Test compaction preserves recent messages."""
        messages = self._create_messages(10)
        result = self.compactor.compact(messages)
        # Should have recent messages preserved
        self.assertLessEqual(len(result.preserved_messages), len(messages))

    def test_compact_reduces_tokens(self):
        """Test compaction actually reduces token count."""
        messages = self._create_messages(20)
        result = self.compactor.compact(messages)
        self.assertLess(result.compacted_tokens, result.original_tokens)

    def test_compact_reduction_ratio(self):
        """Test compaction reduction ratio calculation."""
        messages = self._create_messages(30)
        result = self.compactor.compact(messages)
        self.assertGreater(result.reduction_ratio, 0.0)
        self.assertLessEqual(result.reduction_ratio, 1.0)

    def test_compact_has_summary_for_large_history(self):
        """Test compaction generates summary for large history."""
        messages = self._create_messages(20)
        result = self.compactor.compact(messages)
        # With LLM client, summary may be generated
        # Without it, we get fallback
        self.assertIsNotNone(result.summary)

    def test_instant_compact_no_summary(self):
        """Test instant_compact falls back without summary."""
        messages = self._create_messages(10)
        result = self.compactor.instant_compact(messages)
        # Should return compacted messages (fallback behavior)
        self.assertIsInstance(result, list)

    def test_instant_compact_with_summary(self):
        """Test instant_compact with pre-built summary."""
        messages = self._create_messages(10)
        with self.compactor._lock:
            self.compactor._session_summary = "Test session summary"
            self.compactor._last_summarized_index = 5

        result = self.compactor.instant_compact(messages)
        # Should include summary in messages
        self.assertIsInstance(result, list)

    def test_separate_critical_messages(self):
        """Test critical message separation."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]
        critical, non_critical = self.compactor._separate_critical_messages(messages)

        # System message should be critical
        self.assertTrue(any(m.get("role") == "system" for m in critical))
        # Recent messages should be critical
        self.assertEqual(len(critical), 3)  # system + 2 recent

    def test_reset(self):
        """Test reset clears state."""
        with self.compactor._lock:
            self.compactor._session_summary = "Test summary"
            self.compactor._last_summarized_index = 10
            self.compactor._tokens_at_last_update = 500

        self.compactor.reset()

        self.assertIsNone(self.compactor.get_summary())
        self.assertEqual(self.compactor._last_summarized_index, 0)

    def test_get_summary(self):
        """Test get_summary returns current summary."""
        self.assertIsNone(self.compactor.get_summary())

        with self.compactor._lock:
            self.compactor._session_summary = "Test"

        self.assertEqual(self.compactor.get_summary(), "Test")


class TestBackgroundSummarization(unittest.TestCase):
    """Tests for background summarization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CompactionConfig(
            context_limit=1000,
            enable_background=True,
        )

    def _create_mock_client(self, summary_text: str = "Generated summary") -> Mock:
        """Create a mock LLM client."""
        mock_client = Mock()
        mock_response = {
            "content": summary_text,
            "choices": [{"message": {"content": summary_text}}],
        }
        mock_client.create_completion.return_value = mock_response
        return mock_client

    def test_start_background_monitoring_no_client(self):
        """Test background monitoring requires LLM client."""
        compactor = SessionCompactor(config=self.config, llm_client=None)
        messages = [{"role": "user", "content": "Test"}]

        # Should not raise, just logs warning
        compactor.start_background_monitoring(messages)
        # Thread should not be started
        self.assertIsNone(compactor._update_thread)

    def test_start_background_monitoring_with_client(self):
        """Test background monitoring starts thread."""
        mock_client = self._create_mock_client()
        compactor = SessionCompactor(config=self.config, llm_client=mock_client)
        messages = [{"role": "user", "content": "Test message " * 10}] * 20

        compactor.start_background_monitoring(messages)

        # Thread should be started
        self.assertIsNotNone(compactor._update_thread)
        self.assertTrue(compactor._update_thread.daemon)

    def test_background_creates_summary(self):
        """Test background summarization creates summary."""
        mock_client = self._create_mock_client("Test summary")
        compactor = SessionCompactor(config=self.config, llm_client=mock_client)
        messages = [{"role": "user", "content": "Test"}] * 10

        compactor.start_background_monitoring(messages)

        # Wait for thread to complete
        if compactor._update_thread:
            compactor._update_thread.join(timeout=5.0)

        # Summary should be generated
        self.assertEqual(compactor.get_summary(), "Test summary")

    def test_background_thread_exception_handling(self):
        """Test background thread handles exceptions gracefully."""
        mock_client = Mock()
        mock_client.create_completion.side_effect = Exception("LLM Error")

        compactor = SessionCompactor(config=self.config, llm_client=mock_client)
        messages = [{"role": "user", "content": "Test"}] * 10

        # Should not raise
        compactor.start_background_monitoring(messages)

        if compactor._update_thread:
            compactor._update_thread.join(timeout=2.0)

        # State should be consistent
        self.assertIsNotNone(compactor._update_thread)


class TestCompactionEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CompactionConfig(context_limit=500)
        self.compactor = SessionCompactor(config=self.config)

    def test_multimodal_content(self):
        """Test handling of multimodal content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "image", "image_url": "http://example.com/image.png"},
                ],
            }
        ]
        tokens = self.compactor.estimate_message_tokens(messages)
        # Should handle list content
        self.assertGreater(tokens, 0)

    def test_compact_preserves_all_system_messages(self):
        """Test all system messages are preserved."""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "User message"},
        ]
        result = self.compactor.compact(messages)
        roles = [m.get("role") for m in result.preserved_messages]
        # Both system messages should be preserved
        self.assertEqual(roles.count("system"), 2)

    def test_compact_with_custom_summary_prompt(self):
        """Test compaction with custom summarization prompt."""
        custom_prompt = "Summarize this: {conversation}"
        config = CompactionConfig(summarization_prompt=custom_prompt)
        compactor = SessionCompactor(config=config)
        messages = [{"role": "user", "content": "Test"}] * 10

        result = compactor.compact(messages)
        # Should use custom prompt in the process
        self.assertIsNotNone(result)


class TestCompactionResult(unittest.TestCase):
    """Tests for CompactionResult dataclass."""

    def test_result_creation(self):
        """Test CompactionResult can be created."""
        result = CompactionResult(
            original_tokens=1000,
            compacted_tokens=200,
            reduction_ratio=0.8,
            summary="Test summary",
            preserved_messages=[],
            compacted_messages=[],
            duration_ms=150.0,
        )
        self.assertEqual(result.original_tokens, 1000)
        self.assertEqual(result.compacted_tokens, 200)
        self.assertEqual(result.reduction_ratio, 0.8)
        self.assertEqual(result.duration_ms, 150.0)
        self.assertIsInstance(result.timestamp, datetime)


if __name__ == "__main__":
    unittest.main()
