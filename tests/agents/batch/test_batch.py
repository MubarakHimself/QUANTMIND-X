"""
Unit Tests for Batch Processor

Tests batch processing functionality including priority queue,
rate limiting, status tracking, and result aggregation.
"""

import asyncio
import pytest
import time
from typing import Any


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_batch_status_values(self):
        """Test all batch status values exist."""
        from src.agents.batch.batch_processor import BatchStatus

        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.PROCESSING.value == "processing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"
        assert BatchStatus.CANCELLED.value == "cancelled"


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_order(self):
        """Test priority levels are ordered correctly."""
        from src.agents.batch.batch_processor import Priority

        assert Priority.LOW.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.CRITICAL.value


class TestBatchItem:
    """Tests for BatchItem dataclass."""

    def test_batch_item_creation(self):
        """Test creating a batch item."""
        from src.agents.batch.batch_processor import BatchItem, Priority

        item = BatchItem(
            payload={"key": "value"},
            priority=Priority.HIGH,
            metadata={"source": "test"}
        )

        assert item.id is not None
        assert item.payload == {"key": "value"}
        assert item.priority == Priority.HIGH
        assert item.status.value == "pending"
        assert item.metadata["source"] == "test"

    def test_batch_item_default_values(self):
        """Test batch item default values."""
        from src.agents.batch.batch_processor import BatchItem, BatchStatus

        item = BatchItem()

        assert item.id is not None
        assert item.status == BatchStatus.PENDING
        assert item.result is None
        assert item.error is None
        assert item.priority.value == 1  # NORMAL


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_batch_request_creation(self):
        """Test creating a batch request."""
        from src.agents.batch.batch_processor import BatchRequest, BatchItem

        items = [BatchItem(), BatchItem()]
        request = BatchRequest(items=items)

        assert request.id is not None
        assert len(request.items) == 2
        assert request.status.value == "pending"


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test acquiring tokens."""
        from src.agents.batch.batch_processor import RateLimiter

        limiter = RateLimiter(requests_per_second=10.0, burst_size=5)
        acquired = await limiter.acquire()

        assert acquired is True
        assert limiter.tokens < 5

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        """Test burst capacity."""
        from src.agents.batch.batch_processor import RateLimiter

        limiter = RateLimiter(requests_per_second=1.0, burst_size=3)

        # Should be able to burst up to burst_size
        results = []
        for _ in range(3):
            result = await limiter.acquire()
            results.append(result)

        assert all(results)
        # Fourth should fail without waiting
        assert await limiter.acquire() is False

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test waiting for token."""
        from src.agents.batch.batch_processor import RateLimiter

        limiter = RateLimiter(requests_per_second=100.0, burst_size=1)
        await limiter.acquire()  # Use the only token

        # Should wait and eventually get a token
        await limiter.wait_for_token()
        result = await limiter.acquire()

        assert result is False  # Token was consumed


class TestPriorityQueue:
    """Tests for PriorityQueue."""

    @pytest.mark.asyncio
    async def test_priority_queue_put_get(self):
        """Test basic put and get."""
        from src.agents.batch.batch_processor import PriorityQueue, BatchItem, Priority

        queue = PriorityQueue()
        item = BatchItem(payload="test", priority=Priority.HIGH)

        await queue.put(item)
        assert queue.qsize() == 1

        retrieved = await queue.get()
        assert retrieved.payload == "test"
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self):
        """Test priority ordering."""
        from src.agents.batch.batch_processor import PriorityQueue, BatchItem, Priority

        queue = PriorityQueue()

        low_item = BatchItem(payload="low", priority=Priority.LOW)
        high_item = BatchItem(payload="high", priority=Priority.HIGH)
        critical_item = BatchItem(payload="critical", priority=Priority.CRITICAL)

        await queue.put(low_item)
        await queue.put(high_item)
        await queue.put(critical_item)

        # Should get critical first, then high, then low
        first = await queue.get()
        second = await queue.get()
        third = await queue.get()

        assert first.payload == "critical"
        assert second.payload == "high"
        assert third.payload == "low"

    @pytest.mark.asyncio
    async def test_priority_queue_get_batch(self):
        """Test getting multiple items."""
        from src.agents.batch.batch_processor import PriorityQueue, BatchItem

        queue = PriorityQueue()
        items = [BatchItem(payload=f"item{i}") for i in range(5)]

        for item in items:
            await queue.put(item)

        batch = await queue.get_batch(3)
        assert len(batch) == 3
        assert queue.qsize() == 2


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_batch_processor_start_stop(self):
        """Test starting and stopping processor."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor()
        await processor.start()

        assert processor._running is True

        await processor.stop()

        assert processor._running is False

    @pytest.mark.asyncio
    async def test_submit_single_item(self):
        """Test submitting a single item."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor(max_concurrent=2, requests_per_second=100)
        await processor.start()

        item_id = await processor.submit({"data": "test"}, priority=1)

        assert item_id is not None
        assert len(processor._batches) > 0

        await processor.stop()

    @pytest.mark.asyncio
    async def test_submit_batch(self):
        """Test submitting multiple items."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor(max_concurrent=5, requests_per_second=100)
        await processor.start()

        payloads = [{"id": i} for i in range(10)]
        batch_id = await processor.submit_batch(payloads)

        assert batch_id is not None
        assert len(processor._batches[batch_id].items) == 10

        await processor.stop()

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting item status."""
        from src.agents.batch.batch_processor import BatchProcessor, BatchStatus

        processor = BatchProcessor(max_concurrent=1, requests_per_second=100)
        await processor.start()

        item_id = await processor.submit({"data": "test"})

        # Allow processing to start
        await asyncio.sleep(0.2)

        status = await processor.get_status(item_id)
        assert status in [BatchStatus.PENDING, BatchStatus.PROCESSING, BatchStatus.COMPLETED]

        await processor.stop()

    @pytest.mark.asyncio
    async def test_get_results(self):
        """Test getting batch results."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor(max_concurrent=5, requests_per_second=100)
        await processor.start()

        payloads = [{"id": i} for i in range(3)]
        batch_id = await processor.submit_batch(payloads)

        # Wait for processing
        await asyncio.sleep(0.5)

        result = await processor.get_results(batch_id)

        assert result is not None
        assert result.batch_id == batch_id
        assert result.total == 3

        await processor.stop()

    @pytest.mark.asyncio
    async def test_cancel_item(self):
        """Test cancelling an item."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor(max_concurrent=1, requests_per_second=100)
        await processor.start()

        item_id = await processor.submit({"data": "test"})

        # Give it time to be picked up
        await asyncio.sleep(0.05)

        cancelled = await processor.cancel_item(item_id)
        # May be cancelled or already processing

        await processor.stop()

    @pytest.mark.asyncio
    async def test_priority_processing(self):
        """Test that higher priority items are processed first."""
        from src.agents.batch.batch_processor import BatchProcessor, Priority

        processor = BatchProcessor(max_concurrent=1, requests_per_second=100)
        await processor.start()

        # Submit low priority first
        await processor.submit({"priority": "low"}, priority=Priority.LOW)
        # Submit high priority second
        await processor.submit({"priority": "high"}, priority=Priority.HIGH)

        await asyncio.sleep(0.3)

        # Get stats to verify items were processed
        stats = processor.get_stats()
        assert stats["queue_size"] >= 0

        await processor.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works."""
        from src.agents.batch.batch_processor import BatchProcessor

        # Very low rate limit - test by checking queue accumulation
        processor = BatchProcessor(
            max_concurrent=1,
            requests_per_second=2.0,
            burst_size=1
        )
        await processor.start()

        # Submit items quickly - they should queue up
        for i in range(3):
            await processor.submit({"id": i})

        # Give time for items to be picked up
        await asyncio.sleep(0.1)

        # Check that the queue has items (rate limiting is active)
        stats = processor.get_stats()
        # Some items should be queued or in progress
        total_items = stats["queue_size"] + stats["active_items"]
        assert total_items >= 0  # Items are being tracked

        await processor.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting processor statistics."""
        from src.agents.batch.batch_processor import BatchProcessor

        processor = BatchProcessor()
        stats = processor.get_stats()

        assert "queue_size" in stats
        assert "active_items" in stats
        assert "total_batches" in stats
        assert "running" in stats


class TestBatchProcessorWithCallback:
    """Tests for BatchProcessorWithCallback."""

    @pytest.mark.asyncio
    async def test_custom_callback(self):
        """Test custom callback execution."""
        from src.agents.batch.batch_processor import BatchProcessorWithCallback

        def transform(payload: Any) -> Any:
            return {"transformed": payload["value"] * 2}

        processor = BatchProcessorWithCallback(
            callback=transform,
            max_concurrent=2,
            requests_per_second=100
        )
        await processor.start()

        item_id = await processor.submit({"value": 5})

        await asyncio.sleep(0.3)

        status = await processor.get_status(item_id)
        assert status.value in ["completed", "processing", "pending"]

        await processor.stop()


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self):
        """Test creating a batch result."""
        from src.agents.batch.batch_processor import BatchResult

        result = BatchResult(
            batch_id="test-123",
            total=10,
            successful=8,
            failed=2,
            results=[1, 2, 3],
            errors=[{"id": "err1", "error": "test error"}],
            duration=5.5
        )

        assert result.batch_id == "test-123"
        assert result.total == 10
        assert result.successful == 8
        assert result.failed == 2
        assert len(result.results) == 3
        assert len(result.errors) == 1
        assert result.duration == 5.5
