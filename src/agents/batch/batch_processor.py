"""
Batch Processor for Agent SDK

Provides efficient batch processing with priority queue, rate limiting,
status tracking, and result aggregation.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of a batch request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Priority levels for batch items."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchItem:
    """Individual item in a batch request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRequest:
    """A batch of items to process."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: List[BatchItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    completed_count: int = 0
    failed_count: int = 0


@dataclass
class BatchResult:
    """Aggregated result of a batch operation."""
    batch_id: str
    total: int
    successful: int
    failed: int
    results: List[Any]
    errors: List[Dict[str, str]]
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 20):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token, return True if successful."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(0.05)


class PriorityQueue:
    """Priority queue for batch items."""

    def __init__(self):
        self._queues: Dict[Priority, asyncio.PriorityQueue] = {
            p: asyncio.PriorityQueue() for p in Priority
        }
        self._total_size = 0

    async def put(self, item: BatchItem) -> None:
        """Add item to queue based on priority."""
        # Convert int to Priority if needed
        priority = item.priority
        if isinstance(priority, int):
            priority = Priority(priority)

        priority_value = (-priority.value, item.created_at)
        await self._queues[priority].put((priority_value, item))
        self._total_size += 1

    async def get(self) -> BatchItem:
        """Get highest priority item."""
        # Iterate in reverse order (CRITICAL first)
        for priority in reversed(Priority):
            if not self._queues[priority].empty():
                _, item = await self._queues[priority].get()
                self._total_size -= 1
                return item
        raise asyncio.QueueEmpty()

    async def get_batch(self, max_size: int) -> List[BatchItem]:
        """Get multiple items up to max_size."""
        items = []
        while len(items) < max_size:
            try:
                item = await asyncio.wait_for(self.get(), timeout=0.01)
                items.append(item)
            except asyncio.QueueEmpty:
                break
        return items

    def empty(self) -> bool:
        return self._total_size == 0

    def qsize(self) -> int:
        return self._total_size


class BatchProcessor:
    """
    Process multiple requests efficiently with priority support,
    rate limiting, and result aggregation.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(requests_per_second, burst_size)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._queue = PriorityQueue()
        self._active_items: Dict[str, BatchItem] = {}
        self._batches: Dict[str, BatchRequest] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("Batch processor started")

    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch processor stopped")

    async def submit(
        self,
        payload: Any,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None
    ) -> str:
        """
        Submit a single item for batch processing.

        Returns the item ID.
        """
        # Convert int to Priority if needed
        if isinstance(priority, int):
            priority = Priority(priority)

        item = BatchItem(
            payload=payload,
            priority=priority,
            metadata=metadata or {}
        )

        if batch_id and batch_id in self._batches:
            self._batches[batch_id].items.append(item)
        else:
            batch = BatchRequest(id=batch_id or str(uuid.uuid4()), items=[item])
            self._batches[batch.id] = batch

        await self._queue.put(item)
        logger.debug(f"Submitted item {item.id} with priority {priority.name}")
        return item.id

    async def submit_batch(
        self,
        payloads: List[Any],
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit multiple items as a single batch.

        Returns the batch ID.
        """
        # Convert int to Priority if needed
        if isinstance(priority, int):
            priority = Priority(priority)

        batch = BatchRequest()
        for payload in payloads:
            item = BatchItem(
                payload=payload,
                priority=priority,
                metadata=metadata or {}
            )
            batch.items.append(item)
            await self._queue.put(item)

        self._batches[batch.id] = batch
        logger.info(f"Submitted batch {batch.id} with {len(payloads)} items")
        return batch.id

    async def get_status(self, item_id: str) -> Optional[BatchStatus]:
        """Get status of a specific item."""
        if item_id in self._active_items:
            return self._active_items[item_id].status

        for batch in self._batches.values():
            for item in batch.items:
                if item.id == item_id:
                    return item.status
        return None

    async def get_results(self, batch_id: str) -> Optional[BatchResult]:
        """Get aggregated results for a batch."""
        if batch_id not in self._batches:
            return None

        batch = self._batches[batch_id]
        results = []
        errors = []
        successful = 0
        failed = 0
        duration = 0.0

        for item in batch.items:
            if item.status == BatchStatus.COMPLETED:
                results.append(item.result)
                successful += 1
            elif item.status == BatchStatus.FAILED:
                errors.append({"id": item.id, "error": item.error or "Unknown error"})
                failed += 1

            if item.completed_at and item.created_at:
                duration += item.completed_at - item.created_at

        return BatchResult(
            batch_id=batch_id,
            total=len(batch.items),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            duration=duration,
            metadata={"status": batch.status.value}
        )

    async def cancel_item(self, item_id: str) -> bool:
        """Cancel a pending item."""
        if item_id in self._active_items:
            item = self._active_items[item_id]
            item.status = BatchStatus.CANCELLED
            item.completed_at = time.time()
            del self._active_items[item_id]
            return True
        return False

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                items = await self._queue.get_batch(self.max_concurrent)

                if not items:
                    await asyncio.sleep(0.1)
                    continue

                tasks = []
                for item in items:
                    task = asyncio.create_task(self._process_item(item))
                    tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(0.5)

    async def _process_item(self, item: BatchItem) -> None:
        """Process a single item with retries and rate limiting."""
        await self.rate_limiter.wait_for_token()

        async with self._semaphore:
            if item.status == BatchStatus.CANCELLED:
                return

            item.status = BatchStatus.PROCESSING
            item.started_at = time.time()
            self._active_items[item.id] = item

            # Update batch status
            for batch in self._batches.values():
                if item in batch.items:
                    batch.status = BatchStatus.PROCESSING
                    break

            for attempt in range(self.max_retries):
                try:
                    result = await self._execute_task(item.payload)
                    item.result = result
                    item.status = BatchStatus.COMPLETED
                    item.completed_at = time.time()

                    # Update batch counts
                    for batch in self._batches.values():
                        if item in batch.items:
                            batch.completed_count += 1
                            if batch.completed_count == len(batch.items):
                                batch.status = BatchStatus.COMPLETED
                            break

                    logger.debug(f"Item {item.id} completed successfully")
                    break

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {item.id}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        item.error = str(e)
                        item.status = BatchStatus.FAILED
                        item.completed_at = time.time()

                        for batch in self._batches.values():
                            if item in batch.items:
                                batch.failed_count += 1
                                break

            if item.id in self._active_items:
                del self._active_items[item.id]

    async def _execute_task(self, payload: Any) -> Any:
        """Execute the actual task. Override for custom processing."""
        # Simulate processing - replace with actual implementation
        await asyncio.sleep(0.1)
        return {"status": "processed", "payload": payload}

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "active_items": len(self._active_items),
            "total_batches": len(self._batches),
            "running": self._running
        }


class BatchProcessorWithCallback(BatchProcessor):
    """Batch processor with custom callback support."""

    def __init__(
        self,
        callback: Callable[[Any], Any],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.callback = callback

    async def _execute_task(self, payload: Any) -> Any:
        """Execute custom callback."""
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(payload)
        return self.callback(payload)
