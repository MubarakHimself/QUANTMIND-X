"""
Batch Processing API Endpoints

Provides REST API for batch processing operations.
Implements MUB-87: Batch Processing - Making batch processing available via API.

"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/batch", tags=["batch"])

# Import batch processor
from src.agents.batch.batch_processor import (
    BatchProcessor,
    BatchProcessorWithCallback,
    BatchStatus,
    BatchResult,
    Priority,
    BatchItem,
    BatchRequest,
)


# =============================================================================
# Global Batch Processor Instance
# =============================================================================

# Global processor instance with callback support
_batch_processor: Optional[BatchProcessorWithCallback] = None
_callback_registry: Dict[str, callable] = {}


def get_batch_processor() -> BatchProcessorWithCallback:
    """Get or create the global batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        # Default callback that just processes payload as-is
        async def default_callback(payload: Any) -> Any:
            await asyncio.sleep(0.1)  # Simulate processing
            return {"status": "processed", "payload": payload}

        _batch_processor = BatchProcessorWithCallback(
            callback=default_callback,
            max_concurrent=10,
            requests_per_second=10.0,
            burst_size=20,
            max_retries=3,
            retry_delay=1.0
        )
    return _batch_processor


# =============================================================================
# Request/Response Models
# =============================================================================

class BatchSubmitRequest(BaseModel):
    """Request to submit items for batch processing."""
    payloads: List[Any] = Field(..., description="List of payloads to process")
    priority: str = Field(default="NORMAL", description="Priority level: LOW, NORMAL, HIGH, CRITICAL")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for batch")
    callback_type: Optional[str] = Field(default=None, description="Type of callback to use")


class BatchSubmitResponse(BaseModel):
    """Response from batch submission."""
    batch_id: str
    item_count: int
    status: str
    submitted_at: float


class BatchStatusResponse(BaseModel):
    """Status of a batch or item."""
    batch_id: str
    status: str
    total_items: int
    completed_count: int
    failed_count: int
    queue_size: int
    active_items: int


class BatchItemStatusRequest(BaseModel):
    """Request for item status."""
    item_id: str


class BatchItemStatusResponse(BaseModel):
    """Status of a specific item."""
    item_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class BatchResultResponse(BaseModel):
    """Aggregated results for a batch."""
    batch_id: str
    total: int
    successful: int
    failed: int
    results: List[Any]
    errors: List[Dict[str, str]]
    duration: float
    metadata: Dict[str, Any]


class BatchStatsResponse(BaseModel):
    """Statistics about the batch processor."""
    queue_size: int
    active_items: int
    total_batches: int
    running: bool
    rate_limit_rps: float
    burst_size: int
    max_concurrent: int


class BatchCancelRequest(BaseModel):
    """Request to cancel an item."""
    item_id: str


class BatchRegisterCallbackRequest(BaseModel):
    """Request to register a callback type."""
    callback_type: str
    callback_code: str = Field(..., description="Python code for callback (simple lambda/expression)")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start")
async def start_batch_processor():
    """Start the batch processor if not already running."""
    processor = get_batch_processor()
    if not processor._running:
        await processor.start()
        return {"status": "started", "message": "Batch processor started"}
    return {"status": "already_running", "message": "Batch processor already running"}


@router.post("/stop")
async def stop_batch_processor():
    """Stop the batch processor."""
    processor = get_batch_processor()
    if processor._running:
        await processor.stop()
        return {"status": "stopped", "message": "Batch processor stopped"}
    return {"status": "not_running", "message": "Batch processor not running"}


@router.post("/submit", response_model=BatchSubmitResponse)
async def submit_batch(request: BatchSubmitRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch of items for processing.

    Args:
        payloads: List of items to process
        priority: Priority level (LOW, NORMAL, HIGH, CRITICAL)
        metadata: Optional metadata for the batch
        callback_type: Optional registered callback type to use

    Returns:
        Batch submission confirmation with batch_id
    """
    processor = get_batch_processor()

    # Ensure processor is running
    if not processor._running:
        await processor.start()

    # Convert priority string to enum
    try:
        priority = Priority[request.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority: {request.priority}. Must be one of: LOW, NORMAL, HIGH, CRITICAL"
        )

    # Handle callback type if specified
    if request.callback_type and request.callback_type in _callback_registry:
        # Create processor with custom callback
        callback = _callback_registry[request.callback_type]
        processor = BatchProcessorWithCallback(
            callback=callback,
            max_concurrent=10,
            requests_per_second=10.0,
            burst_size=20,
            max_retries=3,
            retry_delay=1.0
        )
        await processor.start()

    # Submit batch
    batch_id = await processor.submit_batch(
        payloads=request.payloads,
        priority=priority,
        metadata=request.metadata
    )

    return BatchSubmitResponse(
        batch_id=batch_id,
        item_count=len(request.payloads),
        status="submitted",
        submitted_at=time.time()
    )


@router.post("/submit-single")
async def submit_single_item(
    payload: Any,
    priority: str = "NORMAL",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Submit a single item for processing.

    Args:
        payload: Single item to process
        priority: Priority level (LOW, NORMAL, HIGH, CRITICAL)
        metadata: Optional metadata for the item

    Returns:
        Item submission confirmation with item_id
    """
    processor = get_batch_processor()

    # Ensure processor is running
    if not processor._running:
        await processor.start()

    # Convert priority string to enum
    try:
        priority_enum = Priority[priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority: {priority}. Must be one of: LOW, NORMAL, HIGH, CRITICAL"
        )

    # Submit single item
    item_id = await processor.submit(
        payload=payload,
        priority=priority_enum,
        metadata=metadata
    )

    return {
        "item_id": item_id,
        "status": "submitted",
        "submitted_at": time.time()
    }


@router.get("/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """Get the status of a batch."""
    processor = get_batch_processor()

    # Check if batch exists
    if batch_id not in processor._batches:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    batch = processor._batches[batch_id]
    stats = processor.get_stats()

    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch.status.value,
        total_items=len(batch.items),
        completed_count=batch.completed_count,
        failed_count=batch.failed_count,
        queue_size=stats["queue_size"],
        active_items=stats["active_items"]
    )


@router.get("/item/{item_id}", response_model=BatchItemStatusResponse)
async def get_item_status(item_id: str):
    """Get the status of a specific item."""
    processor = get_batch_processor()

    status = await processor.get_status(item_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    # Find the item for additional details
    item = None
    for batch in processor._batches.values():
        for i in batch.items:
            if i.id == item_id:
                item = i
                break
        if item:
            break

    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return BatchItemStatusResponse(
        item_id=item_id,
        status=item.status.value,
        result=item.result,
        error=item.error,
        created_at=item.created_at,
        started_at=item.started_at,
        completed_at=item.completed_at
    )


@router.get("/results/{batch_id}", response_model=BatchResultResponse)
async def get_batch_results(batch_id: str):
    """Get aggregated results for a batch."""
    processor = get_batch_processor()

    result = await processor.get_results(batch_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    return BatchResultResponse(
        batch_id=result.batch_id,
        total=result.total,
        successful=result.successful,
        failed=result.failed,
        results=result.results,
        errors=result.errors,
        duration=result.duration,
        metadata=result.metadata
    )


@router.get("/stats", response_model=BatchStatsResponse)
async def get_batch_stats():
    """Get batch processor statistics."""
    processor = get_batch_processor()
    stats = processor.get_stats()

    return BatchStatsResponse(
        queue_size=stats["queue_size"],
        active_items=stats["active_items"],
        total_batches=stats["total_batches"],
        running=stats["running"],
        rate_limit_rps=processor.rate_limiter.rate,
        burst_size=processor.rate_limiter.burst_size,
        max_concurrent=processor.max_concurrent
    )


@router.post("/cancel", response_model=Dict[str, Any])
async def cancel_batch_item(request: BatchCancelRequest):
    """Cancel a pending item."""
    processor = get_batch_processor()

    success = await processor.cancel_item(request.item_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found or already processed")

    return {
        "status": "cancelled",
        "item_id": request.item_id
    }


@router.post("/register-callback")
async def register_callback(request: BatchRegisterCallbackRequest):
    """
    Register a callback type for batch processing.

    The callback_code should be a simple Python expression or lambda.
    Example: "lambda x: {'result': x['value'] * 2}"
    """
    try:
        # Safety: only allow simple lambda expressions
        callback_code = request.callback_code.strip()
        if not callback_code.startswith("lambda "):
            raise HTTPException(
                status_code=400,
                detail="Only lambda expressions are allowed"
            )

        # Compile and test the callback
        callback = eval(callback_code, {"__builtins__": {}})

        # Test with dummy payload
        try:
            callback({"test": True})
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Callback test failed: {str(e)}"
            )

        _callback_registry[request.callback_type] = callback

        return {
            "status": "registered",
            "callback_type": request.callback_type
        }
    except SyntaxError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lambda syntax: {str(e)}"
        )


@router.get("/callbacks")
async def list_callbacks():
    """List all registered callback types."""
    return {
        "callbacks": list(_callback_registry.keys())
    }


@router.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str):
    """Delete a batch and its items (clears from tracking)."""
    processor = get_batch_processor()

    if batch_id not in processor._batches:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    # Remove batch
    del processor._batches[batch_id]

    return {
        "status": "deleted",
        "batch_id": batch_id
    }


@router.get("/batches")
async def list_batches():
    """List all batches."""
    processor = get_batch_processor()

    batches = []
    for batch_id, batch in processor._batches.items():
        batches.append({
            "batch_id": batch_id,
            "status": batch.status.value,
            "total_items": len(batch.items),
            "completed_count": batch.completed_count,
            "failed_count": batch.failed_count,
            "created_at": batch.created_at
        })

    return {
        "batches": batches,
        "total": len(batches)
    }
