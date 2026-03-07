"""
Pagination utilities for API endpoints.

Provides paginated response models and utility functions for handling
large dataset queries with limit/offset pagination.
"""

from typing import Generic, TypeVar, List, Optional, Protocol
from pydantic import BaseModel, Field


T = TypeVar("T")


class PaginationParams(BaseModel):
    """Standard pagination parameters."""
    limit: int = Field(default=50, ge=1, le=100, description="Maximum items to return")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated response model.

    Returns the data items along with pagination metadata
    that allows clients to navigate through large datasets.
    """
    items: List[T] = Field(description="List of items for the current page")
    total: int = Field(description="Total number of items available")
    limit: int = Field(description="Maximum items requested per page")
    offset: int = Field(description="Number of items skipped")
    has_more: bool = Field(description="Whether there are more items available")

    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        limit: int,
        offset: int
    ) -> "PaginatedResponse[T]":
        """Create a paginated response from items and totals."""
        return cls(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(items)) < total
        )


def paginate(
    items: List[T],
    limit: int,
    offset: int,
    total: Optional[int] = None
) -> PaginatedResponse[T]:
    """
    Paginate a list of items.

    Args:
        items: The list of items to paginate
        limit: Maximum items per page
        offset: Number of items to skip
        total: Total count (if not provided, uses len(items))

    Returns:
        PaginatedResponse with the items and metadata
    """
    if total is None:
        total = len(items)

    # Apply pagination
    paginated_items = items[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated_items,
        total=total,
        limit=limit,
        offset=offset
    )


# Default pagination values
DEFAULT_LIMIT = 50
DEFAULT_OFFSET = 0
MAX_LIMIT = 100
