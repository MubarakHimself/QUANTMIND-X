"""
Base Repository.

Provides a base repository class with common CRUD operations for all repositories.
"""

from contextlib import contextmanager
from typing import Optional, List, TypeVar, Type, Generic, Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.engine import Session as BaseSession
from src.database.models.base import Base

# Generic type for model
T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations.

    Provides a template for domain-specific repositories.
    Subclasses should define the model and any specialized operations.
    """

    model: Type[T] = None  # Must be set by subclass

    def __init__(self):
        """Initialize the repository."""
        if self.model is None:
            raise NotImplementedError("Subclasses must define the model class")

    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions.

        Yields:
            SQLAlchemy session

        Raises:
            Exception: On database errors (propagated after rollback)
        """
        session = BaseSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_by_id(self, id: int) -> Optional[T]:
        """
        Retrieve a record by primary key.

        Args:
            id: Primary key value

        Returns:
            Model instance or None if not found
        """
        with self.get_session() as session:
            result = session.get(self.model, id)
            if result is not None:
                session.expunge(result)
            return result

    def get_all(self, limit: int = 100) -> List[T]:
        """
        Retrieve all records, ordered by creation date descending.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of model instances
        """
        with self.get_session() as session:
            # Try to order by created_at if the column exists
            try:
                query = session.query(self.model).order_by(
                    getattr(self.model, 'created_at').desc()
                ).limit(limit)
            except AttributeError:
                query = session.query(self.model).limit(limit)

            results = query.all()
            for result in results:
                session.expunge(result)
            return results

    def create(self, **kwargs: Any) -> T:
        """
        Create a new record.

        Args:
            **kwargs: Model attributes

        Returns:
            Created model instance
        """
        with self.get_session() as session:
            instance = self.model(**kwargs)
            session.add(instance)
            session.flush()
            session.refresh(instance)
            session.expunge(instance)
            return instance

    def update(self, id: int, **kwargs: Any) -> Optional[T]:
        """
        Update an existing record.

        Args:
            id: Primary key value
            **kwargs: Fields to update

        Returns:
            Updated model instance or None if not found
        """
        with self.get_session() as session:
            instance = session.get(self.model, id)
            if instance is None:
                return None

            for key, value in kwargs.items():
                setattr(instance, key, value)

            session.flush()
            session.refresh(instance)
            session.expunge(instance)
            return instance

    def delete(self, id: int) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Primary key value

        Returns:
            True if deleted, False if not found
        """
        with self.get_session() as session:
            instance = session.get(self.model, id)
            if instance is None:
                return False

            session.delete(instance)
            return True

    def filter_by(self, limit: int = 100, **filters: Any) -> List[T]:
        """
        Filter records by given criteria.

        Args:
            limit: Maximum number of records
            **filters: Field-value pairs to filter by

        Returns:
            List of matching model instances
        """
        with self.get_session() as session:
            query = session.query(self.model)
            for field, value in filters.items():
                query = query.filter(getattr(self.model, field) == value)

            try:
                query = query.order_by(getattr(self.model, 'created_at').desc())
            except AttributeError:
                pass

            results = query.limit(limit).all()
            for result in results:
                session.expunge(result)
            return results

    def count(self, **filters: Any) -> int:
        """
        Count records matching filters.

        Args:
            **filters: Field-value pairs to filter by

        Returns:
            Count of matching records
        """
        with self.get_session() as session:
            query = session.query(self.model)
            for field, value in filters.items():
                query = query.filter(getattr(self.model, field) == value)
            return query.count()
