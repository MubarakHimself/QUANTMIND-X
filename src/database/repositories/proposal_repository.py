"""
Proposal Repository.

Provides database operations for trade proposals.
"""

from typing import Optional, List
from contextlib import contextmanager

from src.database.engine import Session
from src.database.models import TradeProposal


class ProposalRepository:
    """Repository for TradeProposal database operations."""

    def __init__(self):
        """Initialize the proposal repository."""
        pass

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create(
        self,
        bot_id: str,
        symbol: str,
        kelly_score: float,
        regime: str,
        proposed_lot_size: float
    ) -> TradeProposal:
        """
        Create a new trade proposal.

        Args:
            bot_id: Bot/strategy identifier
            symbol: Trading symbol
            kelly_score: Kelly criterion score
            regime: Market regime
            proposed_lot_size: Suggested position size

        Returns:
            Created TradeProposal object
        """
        with self.get_session() as session:
            proposal = TradeProposal(
                bot_id=bot_id,
                symbol=symbol,
                kelly_score=kelly_score,
                regime=regime,
                proposed_lot_size=proposed_lot_size,
                status="pending"
            )
            session.add(proposal)
            session.flush()
            session.refresh(proposal)
            session.expunge(proposal)
            return proposal

    def update(
        self,
        proposal_id: int,
        status: Optional[str] = None,
        approved_lot_size: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[TradeProposal]:
        """Update an existing trade proposal."""
        with self.get_session() as session:
            proposal = session.query(TradeProposal).filter(
                TradeProposal.id == proposal_id
            ).first()

            if proposal is None:
                return None

            if status is not None:
                proposal.status = status
            if approved_lot_size is not None:
                proposal.approved_lot_size = approved_lot_size
            if notes is not None:
                proposal.notes = notes

            session.flush()
            session.refresh(proposal)
            session.expunge(proposal)
            return proposal

    def get_by_id(self, proposal_id: int) -> Optional[TradeProposal]:
        """Get a proposal by ID."""
        with self.get_session() as session:
            proposal = session.query(TradeProposal).filter(
                TradeProposal.id == proposal_id
            ).first()
            if proposal is not None:
                session.expunge(proposal)
            return proposal

    def get_by_bot_id(self, bot_id: str, status: Optional[str] = None) -> List[TradeProposal]:
        """Get all proposals for a bot, optionally filtered by status."""
        with self.get_session() as session:
            query = session.query(TradeProposal).filter(
                TradeProposal.bot_id == bot_id
            )
            if status is not None:
                query = query.filter(TradeProposal.status == status)

            proposals = query.order_by(TradeProposal.created_at.desc()).all()
            for proposal in proposals:
                session.expunge(proposal)
            return proposals

    def get_pending(self) -> List[TradeProposal]:
        """Get all pending proposals."""
        with self.get_session() as session:
            proposals = session.query(TradeProposal).filter(
                TradeProposal.status == "pending"
            ).order_by(TradeProposal.created_at.desc()).all()
            for proposal in proposals:
                session.expunge(proposal)
            return proposals

    def delete(self, proposal_id: int) -> bool:
        """Delete a proposal by ID."""
        with self.get_session() as session:
            proposal = session.query(TradeProposal).filter(
                TradeProposal.id == proposal_id
            ).first()
            if proposal:
                session.delete(proposal)
                return True
            return False
