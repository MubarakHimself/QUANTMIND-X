"""
Proposal Repository

Provides data access methods for TradeProposal model.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from ..models import TradeProposal
from ..engine import Session as SessionFactory


class ProposalRepository:
    """
    Repository for TradeProposal data access.

    Handles all database operations related to trade proposals.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self._session = session

    @property
    def session(self) -> Session:
        """Get the session (creates new if not provided)."""
        if self._session is not None:
            return self._session
        return SessionFactory()

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
        proposal = TradeProposal(
            bot_id=bot_id,
            symbol=symbol,
            kelly_score=kelly_score,
            regime=regime,
            proposed_lot_size=proposed_lot_size,
            status='pending'
        )
        self.session.add(proposal)
        self.session.flush()
        self.session.refresh(proposal)
        self.session.expunge(proposal)
        return proposal

    def update(
        self,
        proposal_id: int,
        status: str
    ) -> Optional[TradeProposal]:
        """
        Update the status of a trade proposal.

        Args:
            proposal_id: Proposal ID
            status: New status (pending/approved/rejected)

        Returns:
            Updated TradeProposal object or None if not found
        """
        proposal = self.session.query(TradeProposal).filter(
            TradeProposal.id == proposal_id
        ).first()

        if proposal is None:
            return None

        proposal.status = status
        proposal.reviewed_at = datetime.utcnow()

        self.session.flush()
        self.session.refresh(proposal)
        self.session.expunge(proposal)
        return proposal

    def get(self, proposal_id: int) -> Optional[TradeProposal]:
        """
        Retrieve a trade proposal by ID.

        Args:
            proposal_id: Proposal ID

        Returns:
            TradeProposal object or None if not found
        """
        proposal = self.session.query(TradeProposal).filter(
            TradeProposal.id == proposal_id
        ).first()

        if proposal is not None:
            self.session.expunge(proposal)
        return proposal
