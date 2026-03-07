"""
Market Repository.

Provides database operations for market opportunities and shared assets.
"""

from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import MarketOpportunity, SharedAsset, StrategyFolder


class MarketRepository(BaseRepository[MarketOpportunity]):
    """Repository for MarketOpportunity database operations."""

    model = MarketOpportunity

    def get_active_opportunities(self, limit: int = 100) -> List[MarketOpportunity]:
        """Get all active market opportunities."""
        with self.get_session() as session:
            opportunities = session.query(MarketOpportunity).filter(
                MarketOpportunity.status == 'active'
            ).order_by(MarketOpportunity.created_at.desc()).limit(limit).all()
            for opp in opportunities:
                session.expunge(opp)
            return opportunities

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[MarketOpportunity]:
        """Get opportunities by symbol."""
        with self.get_session() as session:
            opportunities = session.query(MarketOpportunity).filter(
                MarketOpportunity.symbol == symbol
            ).order_by(MarketOpportunity.created_at.desc()).limit(limit).all()
            for opp in opportunities:
                session.expunge(opp)
            return opportunities


class SharedAssetRepository(BaseRepository[SharedAsset]):
    """Repository for SharedAsset database operations."""

    model = SharedAsset

    def get_by_name(self, name: str) -> Optional[SharedAsset]:
        """Get a shared asset by name."""
        with self.get_session() as session:
            asset = session.query(SharedAsset).filter(
                SharedAsset.name == name
            ).first()
            if asset is not None:
                session.expunge(asset)
            return asset

    def get_by_type(self, asset_type: str, limit: int = 100) -> List[SharedAsset]:
        """Get shared assets by type."""
        with self.get_session() as session:
            assets = session.query(SharedAsset).filter(
                SharedAsset.asset_type == asset_type
            ).order_by(SharedAsset.created_at.desc()).limit(limit).all()
            for asset in assets:
                session.expunge(asset)
            return assets


class StrategyFolderRepository(BaseRepository[StrategyFolder]):
    """Repository for StrategyFolder database operations."""

    model = StrategyFolder

    def get_by_name(self, folder_name: str) -> Optional[StrategyFolder]:
        """Get a strategy folder by name."""
        with self.get_session() as session:
            folder = session.query(StrategyFolder).filter(
                StrategyFolder.name == folder_name
            ).first()
            if folder is not None:
                session.expunge(folder)
            return folder
