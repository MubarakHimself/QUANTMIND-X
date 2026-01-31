"""
Heartbeat API Endpoint

Handles heartbeat requests from MQL5 Expert Advisors.
Validates payload, updates database, and syncs risk matrix.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class HeartbeatPayload(BaseModel):
    """
    Heartbeat payload schema with validation.
    
    **Validates: Property 10: Heartbeat Payload Completeness**
    """
    ea_name: str = Field(..., description="Expert Advisor name", min_length=1)
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)", min_length=1)
    magic_number: int = Field(..., description="EA magic number", ge=0)
    account_id: str = Field(..., description="MT5 account number", min_length=1)
    current_equity: float = Field(..., description="Current account equity", gt=0)
    current_balance: float = Field(..., description="Current account balance", gt=0)
    risk_multiplier: float = Field(default=1.0, description="Current risk multiplier", ge=0, le=2.0)
    timestamp: int = Field(..., description="Unix timestamp", gt=0)
    
    # Optional fields
    open_positions: int = Field(default=0, description="Number of open positions", ge=0)
    daily_pnl: float = Field(default=0.0, description="Daily profit/loss")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format."""
        # Remove common suffixes
        v = v.replace('.', '').replace('_', '').upper()
        if len(v) < 6:
            raise ValueError(f"Invalid symbol format: {v}")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is recent (within 5 minutes)."""
        current_time = datetime.utcnow().timestamp()
        time_diff = abs(current_time - v)
        
        if time_diff > 300:  # 5 minutes
            logger.warning(f"Heartbeat timestamp is {time_diff}s old")
        
        return v


class HeartbeatResponse(BaseModel):
    """Heartbeat response schema."""
    success: bool
    message: str
    risk_multiplier: float
    timestamp: int


class HeartbeatHandler:
    """
    Handles heartbeat requests from MQL5 EAs.
    
    Responsibilities:
    - Validate heartbeat payload
    - Update database with account state
    - Calculate and sync risk multiplier
    - Update global variables
    """
    
    def __init__(self, db_manager=None, disk_syncer=None):
        """
        Initialize heartbeat handler.
        
        Args:
            db_manager: DatabaseManager instance
            disk_syncer: DiskSyncer instance
        """
        self.db_manager = db_manager
        self.disk_syncer = disk_syncer
        
        # Lazy load if not provided
        if self.db_manager is None:
            from src.database.manager import DatabaseManager
            self.db_manager = DatabaseManager()
        
        if self.disk_syncer is None:
            from src.router.sync import DiskSyncer
            self.disk_syncer = DiskSyncer()
    
    def process_heartbeat(self, payload: Dict[str, Any]) -> HeartbeatResponse:
        """
        Process heartbeat request.
        
        Args:
            payload: Raw heartbeat payload dictionary
            
        Returns:
            HeartbeatResponse with updated risk multiplier
        """
        try:
            # Validate payload
            heartbeat = HeartbeatPayload(**payload)
            
            logger.info(
                f"Heartbeat received: EA={heartbeat.ea_name}, "
                f"Symbol={heartbeat.symbol}, Account={heartbeat.account_id}"
            )
            
            # Update database
            self._update_database(heartbeat)
            
            # Calculate risk multiplier
            risk_multiplier = self._calculate_risk_multiplier(heartbeat)
            
            # Sync risk matrix
            self._sync_risk_matrix(heartbeat, risk_multiplier)
            
            # Return response
            return HeartbeatResponse(
                success=True,
                message="Heartbeat processed successfully",
                risk_multiplier=risk_multiplier,
                timestamp=int(datetime.utcnow().timestamp())
            )
            
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")
            return HeartbeatResponse(
                success=False,
                message=f"Error: {str(e)}",
                risk_multiplier=0.0,  # Safe default
                timestamp=int(datetime.utcnow().timestamp())
            )
    
    def _update_database(self, heartbeat: HeartbeatPayload):
        """Update database with heartbeat data."""
        try:
            # Save daily snapshot
            self.db_manager.save_daily_snapshot(
                account_id=heartbeat.account_id,
                equity=heartbeat.current_equity,
                balance=heartbeat.current_balance
            )
            
            logger.debug(f"Database updated for account {heartbeat.account_id}")
            
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
            # Don't fail the heartbeat if database update fails
    
    def _calculate_risk_multiplier(self, heartbeat: HeartbeatPayload) -> float:
        """
        Calculate risk multiplier based on account state.
        
        Uses PropGovernor to calculate quadratic throttle.
        """
        try:
            from src.router.prop.governor import PropGovernor
            
            # Create governor for this account
            governor = PropGovernor(heartbeat.account_id)
            
            # Create mock regime report (simplified for heartbeat)
            from types import SimpleNamespace
            regime_report = SimpleNamespace(news_state="NORMAL")
            
            # Create trade proposal with current balance
            trade_proposal = {
                'current_balance': heartbeat.current_balance,
                'symbol': heartbeat.symbol
            }
            
            # Calculate risk
            mandate = governor.calculate_risk(regime_report, trade_proposal)
            
            return mandate.allocation_scalar
            
        except Exception as e:
            logger.error(f"Failed to calculate risk multiplier: {e}")
            return 1.0  # Safe default
    
    def _sync_risk_matrix(self, heartbeat: HeartbeatPayload, risk_multiplier: float):
        """Sync risk matrix to disk and global variables."""
        try:
            # Build risk matrix
            risk_matrix = {
                heartbeat.symbol: {
                    "multiplier": risk_multiplier,
                    "timestamp": heartbeat.timestamp,
                    "ea_name": heartbeat.ea_name,
                    "magic_number": heartbeat.magic_number
                }
            }
            
            # Sync to disk
            self.disk_syncer.sync_risk_matrix(risk_matrix)
            
            logger.debug(f"Risk matrix synced: {heartbeat.symbol} = {risk_multiplier}")
            
        except Exception as e:
            logger.error(f"Failed to sync risk matrix: {e}")
            # Don't fail the heartbeat if sync fails


# FastAPI endpoint (example)
def create_heartbeat_endpoint():
    """
    Create FastAPI heartbeat endpoint.
    
    Example usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.post("/heartbeat")(create_heartbeat_endpoint())
    """
    handler = HeartbeatHandler()
    
    async def heartbeat_endpoint(payload: Dict[str, Any]) -> HeartbeatResponse:
        """POST /heartbeat endpoint."""
        return handler.process_heartbeat(payload)
    
    return heartbeat_endpoint
