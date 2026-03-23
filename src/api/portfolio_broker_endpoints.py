"""
Portfolio Broker Account Registry API Endpoints.

Story 9.1: Broker Account Registry & Routing Matrix API

Provides REST endpoints for:
- Broker account CRUD operations (database-backed)
- Routing matrix retrieval and configuration
- MT5 auto-detection integration
- Islamic compliance handling
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.models import (
    BrokerAccount,
    RoutingRule,
    BrokerAccountType,
    RegimeType,
    StrategyTypeEnum,
    get_db_session
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio-brokers"])


# ============= Pydantic Models =============

class PortfolioBrokerAccountCreate(BaseModel):
    """Request model for creating a portfolio broker account."""
    broker_name: str = Field(..., min_length=1, max_length=200)
    account_number: str = Field(..., min_length=1, max_length=50)
    account_type: BrokerAccountType = BrokerAccountType.STANDARD
    account_tag: Optional[str] = Field(None, max_length=50)
    mt5_server: str = Field(..., min_length=1, max_length=200)
    login_encrypted: str = Field(..., min_length=1)
    swap_free: bool = False
    leverage: int = Field(100, ge=1, le=1000)
    currency: str = Field("USD", min_length=3, max_length=10)


class PortfolioBrokerAccountUpdate(BaseModel):
    """Request model for updating a portfolio broker account."""
    broker_name: Optional[str] = Field(None, min_length=1, max_length=200)
    account_type: Optional[BrokerAccountType] = None
    account_tag: Optional[str] = Field(None, max_length=50)
    mt5_server: Optional[str] = Field(None, min_length=1, max_length=200)
    login_encrypted: Optional[str] = Field(None, min_length=1)
    swap_free: Optional[bool] = None
    leverage: Optional[int] = Field(None, ge=1, le=1000)
    currency: Optional[str] = Field(None, min_length=3, max_length=10)
    is_active: Optional[bool] = None


class PortfolioBrokerAccountResponse(BaseModel):
    """Response model for portfolio broker account."""
    id: int
    broker_name: str
    account_number: str
    account_type: str
    account_tag: Optional[str]
    mt5_server: str
    swap_free: bool
    leverage: int
    currency: str
    detected_broker: Optional[str]
    detected_account_type: Optional[str]
    detected_leverage: Optional[int]
    detected_currency: Optional[str]
    is_active: bool
    is_demo: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MT5DetectionResult(BaseModel):
    """MT5 auto-detection result."""
    broker: Optional[str] = None
    account_type: Optional[str] = None
    leverage: Optional[int] = None
    currency: Optional[str] = None


class PortfolioRoutingRuleCreate(BaseModel):
    """Request model for creating a portfolio routing rule."""
    account_tag: Optional[str] = Field(None, max_length=50)
    regime_filter: Optional[RegimeType] = None
    strategy_type: StrategyTypeEnum
    priority: int = Field(100, ge=1, le=1000)
    is_active: bool = True


class PortfolioRoutingRuleResponse(BaseModel):
    """Response model for portfolio routing rule."""
    id: int
    broker_account_id: int
    broker_name: str
    account_number: str
    account_tag: Optional[str]
    regime_filter: Optional[str]
    strategy_type: str
    priority: int
    is_active: bool

    class Config:
        from_attributes = True


class PortfolioRoutingMatrixResponse(BaseModel):
    """Response model for portfolio routing matrix."""
    strategies: List[str]
    accounts: List[dict]
    matrix: List[List[dict]]


# ============= Broker Account Endpoints =============

@router.post("/brokers", response_model=PortfolioBrokerAccountResponse, status_code=201)
async def create_portfolio_broker(
    account: PortfolioBrokerAccountCreate,
    db: Session = Depends(get_db_session)
) -> PortfolioBrokerAccountResponse:
    """
    Register a new portfolio broker account.

    - MT5 auto-detection runs automatically
    - Islamic accounts get swap_free=True automatically
    """
    # Check for duplicate account number
    existing = db.query(BrokerAccount).filter(
        BrokerAccount.account_number == account.account_number
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Account number {account.account_number} already registered"
        )

    # Handle Islamic compliance: auto-set swap_free for Islamic accounts
    if account.account_type == BrokerAccountType.ISLAMIC:
        account.swap_free = True

    # Create the account
    db_account = BrokerAccount(
        broker_name=account.broker_name,
        account_number=account.account_number,
        account_type=account.account_type,
        account_tag=account.account_tag,
        mt5_server=account.mt5_server,
        login_encrypted=account.login_encrypted,
        swap_free=account.swap_free,
        leverage=account.leverage,
        currency=account.currency,
        is_active=True,
        is_demo=False
    )

    db.add(db_account)
    db.commit()
    db.refresh(db_account)

    # Run MT5 auto-detection (simulated)
    try:
        detection = _detect_mt5_account(db_account)
        db_account.detected_broker = detection.broker
        db_account.detected_account_type = detection.account_type
        db_account.detected_leverage = detection.leverage
        db_account.detected_currency = detection.currency
        db.commit()
        db.refresh(db_account)
    except Exception as e:
        logger.warning(f"MT5 auto-detection failed for {account.account_number}: {e}")

    logger.info(f"Created portfolio broker account: {db_account.broker_name} - {db_account.account_number}")

    return _broker_account_to_response(db_account)


@router.get("/brokers", response_model=List[PortfolioBrokerAccountResponse])
async def list_portfolio_brokers(
    active_only: bool = Query(True, description="Only return active accounts"),
    db: Session = Depends(get_db_session)
) -> List[PortfolioBrokerAccountResponse]:
    """List all registered portfolio broker accounts."""
    query = db.query(BrokerAccount)

    if active_only:
        query = query.filter(BrokerAccount.is_active == True)

    accounts = query.order_by(BrokerAccount.created_at.desc()).all()

    return [_broker_account_to_response(acc) for acc in accounts]


@router.get("/brokers/{account_id}", response_model=PortfolioBrokerAccountResponse)
async def get_portfolio_broker(
    account_id: int,
    db: Session = Depends(get_db_session)
) -> PortfolioBrokerAccountResponse:
    """Get a specific portfolio broker account by ID."""
    account = db.query(BrokerAccount).filter(BrokerAccount.id == account_id).first()

    if not account:
        raise HTTPException(status_code=404, detail="Portfolio broker account not found")

    return _broker_account_to_response(account)


@router.put("/brokers/{account_id}", response_model=PortfolioBrokerAccountResponse)
async def update_portfolio_broker(
    account_id: int,
    account: PortfolioBrokerAccountUpdate,
    db: Session = Depends(get_db_session)
) -> PortfolioBrokerAccountResponse:
    """Update a portfolio broker account. MT5 auto-detection re-runs on update."""
    db_account = db.query(BrokerAccount).filter(BrokerAccount.id == account_id).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Portfolio broker account not found")

    # Update fields provided
    update_data = account.model_dump(exclude_unset=True)

    # Handle Islamic compliance
    if 'account_type' in update_data:
        if update_data['account_type'] == BrokerAccountType.ISLAMIC:
            update_data['swap_free'] = True

    for field, value in update_data.items():
        setattr(db_account, field, value)

    db_account.updated_at = datetime.now(timezone.utc)

    # Re-run MT5 auto-detection
    try:
        detection = _detect_mt5_account(db_account)
        db_account.detected_broker = detection.broker
        db_account.detected_account_type = detection.account_type
        db_account.detected_leverage = detection.leverage
        db_account.detected_currency = detection.currency
    except Exception as e:
        logger.warning(f"MT5 re-detection failed for {db_account.account_number}: {e}")

    db.commit()
    db.refresh(db_account)

    logger.info(f"Updated portfolio broker account: {db_account.id}")

    return _broker_account_to_response(db_account)


@router.delete("/brokers/{account_id}", status_code=204)
async def delete_portfolio_broker(
    account_id: int,
    db: Session = Depends(get_db_session)
) -> None:
    """Soft-delete a portfolio broker account (marks as inactive, retains history)."""
    db_account = db.query(BrokerAccount).filter(BrokerAccount.id == account_id).first()

    if not db_account:
        raise HTTPException(status_code=404, detail="Portfolio broker account not found")

    # Soft delete
    db_account.is_active = False
    db_account.updated_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(f"Soft-deleted portfolio broker account: {account_id}")


# ============= Routing Matrix Endpoints =============

@router.get("/routing-matrix", response_model=PortfolioRoutingMatrixResponse)
async def get_portfolio_routing_matrix(
    db: Session = Depends(get_db_session)
) -> PortfolioRoutingMatrixResponse:
    """Get the full portfolio routing matrix of strategies × broker accounts."""
    # Get all active accounts
    accounts = db.query(BrokerAccount).filter(BrokerAccount.is_active == True).all()

    # Get all active routing rules
    rules = db.query(RoutingRule).filter(RoutingRule.is_active == True).all()

    # Get unique strategy types
    strategy_types = [s.value for s in StrategyTypeEnum]

    # Build matrix
    matrix = []
    for strategy in strategy_types:
        row = []
        for account in accounts:
            # Find matching rule
            matching_rule = None
            for rule in rules:
                if (rule.broker_account_id == account.id and
                    rule.strategy_type.value == strategy and
                    (rule.account_tag is None or rule.account_tag == account.account_tag)):
                    matching_rule = rule
                    break

            row.append({
                "account_id": account.id,
                "account_number": account.account_number,
                "broker_name": account.broker_name,
                "assigned": matching_rule is not None,
                "rule_id": matching_rule.id if matching_rule else None,
                "priority": matching_rule.priority if matching_rule else None
            })
        matrix.append(row)

    # Build account list
    account_list = [
        {
            "id": acc.id,
            "broker_name": acc.broker_name,
            "account_number": acc.account_number,
            "account_tag": acc.account_tag,
            "swap_free": acc.swap_free,
            "leverage": acc.leverage,
            "currency": acc.currency
        }
        for acc in accounts
    ]

    return PortfolioRoutingMatrixResponse(
        strategies=strategy_types,
        accounts=account_list,
        matrix=matrix
    )


@router.put("/brokers/{account_id}/routing-rules", response_model=PortfolioRoutingRuleResponse)
async def update_portfolio_routing_rule(
    account_id: int,
    rule: PortfolioRoutingRuleCreate,
    response: Response,
    db: Session = Depends(get_db_session)
) -> PortfolioRoutingRuleResponse:
    """Create or update a routing rule for a portfolio broker account."""
    # Verify account exists
    account = db.query(BrokerAccount).filter(BrokerAccount.id == account_id).first()

    if not account:
        raise HTTPException(status_code=404, detail="Portfolio broker account not found")

    # Check for existing rule with exact match on all key fields
    # Use explicit None check to avoid OR-with-NULL false positives
    tag_filter = (
        RoutingRule.account_tag.is_(None) if rule.account_tag is None
        else RoutingRule.account_tag == rule.account_tag
    )
    regime_filter_clause = (
        RoutingRule.regime_filter.is_(None) if rule.regime_filter is None
        else RoutingRule.regime_filter == rule.regime_filter
    )
    existing_rule = db.query(RoutingRule).filter(
        RoutingRule.broker_account_id == account_id,
        tag_filter,
        regime_filter_clause,
        RoutingRule.strategy_type == rule.strategy_type
    ).first()

    if existing_rule:
        # Update existing rule — return 200
        existing_rule.account_tag = rule.account_tag
        existing_rule.regime_filter = rule.regime_filter
        existing_rule.strategy_type = rule.strategy_type
        existing_rule.priority = rule.priority
        existing_rule.is_active = rule.is_active
        existing_rule.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing_rule)
        created_rule = existing_rule
        response.status_code = 200
    else:
        # Create new rule — return 201
        response.status_code = 201
        db_rule = RoutingRule(
            broker_account_id=account_id,
            account_tag=rule.account_tag,
            regime_filter=rule.regime_filter,
            strategy_type=rule.strategy_type,
            priority=rule.priority,
            is_active=rule.is_active
        )
        db.add(db_rule)
        db.commit()
        db.refresh(db_rule)
        created_rule = db_rule

    logger.info(f"Created/updated portfolio routing rule: {created_rule.id} for account {account_id}")

    return _routing_rule_to_response(created_rule, account)


@router.delete("/routing-rules/{rule_id}", status_code=204)
async def delete_portfolio_routing_rule(
    rule_id: int,
    db: Session = Depends(get_db_session)
) -> None:
    """Delete a portfolio routing rule."""
    rule = db.query(RoutingRule).filter(RoutingRule.id == rule_id).first()

    if not rule:
        raise HTTPException(status_code=404, detail="Portfolio routing rule not found")

    rule.is_active = False
    rule.updated_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(f"Deleted portfolio routing rule: {rule_id}")


# ============= Helper Functions =============

def _broker_account_to_response(account: BrokerAccount) -> PortfolioBrokerAccountResponse:
    """Convert BrokerAccount model to response."""
    return PortfolioBrokerAccountResponse(
        id=account.id,
        broker_name=account.broker_name,
        account_number=account.account_number,
        account_type=account.account_type.value,
        account_tag=account.account_tag,
        mt5_server=account.mt5_server,
        swap_free=account.swap_free,
        leverage=account.leverage,
        currency=account.currency,
        detected_broker=account.detected_broker,
        detected_account_type=account.detected_account_type,
        detected_leverage=account.detected_leverage,
        detected_currency=account.detected_currency,
        is_active=account.is_active,
        is_demo=account.is_demo,
        created_at=account.created_at,
        updated_at=account.updated_at
    )


def _routing_rule_to_response(rule: RoutingRule, account: BrokerAccount) -> PortfolioRoutingRuleResponse:
    """Convert RoutingRule model to response."""
    return PortfolioRoutingRuleResponse(
        id=rule.id,
        broker_account_id=rule.broker_account_id,
        broker_name=account.broker_name,
        account_number=account.account_number,
        account_tag=rule.account_tag,
        regime_filter=rule.regime_filter.value if rule.regime_filter else None,
        strategy_type=rule.strategy_type.value,
        priority=rule.priority,
        is_active=rule.is_active
    )


def _detect_mt5_account(account: BrokerAccount) -> MT5DetectionResult:
    """
    Run MT5 auto-detection for a broker account.

    This is a placeholder that simulates detection.
    In production, this would connect to MT5 and get actual account info.
    """
    # Simulate MT5 detection based on server/ broker name
    broker_lower = account.broker_name.lower()

    # Simple detection heuristics
    detected_broker = account.broker_name
    detected_account_type = account.account_type.value
    detected_leverage = account.leverage
    detected_currency = account.currency

    # Refine based on server
    if "icmarkets" in account.mt5_server.lower():
        detected_broker = "IC Markets"
        detected_currency = "USD"
    elif "roboforex" in account.mt5_server.lower():
        detected_broker = "RoboForex"
        detected_currency = "USD"
    elif "exness" in account.mt5_server.lower():
        detected_broker = "Exness"
        detected_currency = "USD"
    elif "hugosway" in account.mt5_server.lower():
        detected_broker = "Hugo's Way"
        detected_currency = "USD"
    elif " FTMO" in account.mt5_server or "prop" in account.mt5_server.lower():
        detected_broker = "FTMO"
        detected_account_type = "prop_firm"
        detected_currency = "USD"

    return MT5DetectionResult(
        broker=detected_broker,
        account_type=detected_account_type,
        leverage=detected_leverage,
        currency=detected_currency
    )