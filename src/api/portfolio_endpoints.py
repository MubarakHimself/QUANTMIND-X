"""
Portfolio API Endpoints

Exposes portfolio report and metrics via REST API endpoints.

Story: 7-8-risk-trading-portfolio-department-real-implementations
AC #3: GET /api/portfolio/report returns total equity, P&L attribution per strategy,
       P&L attribution per broker, drawdown per account.
"""

import logging
from typing import Optional
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@lru_cache
def get_portfolio_head():
    """Get or create PortfolioHead instance (cached)."""
    from src.agents.departments.heads.portfolio_head import PortfolioHead
    return PortfolioHead()


# =============================================================================
# Response Models
# =============================================================================

class StrategyAttributionResponse(BaseModel):
    """P&L attribution for a single strategy."""
    strategy: str = Field(..., description="Strategy name")
    pnl: float = Field(..., description="Profit/Loss in account currency")
    percentage: float = Field(..., description="Percentage of total P&L")


class BrokerAttributionResponse(BaseModel):
    """P&L attribution for a single broker."""
    broker: str = Field(..., description="Broker name")
    pnl: float = Field(..., description="Profit/Loss in account currency")
    percentage: float = Field(..., description="Percentage of total P&L")


class AccountDrawdownResponse(BaseModel):
    """Drawdown for a single account."""
    account_id: str = Field(..., description="Account identifier")
    drawdown_pct: float = Field(..., description="Drawdown percentage")


class PnLAttributionResponse(BaseModel):
    """P&L attribution structure."""
    by_strategy: list[StrategyAttributionResponse] = Field(
        default_factory=list,
        description="P&L attribution by strategy"
    )
    by_broker: list[BrokerAttributionResponse] = Field(
        default_factory=list,
        description="P&L attribution by broker"
    )


class PortfolioReportResponse(BaseModel):
    """
    Complete portfolio report.

    AC #3: Returns total equity, P&L attribution per strategy,
    P&L attribution per broker, drawdown per account.
    """
    total_equity: float = Field(..., description="Total equity across all accounts")
    pnl_attribution: PnLAttributionResponse = Field(
        ...,
        description="P&L attribution by strategy and broker"
    )
    drawdown_by_account: list[AccountDrawdownResponse] = Field(
        default_factory=list,
        description="Drawdown per account"
    )
    generated_at: str = Field(..., description="Report generation timestamp")


class TotalEquityResponse(BaseModel):
    """Total equity response."""
    total_equity: float = Field(..., description="Total equity")
    account_count: int = Field(..., description="Number of accounts")


class StrategyPnLResponse(BaseModel):
    """Strategy P&L attribution response."""
    period: str = Field(..., description="Period for attribution")
    total_pnl: float = Field(..., description="Total P&L")
    by_strategy: list[StrategyAttributionResponse] = Field(
        default_factory=list,
        description="P&L by strategy"
    )


class BrokerPnLResponse(BaseModel):
    """Broker P&L attribution response."""
    period: str = Field(..., description="Period for attribution")
    total_pnl: float = Field(..., description="Total P&L")
    by_broker: list[BrokerAttributionResponse] = Field(
        default_factory=list,
        description="P&L by broker"
    )


class AccountDrawdownsResponse(BaseModel):
    """Account drawdowns response."""
    by_account: list[AccountDrawdownResponse] = Field(
        default_factory=list,
        description="Drawdown by account"
    )


# =============================================================================
# New Response Models for Story 9.2: Portfolio Metrics & Attribution API
# =============================================================================

class AccountSummaryResponse(BaseModel):
    """Summary for a single account."""
    account_id: str = Field(..., description="Account identifier")
    equity: float = Field(..., description="Account equity")
    daily_pnl: float = Field(..., description="Daily P&L")
    drawdown: float = Field(..., description="Drawdown percentage")


class PortfolioSummaryResponse(BaseModel):
    """
    Portfolio summary endpoint response.

    AC1: Returns total equity, daily P&L, drawdown, active strategies, and per-account details.
    """
    total_equity: float = Field(..., description="Total equity across all accounts")
    daily_pnl: float = Field(..., description="Daily profit/loss")
    daily_pnl_pct: float = Field(..., description="Daily P&L percentage")
    total_drawdown: float = Field(..., description="Total portfolio drawdown percentage")
    active_strategies: list[str] = Field(
        default_factory=list,
        description="List of active strategy names"
    )
    accounts: list[AccountSummaryResponse] = Field(
        default_factory=list,
        description="Per-account summary"
    )
    drawdown_alert: bool = Field(
        default=False,
        description="Whether drawdown exceeds 10% threshold"
    )


class StrategyAttributionWithEquityResponse(BaseModel):
    """P&L attribution for a single strategy with equity contribution."""
    strategy: str = Field(..., description="Strategy name")
    pnl: float = Field(..., description="Profit/Loss in account currency")
    percentage: float = Field(..., description="Percentage of total P&L")
    equity_contribution: float = Field(..., description="Equity contribution to portfolio")


class PortfolioAttributionResponse(BaseModel):
    """
    Portfolio attribution endpoint response.

    AC2: Returns P&L attribution by strategy (with equity contribution) and by broker.
    """
    by_strategy: list[StrategyAttributionWithEquityResponse] = Field(
        default_factory=list,
        description="P&L attribution by strategy"
    )
    by_broker: list[BrokerAttributionResponse] = Field(
        default_factory=list,
        description="P&L attribution by broker"
    )


class CorrelationPairResponse(BaseModel):
    """A single correlation pair in the matrix."""
    strategy_a: str = Field(..., description="First strategy name")
    strategy_b: str = Field(..., description="Second strategy name")
    correlation: float = Field(..., description="Correlation coefficient (-1 to 1)")
    period_days: int = Field(..., description="Period in days for calculation")


class PortfolioCorrelationResponse(BaseModel):
    """
    Portfolio correlation matrix endpoint response.

    AC3: Returns NxN correlation matrix of strategy returns.
    """
    matrix: list[CorrelationPairResponse] = Field(
        default_factory=list,
        description="Correlation pairs"
    )
    high_correlation_threshold: float = Field(
        default=0.7,
        description="Threshold for high correlation alerts"
    )
    period_days: int = Field(..., description="Period in days for calculation")
    generated_at: str = Field(..., description="Matrix generation timestamp")


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/report", response_model=PortfolioReportResponse)
async def get_portfolio_report():
    """
    Get complete portfolio report.

    AC #3: Returns:
    - total_equity: Total equity across all accounts
    - pnl_attribution.by_strategy: P&L attribution per strategy
    - pnl_attribution.by_broker: P&L attribution per broker
    - drawdown_by_account: Drawdown per account
    """
    logger.info("Generating portfolio report")

    try:
        report = get_portfolio_head().generate_portfolio_report()
        return PortfolioReportResponse(**report)
    except Exception as e:
        logger.error(f"Failed to generate portfolio report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equity", response_model=TotalEquityResponse)
async def get_total_equity():
    """
    Get total portfolio equity across all accounts.

    Returns total equity and account count.
    """
    try:
        result = get_portfolio_head().get_total_equity()
        return TotalEquityResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get total equity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pnl/strategy", response_model=StrategyPnLResponse)
async def get_strategy_pnl(period: str = "all"):
    """
    Get P&L attribution by strategy.

    Args:
        period: Period for attribution (default: all)

    Returns:
        P&L attribution by strategy
    """
    try:
        result = get_portfolio_head().get_strategy_pnl(period=period)
        return StrategyPnLResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get strategy P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pnl/broker", response_model=BrokerPnLResponse)
async def get_broker_pnl(period: str = "all"):
    """
    Get P&L attribution by broker.

    Args:
        period: Period for attribution (default: all)

    Returns:
        P&L attribution by broker
    """
    try:
        result = get_portfolio_head().get_broker_pnl(period=period)
        return BrokerPnLResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get broker P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drawdowns", response_model=AccountDrawdownsResponse)
async def get_account_drawdowns():
    """
    Get drawdown by account.

    Returns:
        Drawdown per account
    """
    try:
        result = get_portfolio_head().get_account_drawdowns()
        return AccountDrawdownsResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get account drawdowns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Story 9.2: Portfolio Metrics & Attribution API Endpoints
# =============================================================================

@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary():
    """
    Get portfolio summary with key metrics.

    AC1: Returns total equity, daily P&L, drawdown, active strategies, and per-account details.
    Also triggers drawdown alert if portfolio drawdown exceeds 10%.
    """
    logger.info("Generating portfolio summary")

    try:
        result = get_portfolio_head().get_portfolio_summary()
        return PortfolioSummaryResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attribution", response_model=PortfolioAttributionResponse)
async def get_portfolio_attribution():
    """
    Get P&L attribution by strategy and broker.

    AC2: Returns P&L attribution per strategy (with equity contribution) and per broker.
    """
    logger.info("Generating portfolio attribution")

    try:
        result = get_portfolio_head().get_attribution()
        return PortfolioAttributionResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get portfolio attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation", response_model=PortfolioCorrelationResponse)
async def get_portfolio_correlation(period_days: int = 30):
    """
    Get correlation matrix of strategy returns.

    AC3: Returns NxN correlation matrix of strategy returns.
    Uses configurable period (default 30 days).

    Args:
        period_days: Number of days for correlation calculation (default: 30)
    """
    logger.info(f"Generating correlation matrix for {period_days} days")

    try:
        result = get_portfolio_head().get_correlation_matrix(period_days=period_days)
        return PortfolioCorrelationResponse(**result)
    except Exception as e:
        logger.error(f"Failed to get correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))
