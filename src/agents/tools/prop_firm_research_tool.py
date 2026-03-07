"""
Prop Firm Research Tool - Analyzes prop firm rules and requirements.
Supports: FundedNext, TrueForex, MyCryptoBuddy
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PropFirm(Enum):
    FUNDEDNEXT = "fundednext"
    TRUEFOREX = "trueforex"
    MYCRYPTOBUDDY = "mycryptobuddy"


class AccountTier(Enum):
    STARTER = "starter"
    STANDARD = "standard"
    PRO = "pro"
    ELITE = "elite"


@dataclass
class PropFirmRules:
    firm: str
    tier: str
    initial_balance: float
    funded_balance: float
    max_drawdown_pct: float
    daily_drawdown_pct: float
    min_trading_days: int
    min_trade_count: int
    profit_target_pct: float
    evaluation_duration_days: int
    leverage: int
    allowed_instruments: List[str]
    excluded_instruments: List[str]
    news_trading_allowed: bool
    ea_allowed: bool
    hedge_allowed: bool
    news_offset_hours: float = 0.0
    monthly_fee: Optional[float] = None
    payout_frequency: str = "monthly"
    withdrawal_fee: Optional[float] = None


@dataclass
class FirmAnalysis:
    firm: str
    tier: str
    overall_score: float
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: str = ""
    suitability: str = ""


def _create_rule(firm: str, tier: str, balance: float, max_dd: float, daily_dd: float,
                 days: int, trades: int, target: float, eval_days: int, lev: int,
                 instruments: List[str], excludes: List[str], news: bool, ea: bool,
                 hedge: bool, news_offset: float = 0.0, fee: float = 0.0, payout: str = "monthly") -> PropFirmRules:
    return PropFirmRules(firm=firm, tier=tier, initial_balance=balance, funded_balance=balance,
                          max_drawdown_pct=max_dd, daily_drawdown_pct=daily_dd, min_trading_days=days,
                          min_trade_count=trades, profit_target_pct=target, evaluation_duration_days=eval_days,
                          leverage=lev, allowed_instruments=instruments, excluded_instruments=excludes,
                          news_trading_allowed=news, ea_allowed=ea, hedge_allowed=hedge,
                          news_offset_hours=news_offset, monthly_fee=fee, payout_frequency=payout)


class PropFirmResearch:
    FIRM_RULES: Dict[str, Dict[str, PropFirmRules]] = {
        "fundednext": {
            "starter": _create_rule("fundednext", "starter", 5000, 10.0, 5.0, 5, 5, 8.0, 30, 100,
                                    ["Forex", "Indices", "Commodities", "Crypto"], [], False, True, True, fee=39.0),
            "standard": _create_rule("fundednext", "standard", 10000, 10.0, 5.0, 5, 5, 8.0, 30, 100,
                                      ["Forex", "Indices", "Commodities", "Crypto"], [], False, True, True, fee=59.0),
            "pro": _create_rule("fundednext", "pro", 25000, 10.0, 5.0, 5, 5, 8.0, 30, 100,
                                ["Forex", "Indices", "Commodities", "Crypto"], [], False, True, True, fee=99.0),
            "elite": _create_rule("fundednext", "elite", 50000, 10.0, 5.0, 5, 5, 8.0, 30, 100,
                                  ["Forex", "Indices", "Commodities", "Crypto"], [], False, True, True, fee=179.0),
        },
        "trueforex": {
            "starter": _create_rule("trueforex", "starter", 5000, 8.0, 4.0, 10, 10, 10.0, 60, 100,
                                    ["Forex", "Indices", "Metals"], ["Crypto", "Commodities", "Stocks"],
                                    True, True, True, 0.5, 35.0, "bi-weekly"),
            "standard": _create_rule("trueforex", "standard", 10000, 8.0, 4.0, 10, 10, 10.0, 60, 100,
                                      ["Forex", "Indices", "Metals"], ["Crypto", "Commodities", "Stocks"],
                                      True, True, True, 0.5, 55.0, "bi-weekly"),
            "pro": _create_rule("trueforex", "pro", 25000, 8.0, 4.0, 10, 10, 10.0, 60, 100,
                                ["Forex", "Indices", "Metals"], ["Crypto", "Commodities", "Stocks"],
                                True, True, True, 0.5, 95.0, "bi-weekly"),
        },
        "mycryptobuddy": {
            "starter": _create_rule("mycryptobuddy", "starter", 10000, 12.0, 6.0, 7, 5, 10.0, 30, 50,
                                     ["Crypto"], ["Forex", "Indices", "Commodities", "Stocks"], False, True, False, fee=29.0),
            "standard": _create_rule("mycryptobuddy", "standard", 25000, 12.0, 6.0, 7, 5, 10.0, 30, 50,
                                       ["Crypto"], ["Forex", "Indices", "Commodities", "Stocks"], False, True, False, fee=49.0),
            "pro": _create_rule("mycryptobuddy", "pro", 50000, 12.0, 6.0, 7, 5, 10.0, 30, 50,
                               ["Crypto"], ["Forex", "Indices", "Commodities", "Stocks"], False, True, False, fee=89.0),
        },
    }

    def __init__(self):
        self._analysis_cache: Dict[str, FirmAnalysis] = {}

    def get_firms(self) -> List[str]:
        return [firm.value for firm in PropFirm]

    def get_tiers(self, firm: str) -> List[str]:
        if firm.lower() not in self.FIRM_RULES:
            return []
        return list(self.FIRM_RULES[firm.lower()].keys())

    def get_rules(self, firm: str, tier: str = "standard") -> Optional[Dict[str, Any]]:
        firm_lower, tier_lower = firm.lower(), tier.lower()
        if firm_lower not in self.FIRM_RULES or tier_lower not in self.FIRM_RULES.get(firm_lower, {}):
            return None
        r = self.FIRM_RULES[firm_lower][tier_lower]
        return {"firm": r.firm, "tier": r.tier, "initial_balance": r.initial_balance,
                "funded_balance": r.funded_balance, "max_drawdown_pct": r.max_drawdown_pct,
                "daily_drawdown_pct": r.daily_drawdown_pct, "min_trading_days": r.min_trading_days,
                "min_trade_count": r.min_trade_count, "profit_target_pct": r.profit_target_pct,
                "evaluation_duration_days": r.evaluation_duration_days, "leverage": r.leverage,
                "allowed_instruments": r.allowed_instruments, "excluded_instruments": r.excluded_instruments,
                "news_trading_allowed": r.news_trading_allowed, "ea_allowed": r.ea_allowed,
                "hedge_allowed": r.hedge_allowed, "news_offset_hours": r.news_offset_hours,
                "monthly_fee": r.monthly_fee, "payout_frequency": r.payout_frequency, "withdrawal_fee": r.withdrawal_fee}

    def firm_analysis(self, firm: str, tier: str = "standard") -> Dict[str, Any]:
        firm_lower, tier_lower = firm.lower(), tier.lower()
        cache_key = f"{firm_lower}_{tier_lower}"
        if cache_key in self._analysis_cache:
            a = self._analysis_cache[cache_key]
            return {"firm": a.firm, "tier": a.tier, "overall_score": a.overall_score, "pros": a.pros,
                    "cons": a.cons, "recommendations": a.recommendations, "risk_assessment": a.risk_assessment, "suitability": a.suitability}
        if firm_lower not in self.FIRM_RULES:
            return {"error": f"Unknown firm: {firm}. Supported: {self.get_firms()}"}
        if tier_lower not in self.FIRM_RULES.get(firm_lower, {}):
            return {"error": f"Unknown tier: {tier}. Available: {self.get_tiers(firm)}"}
        rules = self.FIRM_RULES[firm_lower][tier_lower]
        analysis = self._analyze_firm(rules)
        self._analysis_cache[cache_key] = analysis
        return {"firm": analysis.firm, "tier": analysis.tier, "overall_score": analysis.overall_score,
                "pros": analysis.pros, "cons": analysis.cons, "recommendations": analysis.recommendations,
                "risk_assessment": analysis.risk_assessment, "suitability": analysis.suitability}

    def _analyze_firm(self, rules: PropFirmRules) -> FirmAnalysis:
        pros, cons, recommendations, score = [], [], [], 50.0
        if rules.max_drawdown_pct >= 10:
            pros.append("Generous max drawdown (10%+)"); score += 10
        elif rules.max_drawdown_pct >= 8:
            pros.append("Moderate max drawdown (8%)"); score += 5
        else:
            cons.append(f"Strict max drawdown ({rules.max_drawdown_pct}%)"); score -= 10
        if rules.daily_drawdown_pct >= 5:
            pros.append("Flexible daily drawdown (5%+)"); score += 5
        else:
            cons.append(f"Tight daily drawdown ({rules.daily_drawdown_pct}%)"); score -= 5
        if rules.profit_target_pct <= 8:
            pros.append("Achievable profit target (8% or less)"); score += 10
        elif rules.profit_target_pct <= 10:
            score += 5
        else:
            cons.append(f"Aggressive profit target ({rules.profit_target_pct}%)"); score -= 5
        if rules.monthly_fee and rules.monthly_fee <= 40:
            pros.append("Low monthly fee"); score += 5
        elif rules.monthly_fee and rules.monthly_fee <= 60:
            score += 2
        if rules.ea_allowed:
            pros.append("EA/automated trading allowed"); score += 5
        else:
            cons.append("No EA/automated trading")
        if rules.hedge_allowed:
            pros.append("Hedging allowed"); score += 5
        if rules.news_trading_allowed:
            pros.append("News trading permitted"); score += 5
        if len(rules.allowed_instruments) >= 4:
            pros.append("Wide instrument range"); score += 5
        if rules.leverage >= 100:
            pros.append("High leverage (100:1)"); score += 5
        if rules.daily_drawdown_pct < 5:
            recommendations.append("Use tighter daily stop-loss")
        if rules.min_trade_count > 5:
            recommendations.append("Meet minimum trades early")
        if not rules.news_trading_allowed:
            recommendations.append("Avoid news events")
        if rules.hedge_allowed:
            recommendations.append("Consider hedging for overnight gaps")
        risk = "Low Risk" if score >= 70 else "Medium Risk" if score >= 50 else "High Risk"
        suitability = {"mycryptobuddy": "Best for crypto-focused traders", "trueforex": "Best for news trading flexibility", "fundednext": "Best for beginners, diverse instruments"}.get(rules.firm, "")
        return FirmAnalysis(firm=rules.firm, tier=rules.tier, overall_score=min(max(score, 0), 100),
                            pros=pros, cons=cons, recommendations=recommendations,
                            risk_assessment=f"{risk} - Good balance" if score >= 50 else f"{risk} - Strict rules", suitability=suitability)

    def compare_firms(self, firms: List[str], tier: str = "standard") -> Dict[str, Any]:
        comparison = {"tier": tier, "firms": []}
        for firm in firms:
            rules, analysis = self.get_rules(firm, tier), self.firm_analysis(firm, tier)
            if rules and "error" not in analysis:
                comparison["firms"].append({"firm": firm, "initial_balance": rules["initial_balance"],
                    "monthly_fee": rules["monthly_fee"], "max_drawdown_pct": rules["max_drawdown_pct"],
                    "daily_drawdown_pct": rules["daily_drawdown_pct"], "profit_target_pct": rules["profit_target_pct"],
                    "evaluation_days": rules["evaluation_duration_days"], "leverage": rules["leverage"],
                    "ea_allowed": rules["ea_allowed"], "hedge_allowed": rules["hedge_allowed"],
                    "score": analysis["overall_score"], "risk_assessment": analysis["risk_assessment"]})
        comparison["firms"].sort(key=lambda x: x["score"], reverse=True)
        return comparison


PROP_FIRM_TOOL_SCHEMAS = [
    {"name": "get_firms", "description": "Get supported prop firms", "parameters": {"type": "object", "properties": {}}},
    {"name": "get_tiers", "description": "Get tiers for a firm", "parameters": {"type": "object", "properties": {"firm": {"type": "string"}}, "required": ["firm"]}},
    {"name": "get_rules", "description": "Get rules for firm/tier", "parameters": {"type": "object", "properties": {"firm": {"type": "string"}, "tier": {"type": "string"}}, "required": ["firm"]}},
    {"name": "firm_analysis", "description": "Analyze firm rules with pros/cons", "parameters": {"type": "object", "properties": {"firm": {"type": "string"}, "tier": {"type": "string"}}, "required": ["firm"]}},
    {"name": "compare_firms", "description": "Compare multiple firms", "parameters": {"type": "object", "properties": {"firms": {"type": "array", "items": {"type": "string"}}, "tier": {"type": "string"}}, "required": ["firms"]}},
]


def get_prop_firm_tool_schemas() -> List[Dict]:
    return PROP_FIRM_TOOL_SCHEMAS


_default_instance: Optional[PropFirmResearch] = None


def get_default_instance() -> PropFirmResearch:
    global _default_instance
    if _default_instance is None:
        _default_instance = PropFirmResearch()
    return _default_instance
