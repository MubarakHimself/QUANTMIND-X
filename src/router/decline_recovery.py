"""
Decline and Recovery Loop

Automated workflow for handling underperforming bots.

Spec (Section 8.2):
- Detect: Live Monitor Subagent detects win rate falling below threshold over last N trades,
          or 3-loss-in-a-row rule triggered 3 times in a week
- Flag: Anomaly logged with timestamp, regime state at time of decline, delta from expected performance
- Quarantine: Bot moved to paper-only mode (not killed). @primal tag removed, @paper_only applied.
- Diagnose: Risk Agent examines quarantine report and regime data during failure period
- Report: Risk Agent writes diagnosis report answering Q17-Q20 from lifecycle report framework
- Improve: Research Agent and Development Agent receive diagnosis and propose parameter changes
- Re-validate: Improved variant goes back to Workflow 2 from paper trading stage
- Promote/Retire: If improved variant passes paper trading -> re-promote to live.
                 If fails twice -> retire original variant.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import uuid

logger = logging.getLogger(__name__)


class DeclineState(Enum):
    """States in the decline and recovery workflow."""
    NORMAL = "normal"              # Bot performing within expected parameters
    FLAGGED = "flagged"           # Decline detected, awaiting review
    QUARANTINED = "quarantined"   # Bot moved to paper-only, under review
    DIAGNOSING = "diagnosing"     # Risk Agent examining failure period
    IMPROVING = "improving"        # Research/Dev proposing parameter changes
    PAPER_RETEST = "paper_retest" # Improved variant in paper trading re-validation
    RECOVERED = "recovered"       # Variant passed paper trading, promoted to live
    RETIRED = "retired"           # Bot variant failed twice, gracefully deprecated


@dataclass
class DeclineRecord:
    """
    Record of a bot's decline incident.

    Tracks the full lifecycle of a decline from initial flag through recovery or retirement.
    """
    bot_id: str
    flagged_at: datetime
    flag_reason: str
    regime_at_flag: str  # HMM regime state when decline was detected
    performance_delta: float  # Delta from expected performance (negative = underperforming)
    current_state: DeclineState = DeclineState.FLAGGED

    # Tracking fields
    quarantine_at: Optional[datetime] = None
    diagnosis_at: Optional[datetime] = None
    diagnosis_report: Optional[str] = None
    improvement_variant_id: Optional[str] = None
    improvement_proposed_at: Optional[datetime] = None
    paper_retest_started_at: Optional[datetime] = None
    paper_retest_passed: Optional[bool] = None
    recovered_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    # Failure tracking
    failure_count: int = 0  # Number of times this lineage has failed paper retest
    original_bot_id: Optional[str] = None  # If this is a variant, reference to original

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bot_id": self.bot_id,
            "flagged_at": self.flagged_at.isoformat() if self.flagged_at else None,
            "flag_reason": self.flag_reason,
            "regime_at_flag": self.regime_at_flag,
            "performance_delta": self.performance_delta,
            "current_state": self.current_state.value,
            "quarantine_at": self.quarantine_at.isoformat() if self.quarantine_at else None,
            "diagnosis_at": self.diagnosis_at.isoformat() if self.diagnosis_at else None,
            "diagnosis_report": self.diagnosis_report,
            "improvement_variant_id": self.improvement_variant_id,
            "improvement_proposed_at": self.improvement_proposed_at.isoformat() if self.improvement_proposed_at else None,
            "paper_retest_started_at": self.paper_retest_started_at.isoformat() if self.paper_retest_started_at else None,
            "paper_retest_passed": self.paper_retest_passed,
            "recovered_at": self.recovered_at.isoformat() if self.recovered_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "failure_count": self.failure_count,
            "original_bot_id": self.original_bot_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeclineRecord":
        """Deserialize from dictionary."""
        return cls(
            bot_id=data["bot_id"],
            flagged_at=datetime.fromisoformat(data["flagged_at"]) if data.get("flagged_at") else datetime.now(timezone.utc),
            flag_reason=data.get("flag_reason", ""),
            regime_at_flag=data.get("regime_at_flag", "UNKNOWN"),
            performance_delta=data.get("performance_delta", 0.0),
            current_state=DeclineState(data.get("current_state", "flagged")),
            quarantine_at=datetime.fromisoformat(data["quarantine_at"]) if data.get("quarantine_at") else None,
            diagnosis_at=datetime.fromisoformat(data["diagnosis_at"]) if data.get("diagnosis_at") else None,
            diagnosis_report=data.get("diagnosis_report"),
            improvement_variant_id=data.get("improvement_variant_id"),
            improvement_proposed_at=datetime.fromisoformat(data["improvement_proposed_at"]) if data.get("improvement_proposed_at") else None,
            paper_retest_started_at=datetime.fromisoformat(data["paper_retest_started_at"]) if data.get("paper_retest_started_at") else None,
            paper_retest_passed=data.get("paper_retest_passed"),
            recovered_at=datetime.fromisoformat(data["recovered_at"]) if data.get("recovered_at") else None,
            retired_at=datetime.fromisoformat(data["retired_at"]) if data.get("retired_at") else None,
            failure_count=data.get("failure_count", 0),
            original_bot_id=data.get("original_bot_id"),
        )


@dataclass
class DiagnosisReport:
    """
    Structured diagnosis report for a declined bot.

    Answers Q17-Q20 from lifecycle report framework:
    - Q17: What market conditions/regime was the bot trading in?
    - Q18: What specific parameters were misaligned with conditions?
    - Q19: What is the proposed fix and expected outcome?
    - Q20: What is the risk of the proposed fix?
    """
    bot_id: str
    diagnosis_id: str
    created_at: datetime
    regime_analysis: str  # Q17: Market conditions during failure
    parameter_misalignment: str  # Q18: What was misaligned
    proposed_fix: str  # Q19: Proposed changes and expected outcome
    risk_assessment: str  # Q20: Risk of proposed fix
    confidence_score: float  # 0.0-1.0 confidence in diagnosis
    recommended_trades_for_validation: int = 50  # Min trades for paper re-validation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bot_id": self.bot_id,
            "diagnosis_id": self.diagnosis_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "regime_analysis": self.regime_analysis,
            "parameter_misalignment": self.parameter_misalignment,
            "proposed_fix": self.proposed_fix,
            "risk_assessment": self.risk_assessment,
            "confidence_score": self.confidence_score,
            "recommended_trades_for_validation": self.recommended_trades_for_validation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisReport":
        """Deserialize from dictionary."""
        return cls(
            bot_id=data["bot_id"],
            diagnosis_id=data["diagnosis_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            regime_analysis=data["regime_analysis"],
            parameter_misalignment=data["parameter_misalignment"],
            proposed_fix=data["proposed_fix"],
            risk_assessment=data["risk_assessment"],
            confidence_score=data.get("confidence_score", 0.5),
            recommended_trades_for_validation=data.get("recommended_trades_for_validation", 50),
        )


class DeclineRecoveryEngine:
    """
    Engine for managing bot decline and recovery workflow.

    Handles detection, flagging, quarantine, diagnosis, and recovery of underperforming bots.
    """

    # Default thresholds
    DEFAULT_WIN_RATE_THRESHOLD = 0.45  # 45% win rate below this is concerning
    DEFAULT_LOOKBACK_TRADES = 50  # Number of trades to analyze
    THREE_LOSS_WEEKLY_THRESHOLD = 3  # 3 separate days with 3-loss streak triggers flag

    def __init__(
        self,
        win_rate_threshold: float = DEFAULT_WIN_RATE_THRESHOLD,
        lookback_trades: int = DEFAULT_LOOKBACK_TRADES,
    ):
        """
        Initialize DeclineRecoveryEngine.

        Args:
            win_rate_threshold: Win rate below this triggers decline detection
            lookback_trades: Number of recent trades to analyze
        """
        self.win_rate_threshold = win_rate_threshold
        self.lookback_trades = lookback_trades

        # In-memory storage for decline records (in production, persist to DB)
        self._decline_records: Dict[str, DeclineRecord] = {}
        self._diagnosis_reports: Dict[str, DiagnosisReport] = {}
        self._bot_state_cache: Dict[str, DeclineState] = {}

        logger.info(
            f"DeclineRecoveryEngine initialized: "
            f"win_rate_threshold={win_rate_threshold}, "
            f"lookback_trades={lookback_trades}"
        )

    def detect_decline(
        self,
        bot_id: str,
        recent_trades: List[Dict[str, Any]],
        regime_state: str = "UNKNOWN",
        expected_win_rate: float = 0.50,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if a bot is declining.

        Detection criteria:
        1. Win rate falling below threshold over last N trades
        2. 3-loss-in-a-row triggered 3 times in a week (tracked separately)

        Args:
            bot_id: Bot identifier
            recent_trades: List of recent trade records with 'is_loss' field
            regime_state: Current HMM regime state
            expected_win_rate: Expected win rate for delta calculation

        Returns:
            Tuple of (is_declining, reason)
        """
        if len(recent_trades) < 10:
            # Not enough data to make a determination
            return False, None

        # Calculate win rate over lookback period
        lookback = recent_trades[-self.lookback_trades:]
        wins = sum(1 for t in lookback if not t.get("is_loss", False))
        total = len(lookback)
        current_win_rate = wins / total if total > 0 else 0.0

        # Check win rate threshold
        if current_win_rate < self.win_rate_threshold:
            delta = current_win_rate - expected_win_rate
            reason = (
                f"Win rate {current_win_rate:.1%} below threshold {self.win_rate_threshold:.1%} "
                f"over last {total} trades (delta: {delta:+.1%})"
            )
            logger.warning(f"Decline detected for {bot_id}: {reason}")
            return True, reason

        # Check for 3-loss-in-a-row streaks (but this is tracked by circuit breaker)
        # Here we just log it - the actual weekly tracking is in BotCircuitBreakerManager
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in reversed(lookback):
            if trade.get("is_loss", False):
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Log but don't trigger decline here - circuit breaker handles 3-loss tracking
        if max_consecutive_losses >= 3:
            logger.info(
                f"Bot {bot_id} had {max_consecutive_losses} consecutive losses "
                f"in recent history (circuit breaker handles weekly tracking)"
            )

        return False, None

    def detect_3loss_weekly_flag(
        self,
        bot_id: str,
        daily_loss_streak_days: int,
        circuit_breaker_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect decline based on 3-loss-in-a-row weekly threshold.

        This is called when the circuit breaker detects the 3rd day with
        3-loss-in-a-row within a week.

        Args:
            bot_id: Bot identifier
            daily_loss_streak_days: Number of days with 3-loss streak this week
            circuit_breaker_state: Current circuit breaker state

        Returns:
            Tuple of (should_flag, reason)
        """
        if daily_loss_streak_days >= self.THREE_LOSS_WEEKLY_THRESHOLD:
            reason = (
                f"3-loss-in-a-row triggered {daily_loss_streak_days} times this week "
                f"(threshold: {self.THREE_LOSS_WEEKLY_THRESHOLD})"
            )
            logger.warning(f"Decline detected for {bot_id}: {reason}")
            return True, reason

        return False, None

    def flag_bot(
        self,
        bot_id: str,
        reason: str,
        regime_state: str,
        performance_delta: float,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> DeclineRecord:
        """
        Flag a bot for decline review.

        Creates a DeclineRecord and transitions bot to FLAGGED state.

        Args:
            bot_id: Bot identifier
            reason: Reason for flagging
            regime_state: HMM regime state at time of flag
            performance_delta: Delta from expected performance
            recent_trades: Optional trade history for the record

        Returns:
            Created DeclineRecord
        """
        # Check if already flagged
        existing = self._decline_records.get(bot_id)
        if existing and existing.current_state != DeclineState.RECOVERED:
            logger.info(f"Bot {bot_id} already has active decline record, updating reason")
            existing.flag_reason = reason
            existing.regime_at_flag = regime_state
            existing.performance_delta = performance_delta
            return existing

        # Create new decline record
        record = DeclineRecord(
            bot_id=bot_id,
            flagged_at=datetime.now(timezone.utc),
            flag_reason=reason,
            regime_at_flag=regime_state,
            performance_delta=performance_delta,
            current_state=DeclineState.FLAGGED,
        )

        self._decline_records[bot_id] = record
        self._bot_state_cache[bot_id] = DeclineState.FLAGGED

        logger.info(f"Bot {bot_id} flagged for decline: {reason}")
        return record

    def quarantine_bot(
        self,
        bot_id: str,
        trading_mode: str = "PAPER",
    ) -> DeclineRecord:
        """
        Quarantine a flagged bot.

        Moves bot to paper-only mode. Real execution is suspended.
        Tag changes: @primal -> @paper_only

        Args:
            bot_id: Bot identifier
            trading_mode: Trading mode to set (default PAPER)

        Returns:
            Updated DeclineRecord
        """
        record = self._decline_records.get(bot_id)

        if record is None:
            # Create a new record if none exists (shouldn't happen normally)
            record = self.flag_bot(
                bot_id=bot_id,
                reason="Quarantine requested without prior flag",
                regime_state="UNKNOWN",
                performance_delta=0.0,
            )

        # Transition state
        record.current_state = DeclineState.QUARANTINED
        record.quarantine_at = datetime.now(timezone.utc)

        self._bot_state_cache[bot_id] = DeclineState.QUARANTINED

        logger.info(f"Bot {bot_id} quarantined (mode: {trading_mode})")
        return record

    def diagnose_bot(
        self,
        bot_id: str,
        regime_data: Optional[Dict[str, Any]] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> DiagnosisReport:
        """
        Diagnose a quarantined bot.

        Risk Agent examines quarantine report and regime data during failure period.
        Produces a structured diagnosis answering Q17-Q20.

        Args:
            bot_id: Bot identifier
            regime_data: Optional regime data during failure period
            trade_history: Optional trade history for analysis

        Returns:
            DiagnosisReport with structured findings
        """
        record = self._decline_records.get(bot_id)

        if record is None:
            raise ValueError(f"No decline record found for bot {bot_id}")

        # Transition to diagnosing state
        record.current_state = DeclineState.DIAGNOSING
        record.diagnosis_at = datetime.now(timezone.utc)
        self._bot_state_cache[bot_id] = DeclineState.DIAGNOSING

        # Create diagnosis report (in production, this would use AI analysis)
        # For now, create a structured report based on available data
        diagnosis = DiagnosisReport(
            bot_id=bot_id,
            diagnosis_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            regime_analysis=self._analyze_regime_failure(regime_data, record),
            parameter_misalignment=self._analyze_parameter_misalignment(bot_id, trade_history, record),
            proposed_fix=self._generate_proposed_fix(bot_id, record),
            risk_assessment=self._assess_fix_risk(bot_id, record),
            confidence_score=0.75,  # Placeholder confidence
            recommended_trades_for_validation=50,
        )

        self._diagnosis_reports[diagnosis.diagnosis_id] = diagnosis
        record.diagnosis_report = diagnosis.diagnosis_id

        logger.info(f"Diagnosis completed for {bot_id}: {diagnosis.diagnosis_id}")
        return diagnosis

    def _analyze_regime_failure(
        self,
        regime_data: Optional[Dict[str, Any]],
        record: DeclineRecord,
    ) -> str:
        """Analyze what regime conditions led to failure."""
        if regime_data:
            return (
                f"Bot was trading in {record.regime_at_flag} regime. "
                f"Regime data: {regime_data.get('description', 'N/A')}. "
                f"Performance delta: {record.performance_delta:+.1%}"
            )
        return (
            f"Bot was trading in {record.regime_at_flag} regime. "
            f"Performance delta: {record.performance_delta:+.1%}. "
            f"Detailed regime analysis pending data."
        )

    def _analyze_parameter_misalignment(
        self,
        bot_id: str,
        trade_history: Optional[List[Dict[str, Any]]],
        record: DeclineRecord,
    ) -> str:
        """Analyze what parameters were misaligned with conditions."""
        # Placeholder - in production would analyze actual parameters
        return (
            f"Underlying cause analysis for {bot_id} indicates potential "
            f"parameter misalignment with current market conditions. "
            f"Flag reason: {record.flag_reason}. "
            f"Detailed parameter review required."
        )

    def _generate_proposed_fix(self, bot_id: str, record: DeclineRecord) -> str:
        """Generate proposed parameter changes."""
        return (
            f"Proposed fix for {bot_id}: Adjust risk parameters to better "
            f"match {record.regime_at_flag} regime conditions. "
            f"Reduce position sizing by 20-30% and widen stop loss by 10-15 pips "
            f"to accommodate increased volatility in current regime."
        )

    def _assess_fix_risk(self, bot_id: str, record: DeclineRecord) -> str:
        """Assess risk of the proposed fix."""
        return (
            f"Risk assessment for {bot_id}: MODERATE. "
            f"Proposed changes reduce exposure which limits downside but may "
            f"also reduce win rate. Risk of continued decline if regime "
            f"changes again is MEDIUM. Monitor closely during paper retest."
        )

    def create_improvement_variant(
        self,
        bot_id: str,
        diagnosis: DiagnosisReport,
    ) -> str:
        """
        Create an improved variant of a declined bot.

        Forks a new variant with proposed parameter changes.

        Args:
            bot_id: Original bot identifier
            diagnosis: DiagnosisReport with proposed fixes

        Returns:
            new_variant_id for the created variant
        """
        record = self._decline_records.get(bot_id)

        if record is None:
            raise ValueError(f"No decline record found for bot {bot_id}")

        # Generate new variant ID
        variant_id = f"{bot_id}_variant_{uuid.uuid4().hex[:8]}"

        # Transition to improving state
        record.current_state = DeclineState.IMPROVING
        record.improvement_variant_id = variant_id
        record.improvement_proposed_at = datetime.now(timezone.utc)
        record.original_bot_id = bot_id

        self._bot_state_cache[bot_id] = DeclineState.IMPROVING

        logger.info(f"Improvement variant created for {bot_id}: {variant_id}")
        return variant_id

    def start_paper_retest(self, variant_id: str, original_bot_id: str) -> None:
        """
        Start paper retest phase for an improved variant.

        Args:
            variant_id: The improved variant bot ID
            original_bot_id: Reference to original bot
        """
        record = self._decline_records.get(original_bot_id)

        if record is None:
            raise ValueError(f"No decline record found for original bot {original_bot_id}")

        record.current_state = DeclineState.PAPER_RETEST
        record.paper_retest_started_at = datetime.now(timezone.utc)

        self._bot_state_cache[variant_id] = DeclineState.PAPER_RETEST

        logger.info(f"Paper retest started for variant {variant_id}")

    def complete_paper_retest(
        self,
        variant_id: str,
        original_bot_id: str,
        passed: bool,
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Complete paper retest phase.

        Args:
            variant_id: The variant that was retested
            original_bot_id: Reference to original bot
            passed: Whether paper retest passed
            performance_data: Optional performance metrics from retest

        Returns:
            True if variant passed and can be promoted
        """
        record = self._decline_records.get(original_bot_id)

        if record is None:
            raise ValueError(f"No decline record found for original bot {original_bot_id}")

        record.paper_retest_passed = passed

        if passed:
            # Transition to recovered
            record.current_state = DeclineState.RECOVERED
            record.recovered_at = datetime.now(timezone.utc)
            self._bot_state_cache[original_bot_id] = DeclineState.RECOVERED
            self._bot_state_cache[variant_id] = DeclineState.RECOVERED

            logger.info(f"Variant {variant_id} passed paper retest, recovered")
            return True
        else:
            # Increment failure count and check if should retire
            record.failure_count += 1
            logger.warning(
                f"Variant {variant_id} failed paper retest "
                f"(failure count: {record.failure_count})"
            )

            if record.failure_count >= 2:
                # Retire after 2 failures
                return self.retire_bot(original_bot_id)

            return False

    def promote_variant(self, variant_id: str, original_bot_id: str) -> Dict[str, Any]:
        """
        Promote an improved variant to live trading.

        Args:
            variant_id: The variant to promote
            original_bot_id: Reference to original bot

        Returns:
            Promotion result dict
        """
        record = self._decline_records.get(original_bot_id)

        if record is None:
            return {
                "success": False,
                "reason": f"No decline record found for original bot {original_bot_id}",
            }

        if record.current_state != DeclineState.RECOVERED:
            return {
                "success": False,
                "reason": f"Bot not in RECOVERED state (current: {record.current_state.value})",
            }

        # In production, this would trigger the actual promotion workflow
        logger.info(
            f"Variant {variant_id} promoted to live trading "
            f"(replacing {original_bot_id})"
        )

        return {
            "success": True,
            "variant_id": variant_id,
            "original_bot_id": original_bot_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

    def retire_bot(self, bot_id: str) -> bool:
        """
        Retire a bot after repeated failures.

        Gracefully deprecates the bot variant.

        Args:
            bot_id: Bot to retire

        Returns:
            True if retired successfully
        """
        record = self._decline_records.get(bot_id)

        if record is None:
            logger.warning(f"No decline record found for bot {bot_id}, retiring anyway")
            # Create a record if none exists
            record = DeclineRecord(
                bot_id=bot_id,
                flagged_at=datetime.now(timezone.utc),
                flag_reason="Retirement requested",
                regime_at_flag="UNKNOWN",
                performance_delta=0.0,
                current_state=DeclineState.RETIRED,
            )
            self._decline_records[bot_id] = record

        record.current_state = DeclineState.RETIRED
        record.retired_at = datetime.now(timezone.utc)

        self._bot_state_cache[bot_id] = DeclineState.RETIRED

        logger.warning(f"Bot {bot_id} retired after {record.failure_count} failures")
        return True

    def get_decline_state(self, bot_id: str) -> DeclineState:
        """
        Get current decline state for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Current DeclineState
        """
        return self._bot_state_cache.get(bot_id, DeclineState.NORMAL)

    def get_decline_record(self, bot_id: str) -> Optional[DeclineRecord]:
        """Get decline record for a bot."""
        return self._decline_records.get(bot_id)

    def get_diagnosis_report(self, diagnosis_id: str) -> Optional[DiagnosisReport]:
        """Get diagnosis report by ID."""
        return self._diagnosis_reports.get(diagnosis_id)

    def list_bots_in_state(self, state: DeclineState) -> List[str]:
        """List all bots currently in a given state."""
        return [
            bot_id for bot_id, s in self._bot_state_cache.items()
            if s == state
        ]

    def list_all_declined(self) -> List[DeclineRecord]:
        """List all bots with active decline records."""
        return [
            r for r in self._decline_records.values()
            if r.current_state not in [DeclineState.RECOVERED, DeclineState.RETIRED]
        ]


class DeclineRecoveryWorkflow:
    """
    Orchestrates the full Detect -> Flag -> Quarantine -> Diagnose -> Improve ->
    Re-validate -> Promote/Retire loop.

    This class coordinates the workflow between the various components:
    - Live Monitor Subagent (detection)
    - Risk Agent (diagnosis)
    - Research Agent (improvements)
    - Development Agent (parameter changes)
    """

    def __init__(self, engine: Optional[DeclineRecoveryEngine] = None):
        """
        Initialize workflow orchestrator.

        Args:
            engine: Optional DeclineRecoveryEngine instance
        """
        self.engine = engine or DeclineRecoveryEngine()

        # Workflow state callbacks (set by external systems)
        self._on_bot_quarantined: Optional[callable] = None
        self._on_diagnosis_ready: Optional[callable] = None
        self._on_improvement_proposed: Optional[callable] = None
        self._on_paper_retest_started: Optional[callable] = None
        self._on_bot_recovered: Optional[callable] = None
        self._on_bot_retired: Optional[callable] = None

    def set_callback(
        self,
        event: str,
        callback: callable,
    ) -> None:
        """
        Set workflow event callbacks.

        Args:
            event: Event name ('bot_quarantined', 'diagnosis_ready', etc.)
            callback: Function to call when event occurs
        """
        callback_map = {
            'bot_quarantined': '_on_bot_quarantined',
            'diagnosis_ready': '_on_diagnosis_ready',
            'improvement_proposed': '_on_improvement_proposed',
            'paper_retest_started': '_on_paper_retest_started',
            'bot_recovered': '_on_bot_recovered',
            'bot_retired': '_on_bot_retired',
        }

        attr = callback_map.get(event)
        if attr:
            setattr(self, attr, callback)
        else:
            raise ValueError(f"Unknown event: {event}")

    def run_detection(
        self,
        bot_id: str,
        recent_trades: List[Dict[str, Any]],
        regime_state: str,
        expected_win_rate: float = 0.50,
        daily_loss_streak_days: int = 0,
        circuit_breaker_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Run decline detection for a bot.

        Args:
            bot_id: Bot identifier
            recent_trades: Recent trade history
            regime_state: Current HMM regime
            expected_win_rate: Expected win rate
            daily_loss_streak_days: Days with 3-loss streak this week
            circuit_breaker_state: Circuit breaker state for 3-loss detection

        Returns:
            Tuple of (declined, reason)
        """
        # Check win rate threshold
        declined, reason = self.engine.detect_decline(
            bot_id=bot_id,
            recent_trades=recent_trades,
            regime_state=regime_state,
            expected_win_rate=expected_win_rate,
        )

        if declined:
            return True, reason

        # Check 3-loss weekly threshold
        if circuit_breaker_state and daily_loss_streak_days >= 3:
            declined, reason = self.engine.detect_3loss_weekly_flag(
                bot_id=bot_id,
                daily_loss_streak_days=daily_loss_streak_days,
                circuit_breaker_state=circuit_breaker_state,
            )

        return declined, reason

    def execute_full_workflow(
        self,
        bot_id: str,
        reason: str,
        regime_state: str,
        performance_delta: float,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full decline and recovery workflow.

        Args:
            bot_id: Bot identifier
            reason: Reason for decline
            regime_state: HMM regime at time of decline
            performance_delta: Delta from expected performance
            trade_history: Optional trade history for diagnosis
            regime_data: Optional regime data for analysis

        Returns:
            Dict with workflow results
        """
        results = {
            "bot_id": bot_id,
            "steps_completed": [],
        }

        # Step 1: Flag
        record = self.engine.flag_bot(
            bot_id=bot_id,
            reason=reason,
            regime_state=regime_state,
            performance_delta=performance_delta,
        )
        results["steps_completed"].append("flag")

        # Step 2: Quarantine
        record = self.engine.quarantine_bot(bot_id=bot_id)
        results["steps_completed"].append("quarantine")

        if self._on_bot_quarantined:
            self._on_bot_quarantined(bot_id, record)

        # Step 3: Diagnose
        diagnosis = self.engine.diagnose_bot(
            bot_id=bot_id,
            regime_data=regime_data,
            trade_history=trade_history,
        )
        results["steps_completed"].append("diagnose")
        results["diagnosis"] = diagnosis.to_dict()

        if self._on_diagnosis_ready:
            self._on_diagnosis_ready(bot_id, diagnosis)

        # Step 4: Create improvement variant
        variant_id = self.engine.create_improvement_variant(
            bot_id=bot_id,
            diagnosis=diagnosis,
        )
        results["steps_completed"].append("improve")
        results["variant_id"] = variant_id

        if self._on_improvement_proposed:
            self._on_improvement_proposed(bot_id, variant_id, diagnosis)

        return results

    def handle_paper_retest_result(
        self,
        variant_id: str,
        original_bot_id: str,
        passed: bool,
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle paper retest completion.

        Args:
            variant_id: Variant that was retested
            original_bot_id: Original bot ID
            passed: Whether retest passed
            performance_data: Optional performance metrics

        Returns:
            Dict with result
        """
        success = self.engine.complete_paper_retest(
            variant_id=variant_id,
            original_bot_id=original_bot_id,
            passed=passed,
            performance_data=performance_data,
        )

        record = self.engine.get_decline_record(original_bot_id)

        if record and record.current_state == DeclineState.RECOVERED:
            if self._on_bot_recovered:
                self._on_bot_recovered(variant_id, original_bot_id)

            # Auto-promote
            promotion_result = self.engine.promote_variant(
                variant_id=variant_id,
                original_bot_id=original_bot_id,
            )

            return {
                "success": True,
                "passed": True,
                "variant_id": variant_id,
                "promotion": promotion_result,
            }
        elif record and record.current_state == DeclineState.RETIRED:
            if self._on_bot_retired:
                self._on_bot_retired(original_bot_id, record.failure_count)

            return {
                "success": False,
                "passed": False,
                "retired": True,
                "failure_count": record.failure_count,
            }

        return {
            "success": False,
            "passed": passed,
            "current_state": record.current_state.value if record else "UNKNOWN",
        }
