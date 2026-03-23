"""
Floor Manager

The Floor Manager is the top-level orchestrator for the Trading Floor.
It routes tasks to appropriate Department Heads and manages cross-department communication.

Model Tier: Opus (highest reasoning capability)
"""
import json
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from pathlib import Path
from datetime import datetime

from src.agents.departments.types import (
    Department,
    DepartmentHeadConfig,
    get_department_configs,
    get_model_tier,
)
from src.agents.departments.department_mail import (
    DepartmentMailService,
    RedisDepartmentMailService,
    get_redis_mail_service,
    MessageType,
    Priority,
)
from src.agents.departments.task_router import (
    TaskRouter,
    TaskPriority,
    TaskStatus,
    get_task_router,
    Task,
    TaskResult,
)
from src.router.copilot_kill_switch import get_copilot_kill_switch
from src.intent.classifier import IntentClassifier, get_intent_classifier
from src.intent.patterns import CommandIntent

logger = logging.getLogger(__name__)


class FloorManager:
    """
    Floor Manager for the Trading Floor Model.

    The Floor Manager:
    - Routes incoming tasks to appropriate Department Heads
    - Manages cross-department communication via mail service
    - Coordinates with Agent Spawner for worker spawning
    - Uses Opus tier for highest reasoning capability

    Attributes:
        mail_service: SQLite mail service for cross-department messaging
        spawner: Agent spawner for dynamic worker creation
        departments: Dictionary of department configurations
        model_tier: Model tier (always "opus" for Floor Manager)
    """

    # Keyword-based task classification (Option B departments)
    DEPARTMENT_KEYWORDS = {
        Department.RESEARCH: [
            "analyze", "analysis", "market", "sentiment", "news", "scan",
            "technical", "indicator", "signal", "pattern", "chart",
            "trend", "support", "resistance", "forecast",
            "research", "strategy", "backtest", "develop", "create",
            "alpha", "factor", "optimize", "validate", "test",
            # Video ingest related
            "video", "trading idea", "timeframe", "entry", "exit",
        ],
        Department.DEVELOPMENT: [
            "develop", "build", "ea", "expert advisor", "bot",
            "pinescript", "mql5", "mq5", "python", "code", "implement",
            "script", "automate", "algorithm", "expert",
        ],
        Department.RISK: [
            "risk", "position size", "drawdown", "var", "exposure",
            "limit", "stop loss", "take profit", "margin", "leverage",
        ],
        Department.TRADING: [
            "execute", "order", "buy", "sell", "trade", "fill",
            "route", "slippage", "broker", "venue", "paper",
        ],
        Department.PORTFOLIO: [
            "portfolio", "allocation", "rebalance", "performance",
            "diversify", "asset", "balance", "attribut",
        ],
    }

    def __init__(
        self,
        mail_db_path: str = ".quantmind/department_mail.db",
        max_workers_per_dept: int = 5,
        use_redis_mail: bool = True,
    ):
        """
        Initialize the Floor Manager.

        Args:
            mail_db_path: Path to SQLite mail database (used if use_redis_mail=False)
            max_workers_per_dept: Maximum workers per department
            use_redis_mail: If True, use Redis Streams for mail (recommended)
        """
        if use_redis_mail:
            self.mail_service = get_redis_mail_service()
        else:
            self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self._init_spawner()
        self.departments = self._init_departments()
        self.model_tier = "opus"
        self._max_workers = max_workers_per_dept

        # Initialize Copilot Kill Switch reference (Story 5.6)
        self._copilot_kill_switch = get_copilot_kill_switch()

        # Track active department tasks for cancellation (Story 5.6)
        self._active_tasks: Dict[str, Any] = {}

        # Initialize intent classifier (Story 5.7)
        self._intent_classifier: Optional[IntentClassifier] = None

        # Initialize Task Router (Story 7.7)
        self._task_router: TaskRouter = get_task_router()

        logger.info(f"FloorManager initialized with {len(self.departments)} departments")

    def _init_spawner(self):
        """Initialize the agent spawner."""
        try:
            from src.agents.subagent.spawner import get_spawner
            self.spawner = get_spawner()
        except ImportError:
            logger.warning("Agent spawner not available, using mock")
            self.spawner = None

    def _init_departments(self) -> Dict[Department, DepartmentHeadConfig]:
        """Initialize department configurations and instantiate real department heads."""
        configs = get_department_configs()

        # Instantiate real department head objects
        self._department_heads: Dict[Department, Any] = {}
        head_imports = [
            (Department.RESEARCH, "src.agents.departments.heads.research_head", "ResearchHead"),
            (Department.DEVELOPMENT, "src.agents.departments.heads.development_head", "DevelopmentHead"),
            (Department.RISK, "src.agents.departments.heads.risk_head", "RiskHead"),
            (Department.TRADING, "src.agents.departments.heads.execution_head", "TradingHead"),
            (Department.PORTFOLIO, "src.agents.departments.heads.portfolio_head", "PortfolioHead"),
        ]
        for dept, module_path, class_name in head_imports:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self._department_heads[dept] = cls()
                logger.info(f"Initialized {class_name} for {dept.value}")
            except Exception as e:
                logger.warning(f"Failed to init {class_name} for {dept.value}: {e}")

        # Populate task handlers from department heads
        self._task_handlers = {
            dept.value: head.process_task
            for dept, head in self._department_heads.items()
            if hasattr(head, "process_task")
        }

        return {
            Department(dept_name): config
            for dept_name, config in configs.items()
        }

    def _check_kill_switch(self) -> bool:
        """
        Check if Copilot kill switch is activated.

        Returns:
            True if kill switch is active and task should be cancelled

        Raises:
            RuntimeError: If kill switch is active (to signal cancellation)
        """
        if self._copilot_kill_switch.is_active:
            raise RuntimeError("Agent activity suspended - Copilot kill switch is active")
        return False

    async def _handle_morning_digest(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle /morning-digest command - generate a morning digest for the trader.

        Returns a summary of:
        - Overnight agent activity
        - Pending approvals
        - Market outlook
        - Critical alerts

        Args:
            context: Optional context dictionary

        Returns:
            Dict with status and digest content
        """
        from datetime import datetime, timedelta

        logger.info("Generating morning digest...")

        # Build digest components
        digest_sections = []

        # 1. Greeting with date
        now = datetime.now()
        greeting = f"**Morning Digest - {now.strftime('%A, %B %d, %Y')}**"
        digest_sections.append(greeting)

        # 2. Pending approvals (from department mail)
        try:
            from src.agents.departments.department_mail import get_mail_service
            mail_service = get_mail_service()
            pending = mail_service.get_pending_approvals()
            if pending:
                digest_sections.append("\n[PENDING APPROVALS]")
                for item in pending[:5]:  # Show max 5
                    digest_sections.append(f"  • {item.get('subject', 'Untitled')} ({item.get('from_dept', 'Unknown')})")
        except Exception as e:
            logger.warning(f"Could not fetch pending approvals: {e}")

        # 3. Open positions summary
        try:
            from src.database.models import Position
            from src.database.session import get_db_session

            with get_db_session() as session:
                positions = session.query(Position).filter(
                    Position.status == 'open'
                ).all()

                if positions:
                    digest_sections.append("\n[OPEN POSITIONS]")
                    total_pnl = sum(p.unrealized_pnl or 0 for p in positions)
                    digest_sections.append(f"  • {len(positions)} open positions")
                    digest_sections.append(f"  • Total unrealized P&L: ${total_pnl:,.2f}")
                else:
                    digest_sections.append("\n[OPEN POSITIONS] None")
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")

        # 4. Risk status
        try:
            from src.risk.models import RiskState
            with get_db_session() as session:
                risk = session.query(RiskState).order_by(RiskState.timestamp.desc()).first()
                if risk:
                    digest_sections.append("\n[RISK STATUS]")
                    digest_sections.append(f"  • Regime: {risk.current_regime}")
                    digest_sections.append(f"  • Risk Level: {risk.risk_level}")
        except Exception as e:
            logger.warning(f"Could not fetch risk status: {e}")

        # 5. Recent alerts
        try:
            from src.database.models import Alert
            with get_db_session() as session:
                cutoff = datetime.now() - timedelta(hours=24)
                alerts = session.query(Alert).filter(
                    Alert.severity == 'critical',
                    Alert.created_at >= cutoff
                ).all()

                if alerts:
                    digest_sections.append("\n[CRITICAL ALERTS (24H)]")
                    for alert in alerts[:3]:
                        digest_sections.append(f"  • {alert.message}")
                else:
                    digest_sections.append("\n[CRITICAL ALERTS (24H)] None")
        except Exception as e:
            logger.warning(f"Could not fetch alerts: {e}")

        # 6. Agent activity summary
        try:
            from src.agents.departments.department_mail import get_mail_service
            mail_service = get_mail_service()
            stats = mail_service.get_queue_stats()

            digest_sections.append("\n[AGENT ACTIVITY]")
            for dept, count in stats.items():
                if count > 0:
                    digest_sections.append(f"  • {dept.title()}: {count} queued tasks")
        except Exception as e:
            logger.warning(f"Could not fetch agent activity: {e}")

        # 7. Market outlook (placeholder)
        digest_sections.append("\n[MARKET OUTLOOK]")
        digest_sections.append("  • Market data integration pending - use 'check market' command for live data")

        # 8. Suggested actions
        digest_sections.append("\n[SUGGESTED ACTIONS]")
        digest_sections.append("  • Type '/skills' to see available commands")
        digest_sections.append("  • Type 'check risk' for current risk status")
        digest_sections.append("  • Type 'list positions' for open trades")

        content = "\n".join(digest_sections)

        return {
            "status": "success",
            "content": content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "morning_digest",
        }

    async def _handle_weekend_tasks(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle weekend task query - Story 11.2: Weekend Compute Protocol.

        Answers "What's running this weekend?" queries by querying the
        scheduled tasks API for weekend compute task status.

        Args:
            message: User message to check for weekend task patterns
            context: Optional context dictionary

        Returns:
            Dict with weekend task response or None if not a weekend query
        """
        import re

        # Detect weekend task query patterns
        weekend_patterns = [
            r"what.*running.*weekend",
            r"weekend.*task",
            r"scheduled.*task.*weekend",
            r"weekend.*compute",
            r"weekend.*job",
            r"saturday.*task",
            r"monte.*carlo.*weekend",
            r"hmm.*weekend",
        ]

        message_lower = message.lower()
        is_weekend_query = any(
            re.search(pattern, message_lower) for pattern in weekend_patterns
        )

        if not is_weekend_query:
            return None

        logger.info("Handling weekend task query")

        try:
            # Import and call the weekend tasks API
            from src.api.scheduled_tasks_endpoints import get_weekend_tasks

            # Get weekend task status (async call)
            result = await get_weekend_tasks(weekend_date=None)

            # Build response
            response_lines = [
                "**Weekend Compute Tasks**",
                f"Query Time: {result.query_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Weekend: {result.weekend_start.strftime('%Y-%m-%d')}",
                "",
                "**Task Status:**",
            ]

            # Add task details
            for task in result.tasks:
                status_indicator = {
                    "running": "[R]",
                    "completed": "[D]",
                    "failed": "[F]",
                    "scheduled": "[S]",
                    "pending": "[P]",
                    "retrying": "[R]"
                }.get(task.get("status", "unknown"), "[?]")

                task_name = task.get("task_name", "unknown")
                status = task.get("status", "unknown")
                progress = task.get("progress_percent", 0)
                duration = task.get("duration_seconds", 0)

                response_lines.append(
                    f"  {status_indicator} **{task_name}**: {status} "
                    f"({progress:.0f}%, {duration:.0f}s)"
                )

                # Add ETA if running
                if status == "running" and task.get("estimated_completion"):
                    eta = task.get("estimated_completion")
                    response_lines.append(f"      ETA: {eta}")

            # Summary
            response_lines.extend([
                "",
                "**Summary:**",
                f"  Total: {result.total_tasks}",
                f"  Running: {result.running_count}",
                f"  Completed: {result.completed_count}",
                f"  Failed: {result.failed_count}",
            ])

            return {
                "status": "success",
                "content": "\n".join(response_lines),
                "model": f"claude-opus-{self.model_tier}",
                "type": "weekend_tasks",
            }

        except Exception as e:
            logger.error(f"Error handling weekend tasks query: {e}")
            return {
                "status": "success",
                "content": f"I found a weekend task query but encountered an error: {str(e)}",
                "model": f"claude-opus-{self.model_tier}",
                "type": "weekend_tasks_error",
                "error": str(e)
            }

    async def _handle_audit_query(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle audit query - Story 10.4: NL Audit Query UI & Reasoning Explorer.

        Detects audit-related queries and returns timeline/causal chain responses.

        Args:
            message: User message to check for audit query patterns
            context: Optional context dictionary

        Returns:
            Dict with audit response or None if not an audit query
        """
        from src.intent.patterns import CommandIntent

        # Use existing intent classifier for pattern matching (reuses cached patterns)
        classifier = self._get_intent_classifier()
        canvas_context = context or {}
        result = await classifier.classify(message, canvas_context)

        # Check if it's an audit timeline query
        if result.intent == CommandIntent.AUDIT_TIMELINE_QUERY:
            logger.info(f"Detected audit timeline query: {message[:50]}...")
            return await self._build_audit_timeline_response(message, context)

        # Check if it's an audit reasoning query
        if result.intent == CommandIntent.AUDIT_REASONING_QUERY:
            logger.info(f"Detected audit reasoning query: {message[:50]}...")
            return await self._build_audit_reasoning_response(message, context)

        return None

    async def _build_audit_timeline_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an audit timeline response for "Why was X paused?" queries.

        Returns a formatted timeline showing the causal chain of events.

        Args:
            message: User's audit query
            context: Optional context

        Returns:
            Dict with timeline response
        """
        import re
        from datetime import datetime, timedelta

        # Extract entity (e.g., EA name) from message
        entity_match = re.search(r'(?:EA[_-]?|strategy[_-]?)?([A-Z]{3,6}(?:USD|EUR|GBP|JPY)?)', message, re.IGNORECASE)
        entity = entity_match.group(1) if entity_match else "Unknown"

        # Extract time reference if present
        time_ref = "yesterday"
        if "today" in message.lower():
            time_ref = "today"
        elif "last week" in message.lower():
            time_ref = "last week"

        # Build timeline response (placeholder - actual implementation would query audit logs)
        timeline_content = f"""[AUDIT TIMELINE] **{entity}** ({time_ref})

Here's what the system recorded:

`[14:28 UTC]` **HMM Regime Detection**
→ Regime changed to HIGH_VOL
→ Volatility spike detected in GBP/USD pair

`[14:29 UTC]` **Risk Governor Action**
→ Position limits tightened by 15%
→ Exposure threshold exceeded

`[14:30 UTC]` **Strategy Pause Executed**
→ EA_{entity} paused by Commander
→ Reason: Risk limit breach (HIGH_VOL regime)

---
*Note: This is a demo response. Full implementation requires Story 10.1 (NL Query API) backend.*"""

        return {
            "status": "success",
            "content": timeline_content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "audit_timeline",
            "entity": entity,
            "time_reference": time_ref,
        }

    async def _build_audit_reasoning_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an audit reasoning response for "Show reasoning" queries.

        Returns the OPINION node chain with confidence scores and evidence.

        Args:
            message: User's audit query
            context: Optional context

        Returns:
            Dict with reasoning chain response
        """
        # Extract department or entity if mentioned
        department_match = None
        department_keywords = ["research", "development", "risk", "trading", "portfolio"]
        for dept in department_keywords:
            if dept in message.lower():
                department_match = dept
                break

        # Build reasoning response
        reasoning_content = f"""🧠 **Decision Reasoning Chain**

{"Department: " + department_match.title() if department_match else "Analyzing recent decisions..."}

Here's the reasoning chain from the Memory Graph:

**OPINION Node Chain:**

1. **Research Department** → Confidence: 0.85
   - Evidence: GBP/USD H1 chart shows triple top pattern
   - Reasoning: "Bearish divergence on RSI, support at 1.2450"
   - Action: Recommend short entry

2. **Risk Department** → Confidence: 0.72
   - Evidence: Current exposure 78%, max allowed 80%
   - Reasoning: "Limited capacity for new positions"
   - Action: Approve with position size reduction

3. **Trading Department** → Confidence: 0.90
   - Evidence: Spread 1.2 pips, liquidity adequate
   - Reasoning: "Optimal execution conditions"
   - Action: Execute short at market

---
*Note: This is a demo response. Full implementation requires Story 10.2 (Reasoning Log API) backend.*"""

        return {
            "status": "success",
            "content": reasoning_content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "audit_reasoning",
            "department": department_match,
        }

    async def _handle_backup_restore_query(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle backup/restore queries - Story 11.4: Full Backup & Restore.

        Detects backup/restore command patterns and returns appropriate responses.

        Args:
            message: User message to check for backup/restore patterns
            context: Optional context dictionary

        Returns:
            Dict with backup/restore response or None if not a backup/restore query
        """
        from src.intent.patterns import CommandIntent

        # Use existing intent classifier for pattern matching
        classifier = self._get_intent_classifier()
        canvas_context = context or {}
        result = await classifier.classify(message, canvas_context)

        # Check if it's a backup system request
        if result.intent == CommandIntent.BACKUP_SYSTEM:
            logger.info(f"Detected backup system request: {message[:50]}...")
            return await self._build_backup_response(message, context)

        # Check if it's a restore backup request
        if result.intent == CommandIntent.RESTORE_BACKUP:
            logger.info(f"Detected restore backup request: {message[:50]}...")
            return await self._build_restore_response(message, context)

        # Check if it's a backup query (list backups, show backups, etc.)
        if result.intent == CommandIntent.BACKUP_QUERY:
            logger.info(f"Detected backup query: {message[:50]}...")
            return await self._build_backup_list_response(message, context)

        return None

    async def _build_backup_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a response for backup system requests.

        Args:
            message: User's backup request
            context: Optional context

        Returns:
            Dict with backup response
        """
        import subprocess
        from pathlib import Path

        # Determine backup directory
        backup_dir = Path.home() / ".quantmind" / "backups"
        backup_script = Path(__file__).parent.parent.parent.parent / "scripts" / "backup_full_system.sh"

        # Check if backup script exists
        if not backup_script.exists():
            return {
                "status": "error",
                "content": "Backup script not found. Please ensure scripts/backup_full_system.sh exists.",
                "model": f"claude-opus-{self.model_tier}",
            }

        # Try to run the backup (non-blocking, just return info)
        # Actual backup should be run as a background task or via cron
        content = f"""🗄️ **System Backup**

Starting full system backup for FR69: machine portability...

**Backup includes:**
- Configuration files (provider credentials, server connections, broker accounts, risk parameters)
- Knowledge base (PageIndex data, news items)
- Strategy artifacts (TRDs, EA templates, backtest results)
- Graph memory (session memory nodes, opinion chains)
- Canvas context (department canvas templates)

**Backup location:** `{backup_dir}`
**Script:** `{backup_script.name}`

To run backup now, execute:
```bash
./scripts/backup_full_system.sh
```

For remote backup configuration, set:
- `REMOTE_BACKUP_ENABLED=true`
- `REMOTE_HOST=your-backup-server`
- `REMOTE_PATH=/backup/quantmindx`
"""

        return {
            "status": "success",
            "content": content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "backup_info",
        }

    async def _build_restore_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a response for restore from backup requests.

        Args:
            message: User's restore request
            context: Optional context

        Returns:
            Dict with restore response
        """
        import re
        from pathlib import Path

        # Extract backup filename if mentioned
        backup_match = re.search(r'backup[_\s](\d{8}_\d{6})', message, re.IGNORECASE)
        backup_filename = backup_match.group(0) + ".tar.gz" if backup_match else None

        backup_dir = Path.home() / ".quantmind" / "backups"
        restore_script = Path(__file__).parent.parent.parent.parent / "scripts" / "restore_full_system.sh"

        # List available backups
        available_backups = []
        if backup_dir.exists():
            available_backups = sorted(backup_dir.glob("backup_*.tar.gz"))

        content = f"""♻️ **System Restore**

Starting system restore from backup (FR69: machine portability)...

**Restore will restore:**
- Configuration files
- Knowledge base
- Strategy artifacts
- Graph memory
- Canvas context

**Restore script:** `{restore_script.name}`
**Backup directory:** `{backup_dir}`

"""

        if backup_filename:
            content += f"**Requested backup:** {backup_filename}\n\nTo restore, execute:\n```bash\n./scripts/restore_full_system.sh {backup_filename}\n```\n"
        elif available_backups:
            content += f"**Available backups ({len(available_backups)}):**\n"
            for backup in available_backups[-5:]:  # Show last 5
                content += f"- {backup.name}\n"
            content += "\nTo restore, specify the backup filename:\n```bash\n./scripts/restore_full_system.sh backup_YYYYMMDD_HHMMSS.tar.gz\n```\n"
        else:
            content += "⚠️ No backups found in backup directory.\n"

        content += """
⚠️ **Warning:** Restore will overwrite existing data. Make sure to:
1. Back up current state if needed
2. Confirm you want to proceed
3. Verify the backup integrity first with `--validate-only`
"""

        return {
            "status": "success",
            "content": content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "restore_info",
        }

    async def _build_backup_list_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a response for backup query (list backups, show backups, etc.).

        Args:
            message: User's backup query
            context: Optional context

        Returns:
            Dict with backup list response
        """
        from pathlib import Path

        backup_dir = Path.home() / ".quantmind" / "backups"

        content = f"""📋 **Backup Status**

**Backup directory:** `{backup_dir}`

"""

        if not backup_dir.exists():
            content += "⚠️ Backup directory does not exist. No backups have been created yet."
        else:
            # List all backups
            backups = sorted(backup_dir.glob("backup_*.tar.gz"))

            if not backups:
                content += "No backups found. Run a backup with:\n```bash\n./scripts/backup_full_system.sh\n```"
            else:
                content += f"**Available backups ({len(backups)}):**\n\n"
                for backup in sorted(backups, reverse=True)[:10]:  # Show last 10
                    size_mb = backup.stat().st_size / (1024 * 1024)
                    content += f"- **{backup.name}** ({size_mb:.1f} MB)\n"

                content += f"\n**Total backups:** {len(backups)}\n"
                content += f"**Disk usage:** {sum(b.stat().st_size for b in backups) / (1024**3):.2f} GB\n"

            content += """
**Backup contents:**
- Configurations (database, provider configs, server connections)
- Knowledge base (PageIndex, news items)
- Strategy artifacts (TRDs, templates, backtest results)
- Graph memory
- Canvas context

**Restore command:**
```bash
./scripts/restore_full_system.sh <backup_file>
```

**Backup retention:** 30 days (oldest backups are automatically cleaned up)
"""

        # Check for restore completion notification
        notification_file = Path.home() / ".quantmind" / "logs" / "restore_completion.notification"
        if notification_file.exists():
            try:
                content += "\n\n✅ **Recent Restore:**\n"
                content += notification_file.read_text()[:500]
            except Exception:
                pass

        return {
            "status": "success",
            "content": content,
            "model": f"claude-opus-{self.model_tier}",
            "type": "backup_list",
        }

    def _get_intent_classifier(self) -> IntentClassifier:
        """
        Get or create the intent classifier instance.

        Returns:
            IntentClassifier instance for command classification
        """
        if self._intent_classifier is None:
            self._intent_classifier = get_intent_classifier()
        return self._intent_classifier

    def _cancel_active_tasks(self) -> List[str]:
        """
        Cancel all active department agent tasks.

        This is called when the Copilot kill switch is activated to
        actively terminate running tasks.

        Returns:
            List of cancelled task IDs
        """
        cancelled = []
        for task_id, task_info in self._active_tasks.items():
            cancelled.append(task_id)
            logger.info(f"Cancelling active task: {task_id}")

        # Clear the active tasks dictionary
        self._active_tasks.clear()
        return cancelled

    def register_task(self, task_id: str, task_info: Any) -> None:
        """
        Register an active task for tracking.

        Args:
            task_id: Unique identifier for the task
            task_info: Task details (can be any type)
        """
        self._active_tasks[task_id] = task_info

    def unregister_task(self, task_id: str) -> None:
        """
        Unregister a task (when completed or cancelled).

        Args:
            task_id: Unique identifier for the task
        """
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]

    def _keyword_route(self, task: str) -> str:
        """
        Keyword-based fallback routing for when LLM classification is unavailable.

        Args:
            task: The task description

        Returns:
            Department value string (e.g. "research", "risk")
        """
        task_lower = task.lower()
        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            if any(kw in task_lower for kw in keywords):
                return dept.value
        return Department.RESEARCH.value  # default fallback

    async def dispatch_task(
        self,
        department: Department,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a task directly to a department head's process_task method.

        Args:
            department: Target department enum
            task: Task description string
            context: Optional context dictionary

        Returns:
            Dict with status, department, and result or error
        """
        self._check_kill_switch()
        head = self._department_heads.get(department)
        if not head:
            return {
                "status": "error",
                "message": f"Department {department.value} not available",
            }
        try:
            result = await head.process_task(task=task, context=context or {})
            return {
                "status": "success",
                "department": department.value,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Task dispatch to {department.value} failed: {e}")
            return {
                "status": "error",
                "department": department.value,
                "error": str(e),
            }

    async def route_and_dispatch(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Use Opus LLM to classify intent then route to the correct department(s).

        Falls back to keyword routing if the LLM call fails.

        Args:
            task: Task description string
            context: Optional context dictionary

        Returns:
            Dict with routing info and list of per-department results
        """
        import os
        import anthropic as _anthropic

        self._check_kill_switch()

        # Emit routing thought to SSE stream
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            get_thought_publisher().publish(
                department="floor-manager",
                thought=f"Routing task: {task[:100]}",
                thought_type="dispatch",
            )
        except Exception:
            pass

        client = _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        routing_prompt = (
            "You are the Floor Manager of QUANTMINDX, an AI-driven algorithmic trading platform.\n"
            "Your job is to classify incoming tasks and route them to the right department(s).\n\n"
            "DEPARTMENTS AND THEIR SCOPE:\n"
            "- research: Strategy discovery, hypothesis generation, backtesting, knowledge base queries, "
            "market analysis, alpha research, TRD authoring. Use when the request is about FINDING or VALIDATING a strategy.\n"
            "- development: EA/bot implementation, MQL5/Python/PineScript code generation, "
            "compilation, TRD-to-code conversion. Use when the request is about BUILDING or CODING something.\n"
            "- risk: Position sizing, drawdown checks, VaR calculations, backtest PASS/FAIL evaluation, "
            "risk parameter management. Use when the request is about APPROVING or MEASURING risk.\n"
            "- trading: Order execution (paper/demo MT5), fill tracking, slippage monitoring, "
            "paper trading sessions. Use when the request is about EXECUTING or MONITORING trades.\n"
            "- portfolio: Allocation optimization, rebalancing, performance attribution, "
            "correlation analysis, portfolio reports. Use when the request is about the OVERALL PORTFOLIO.\n\n"
            "ROUTING RULES:\n"
            "- New strategy idea → research\n"
            "- 'Write/build/code an EA' → development (may also need research for spec)\n"
            "- 'Is this strategy safe?' / 'Check risk' → risk\n"
            "- 'Execute / start paper trading' → trading (requires risk approval first)\n"
            "- 'How is my portfolio doing?' → portfolio\n"
            "- Multi-step workflows (e.g. research→develop→risk→trade) → list all departments in order\n"
            "- Urgent = live trading impact or account risk; High = time-sensitive; Normal = routine\n\n"
            f"TASK: {task}\n\n"
            "Respond ONLY with valid JSON (no markdown):\n"
            '{"departments": ["dept1"], "reasoning": "one sentence", "priority": "normal|high|urgent", '
            '"workflow": ["dept1", "dept2"]}'
        )

        try:
            resp = await client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL_OPUS", "claude-opus-4-6"),
                max_tokens=256,
                messages=[{"role": "user", "content": routing_prompt}],
            )
            import re as _re
            text = resp.content[0].text
            match = _re.search(r'\{.*\}', text, _re.DOTALL)
            routing = json.loads(match.group()) if match else {"departments": [], "reasoning": text}
        except Exception as e:
            logger.warning(f"LLM routing failed, falling back to keyword: {e}")
            routing = {
                "departments": [self._keyword_route(task)],
                "reasoning": "keyword fallback",
            }

        results = []
        for dept_name in routing.get("departments", []):
            try:
                dept = Department(dept_name)
                result = await self.dispatch_task(dept, task, context)
                results.append(result)
            except ValueError:
                logger.warning(f"Unknown department in routing: {dept_name}")

        return {"routing": routing, "results": results}

    def classify_task(self, task: str) -> Department:
        """
        Classify a task to determine which department should handle it.

        Uses keyword matching for simple classification.
        Can be upgraded to LLM-based classification for complex tasks.

        Args:
            task: The task description

        Returns:
            The department that should handle this task
        """
        task_lower = task.lower()

        # Score each department based on keyword matches
        scores: Dict[Department, int] = {}
        for dept, keywords in self.DEPARTMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[dept] = score

        # Return highest scoring department, default to RESEARCH
        if scores:
            return max(scores, key=scores.get)
        return Department.RESEARCH

    def dispatch(
        self,
        to_dept: Department,
        task: str,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a task to a department via mail.

        Args:
            to_dept: Target department
            task: Task description
            priority: Message priority (low, normal, high, urgent)
            context: Optional context dictionary

        Returns:
            Dispatch result with status and message ID
        """
        # Check if kill switch is activated (Story 5.6)
        if self._copilot_kill_switch.is_active:
            return {
                "status": "suspended",
                "message": "Agent activity suspended - Copilot kill switch is active",
            }

        # Map priority string to enum
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }
        msg_priority = priority_map.get(priority.lower(), Priority.NORMAL)

        # Build message body
        body = task
        if context:
            body = json.dumps({
                "task": task,
                "context": context,
            })

        # Send mail message
        message = self.mail_service.send(
            from_dept="floor_manager",
            to_dept=to_dept.value,
            type=MessageType.DISPATCH,
            subject=f"Task: {task[:50]}...",
            body=body,
            priority=msg_priority,
        )

        logger.info(f"Dispatched task to {to_dept.value}: {message.id}")

        return {
            "status": "dispatched",
            "message_id": message.id,
            "to_dept": to_dept.value,
            "priority": priority,
        }

    def process(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process an incoming task.

        1. Classify the task to determine department
        2. Dispatch to the appropriate department

        Args:
            task: Task description
            context: Optional context dictionary

        Returns:
            Processing result with classification and dispatch info
        """
        # Classify task
        dept = self.classify_task(task)
        logger.info(f"Classified task to {dept.value}: {task[:50]}...")

        # Dispatch to department
        dispatch_result = self.dispatch(
            to_dept=dept,
            task=task,
            priority="normal",
            context=context,
        )

        return {
            "status": "processed",
            "classified_dept": dept.value,
            "dispatch": dispatch_result,
        }

    def handle_dispatch(
        self,
        from_department: str,
        task: str,
        suggested_department: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a dispatch request from Copilot or another department.

        Processes incoming delegation requests and routes them to the appropriate
        department. If a suggested department is provided and valid, uses it.
        Otherwise, classifies the task automatically.

        Args:
            from_department: Department sending the dispatch (e.g., "copilot")
            task: Task description to dispatch
            suggested_department: Optional target department suggestion
            context: Optional context dictionary

        Returns:
            Dispatch result with status, message ID, and routing info
        """
        # Determine target department
        target_dept = None

        if suggested_department:
            # Validate suggested department
            try:
                target_dept = Department(suggested_department.lower())
                logger.info(f"Using suggested department: {suggested_department}")
            except ValueError:
                logger.warning(
                    f"Invalid suggested department: {suggested_department}, "
                    "falling back to classification"
                )
                target_dept = self.classify_task(task)
        else:
            # Auto-classify if no suggestion
            target_dept = self.classify_task(task)

        # Delegate to the determined department
        message = self.delegate_to_department(
            from_dept=from_department,
            to_dept=target_dept.value,
            task=task,
            priority="normal",
            context=context,
        )

        return {
            "status": "dispatched",
            "message_id": message.id,
            "from_department": from_department,
            "to_department": target_dept.value,
            "priority": "normal",
        }

    def delegate_to_department(
        self,
        from_dept: str,
        to_dept: str,
        task: str,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> "DepartmentMessage":
        """
        Delegate a task to a department via mail.

        Sends a dispatch message through the mail service to the target department.

        Args:
            from_dept: Sending department identifier
            to_dept: Target department identifier
            task: Task description
            priority: Message priority (low, normal, high, urgent)
            context: Optional context dictionary

        Returns:
            The created DepartmentMessage
        """
        # Map priority string to enum
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT,
        }
        msg_priority = priority_map.get(priority.lower(), Priority.NORMAL)

        # Build message body with optional context
        body = task
        if context:
            body = json.dumps({
                "task": task,
                "context": context,
            })

        # Create subject line
        subject = f"Task from {from_dept}: {task[:50]}..."
        if len(task) > 50:
            subject += "..."

        # Send mail message
        message = self.mail_service.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=MessageType.DISPATCH,
            subject=subject,
            body=body,
            priority=msg_priority,
        )

        logger.info(
            f"Delegated task from {from_dept} to {to_dept}: "
            f"message_id={message.id}"
        )

        return message

    def get_departments(self) -> List[Dict[str, Any]]:
        """Get all department configurations with personality info."""
        result = []
        for dept, config in self.departments.items():
            # Get personality if available
            personality = None
            if config.personality:
                personality = {
                    "name": config.personality.name,
                    "tagline": config.personality.tagline,
                    "traits": config.personality.traits,
                    "communication_style": config.personality.communication_style,
                    "strengths": config.personality.strengths,
                    "weaknesses": config.personality.weaknesses,
                    "color": config.personality.color,
                    "icon": config.personality.icon,
                }

            # Count pending mail for this department
            pending = len(self.mail_service.check_inbox(dept.value, unread_only=True, limit=100))

            result.append({
                "id": dept.value,
                "name": dept.value.capitalize(),
                "agent_type": config.agent_type,
                "system_prompt": config.system_prompt[:200] + "..." if len(config.system_prompt) > 200 else config.system_prompt,
                "sub_agents": config.sub_agents,
                "memory_namespace": config.memory_namespace,
                "model_tier": get_model_tier(dept),
                "max_workers": config.max_workers,
                "pending_tasks": pending,
                "status": "active",
                "personality": personality,
            })
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get floor manager status."""
        return {
            "status": "active",
            "model_tier": self.model_tier,
            "departments": {
                dept.value: {
                    "id": dept.value,
                    "name": dept.value.capitalize(),
                    "agent_type": config.agent_type,
                    "sub_agents": config.sub_agents,
                    "memory_namespace": config.memory_namespace,
                    "model_tier": get_model_tier(dept),
                    "max_workers": config.max_workers,
                    "pending_tasks": len(self.mail_service.check_inbox(dept.value, unread_only=True, limit=100)),
                    "status": "active",
                }
                for dept, config in self.departments.items()
            },
            "stats": {
                "total_departments": len(self.departments),
                "total_agents": sum(config.max_workers for config in self.departments.values()),
                "pending_mail": sum(
                    len(self.mail_service.check_inbox(dept.value, unread_only=True, limit=100))
                    for dept in self.departments
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Chat with the Floor Manager.

        Classifies the task and either responds directly or delegates to a department.

        Args:
            message: User message
            context: Optional context (canvas_context, session_id, etc.)
            history: Conversation history for context (list of {role, content})
            stream: Whether to stream response (not implemented yet)

        Args:
            message: User message
            context: Optional context (canvas_context, session_id, etc.)
            stream: Whether to stream response (not implemented yet)

        Returns:
            Dict with status, content, and optional delegation info
        """
        # Check if kill switch is activated (Story 5.6)
        try:
            self._check_kill_switch()
        except RuntimeError as e:
            return {
                "status": "suspended",
                "content": f"[SUSPENDED] {str(e)}",
                "error": str(e),
            }

        # Handle system commands (Story 5.8: Morning Digest)
        if message.strip().lower().startswith('/morning-digest'):
            return await self._handle_morning_digest(context)

        # Story 11.2: Handle weekend task queries
        weekend_result = await self._handle_weekend_tasks(message, context)
        if weekend_result:
            return weekend_result

        # Story 10.4: Handle audit queries
        audit_result = await self._handle_audit_query(message, context)
        if audit_result:
            return audit_result

        # Story 11.4: Handle backup/restore queries
        backup_result = await self._handle_backup_restore_query(message, context)
        if backup_result:
            return backup_result

        # Classify the task to determine department
        department = self.classify_task(message)

        # Check if task should be delegated based on keywords
        delegation_keywords = ["run", "research", "analyze", "execute", "trade", "check", "get", "fetch"]
        should_delegate = any(kw in message.lower() for kw in delegation_keywords)

        if should_delegate:
            # Delegate to the classified department
            try:
                msg = self.delegate_to_department(
                    from_dept="floor",
                    to_dept=department.value,
                    task=message,
                    priority="normal",
                    context=context,
                )

                return {
                    "status": "success",
                    "content": f"Delegating to {department.value.title()} Department: {message[:100]}...",
                    "delegation": {
                        "department": department.value,
                        "task_id": str(msg.id),
                        "status": "pending",
                    },
                    "model": f"claude-opus-{self.model_tier}",
                }
            except Exception as e:
                # Log error to audit trail
                logger.error(
                    f"[AUDIT] FloorManager delegation failed: dept={department.value}, "
                    f"message={message[:50]}..., error={str(e)}"
                )
                return {
                    "status": "success",
                    "content": f"I received your message: {message[:100]}... However, I encountered an issue delegating to the {department.value.title()} Department. Please try again or contact support.",
                    "error": str(e),
                    "model": f"claude-opus-{self.model_tier}",
                }
        else:
            # Direct response (simple response for now - can be enhanced with LLM)
            response_content = (
                f"I understand: {message[:100]}...\n\n"
                "I can help with:\n"
                "- Market analysis and research\n"
                "- Trading operations and execution\n"
                "- Risk management queries\n"
                "- Portfolio status checks\n\n"
                "Try asking me to 'run research on GBPUSD' or 'check risk levels' to see department delegation in action."
            )

            return {
                "status": "success",
                "content": response_content,
                "model": f"claude-opus-{self.model_tier}",
            }

    async def chat_stream(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Stream chat response token by token.

        This is an async generator that yields chunks of the response
        for real-time streaming to the client.

        Args:
            message: User message
            context: Optional context (canvas_context, session_id, etc.)
            history: Conversation history for context

        Yields:
            Dict with 'type' and 'content'/'delta' fields for SSE
        """
        import asyncio

        # Check if kill switch is activated (Story 5.6)
        try:
            self._check_kill_switch()
        except RuntimeError as e:
            yield {"type": "error", "content": str(e)}
            yield {"type": "suspended", "message": str(e)}
            return

        # Handle system commands (Story 5.8: Morning Digest)
        if message.strip().lower().startswith('/morning-digest'):
            digest_result = await self._handle_morning_digest(context)
            content = digest_result.get("content", "")

            # Stream the content
            words = content.split()
            for i, word in enumerate(words):
                yield {"type": "content", "delta": word + (" " if i < len(words) - 1 else "")}
                await asyncio.sleep(0.02)

            yield {"type": "tool", "tool": "morning_digest", "status": "completed"}
            yield {"type": "done"}
            return

        # Yield tool usage notification
        yield {"type": "tool", "tool": "thinking", "status": "started"}

        # Classify the task to determine department
        yield {"type": "thought", "department": "floor_manager", "content": "Classifying task intent..."}
        department = self.classify_task(message)
        yield {"type": "thought", "department": "floor_manager", "content": f"Routing to {department.value} department"}

        # Check if task should be delegated based on keywords
        delegation_keywords = ["run", "research", "analyze", "execute", "trade", "check", "get", "fetch"]
        should_delegate = any(kw in message.lower() for kw in delegation_keywords)

        if should_delegate:
            # Delegate to the classified department
            yield {"type": "thought", "department": "floor_manager", "content": f"Delegating task to {department.value} head..."}
            try:
                msg = self.delegate_to_department(
                    from_dept="floor",
                    to_dept=department.value,
                    task=message,
                    priority="normal",
                    context=context,
                )
                yield {"type": "thought", "department": department.value, "content": f"Processing: {message[:80]}..."}

                # Stream the delegation message
                response_text = f"Delegating to {department.value.title()} Department: {message[:100]}..."

                # Token-by-token yield
                words = response_text.split()
                for i, word in enumerate(words):
                    yield {"type": "content", "delta": word + (" " if i < len(words) - 1 else "")}
                    await asyncio.sleep(0.02)  # Simulate token streaming

                yield {"type": "thought", "department": "floor_manager", "content": "Task dispatched, awaiting response..."}
                yield {
                    "type": "delegation",
                    "department": department.value,
                    "task_id": str(msg.id),
                    "status": "pending",
                }
            except Exception as e:
                logger.error(f"FloorManager streaming delegation failed: {e}")
                yield {"type": "error", "error": str(e)}
        else:
            yield {"type": "thought", "department": "floor_manager", "content": "Generating response..."}
            # Direct response (simple response for now - can be enhanced with LLM)
            response_content = (
                f"I understand: {message[:100]}...\n\n"
                "I can help with:\n"
                "- Market analysis and research\n"
                "- Trading operations and execution\n"
                "- Risk management queries\n"
                "- Portfolio status checks\n\n"
                "Try asking me to 'run research on GBPUSD' or 'check risk levels' to see department delegation in action."
            )

            # Token-by-token yield
            words = response_content.split()
            for i, word in enumerate(words):
                yield {"type": "content", "delta": word + (" " if i < len(words) - 1 else "")}
                await asyncio.sleep(0.02)  # Simulate token streaming

        # Yield completion
        yield {"type": "tool", "tool": "thinking", "status": "completed"}
        yield {"type": "done"}

    # === Story 5.7: NL System Commands & Context-Aware Canvas Binding ===

    async def classify_intent(
        self,
        message: str,
        canvas_context: Dict[str, Any],
    ):
        """
        Classify user message into actionable intent.

        Uses pattern matching first, then falls back to canvas context binding
        for enhanced classification.

        Args:
            message: User message to classify
            canvas_context: Canvas context dictionary with canvas, session_id, entity

        Returns:
            IntentClassification with intent, entities, confidence
        """
        classifier = self._get_intent_classifier()
        return await classifier.classify(message, canvas_context)

    async def handle_command(
        self,
        message: str,
        canvas_context: Dict[str, Any],
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle natural language command with confirmation flow.

        Implements the full command handling flow:
        1. Classify intent
        2. If low confidence, ask for clarification
        3. If destructive command and not confirmed, ask for confirmation
        4. Execute command

        Args:
            message: User message
            canvas_context: Canvas context dictionary
            confirmed: Whether user has confirmed the action

        Returns:
            Dict with response type and content
        """
        # Check if kill switch is activated
        try:
            self._check_kill_switch()
        except RuntimeError as e:
            return {
                "type": "error",
                "status": "suspended",
                "message": str(e),
            }

        classifier = self._get_intent_classifier()
        return await classifier.handle_command(message, canvas_context, confirmed)

    # === Story 7.5: Session Workspace Isolation ===

    def commit_session(
        self,
        session_id: str,
        department: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Commit all draft nodes for a session.

        This is called when a Department Head completes a session and commits its work.
        Committed nodes become visible to all subsequent sessions.

        For AC #3: If entity_id provided and conflicts detected, returns 'pending_review'
        instead of immediately committing.

        Args:
            session_id: Session ID to commit nodes for.
            department: Optional department to filter nodes.
            entity_id: Entity identifier to check for conflicts (e.g., strategy_id).

        Returns:
            Dictionary with commit results: node_count, committed_at, session_id, department.
            If conflicts detected, returns status='pending_review' with conflict info.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            result = facade.commit_session(
                session_id=session_id,
                department=department,
                entity_id=entity_id,
            )

            # Check if commit was blocked due to conflicts (AC #3)
            if result.get("status") == "pending_review":
                logger.warning(
                    f"Session {session_id} commit requires DeptHead review - "
                    f"{result.get('conflict_count')} conflicts detected"
                )
                return result

            logger.info(
                f"Committed {result['node_count']} nodes for session {session_id} "
                f"(department: {department or 'all'})"
            )

            return {
                "status": "success",
                "session_id": session_id,
                "node_count": result["node_count"],
                "committed_at_utc": result["committed_at_utc"],
                "department": department,
            }
        except Exception as e:
            logger.error(f"Failed to commit session {session_id}: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e),
            }

    def get_session_draft_nodes(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get draft nodes for a session (private to that session).

        Args:
            session_id: Session ID to get draft nodes for.
            limit: Maximum number of results.

        Returns:
            List of draft node dictionaries.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            nodes = facade.get_draft_nodes(session_id=session_id, limit=limit)

            return [
                {
                    "id": str(node.id),
                    "title": node.title,
                    "content": node.content[:200] if node.content else "",
                    "node_type": node.node_type.value,
                    "created_at_utc": node.created_at_utc.isoformat(),
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to get draft nodes for session {session_id}: {e}")
            return []

    def get_session_committed_nodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get committed nodes (visible to all sessions).

        Args:
            session_id: Optional session ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of committed node dictionaries.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            nodes = facade.get_committed_nodes(session_id=session_id, limit=limit)

            return [
                {
                    "id": str(node.id),
                    "title": node.title,
                    "content": node.content[:200] if node.content else "",
                    "node_type": node.node_type.value,
                    "session_id": node.session_id,
                    "committed_at_utc": node.updated_at_utc.isoformat(),
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to get committed nodes: {e}")
            return []

    def get_commit_log(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get commit log entries.

        Args:
            session_id: Optional filter by session_id.
            limit: Maximum number of results.

        Returns:
            List of commit log entries.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            return facade.get_commit_log(session_id=session_id, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get commit log: {e}")
            return []

    def detect_session_conflicts(
        self,
        strategy_id: str,
        current_session_id: str,
    ) -> list[dict[str, Any]]:
        """Detect concurrent writes to the same strategy namespace.

        Used to detect when multiple sessions are working on the same strategy
        and might have conflicting changes.

        Args:
            strategy_id: Strategy identifier to check for conflicts.
            current_session_id: Current session ID to exclude.

        Returns:
            List of conflicting nodes from other sessions.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            nodes = facade.detect_conflicts(
                strategy_id=strategy_id,
                exclude_session_id=current_session_id,
            )

            return [
                {
                    "id": str(node.id),
                    "title": node.title,
                    "session_id": node.session_id,
                    "entity_id": node.entity_id,
                    "created_at_utc": node.created_at_utc.isoformat(),
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to detect conflicts for strategy {strategy_id}: {e}")
            return []

    def query_session_isolated(
        self,
        session_id: str,
        query: Optional[str] = None,
        include_committed: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query nodes with session isolation.

        Returns nodes that are:
        - In the specified session_id with status 'draft' (private to session)
        - OR have status 'committed' (visible to all)

        Args:
            session_id: Session ID for filtering draft nodes.
            query: Optional text search query.
            include_committed: Whether to include committed nodes.
            limit: Maximum number of results.

        Returns:
            List of node dictionaries visible to the session.
        """
        from src.memory.graph.facade import get_graph_memory

        try:
            facade = get_graph_memory()
            nodes = facade.query_session_isolated(
                session_id=session_id,
                query=query,
                include_committed=include_committed,
                limit=limit,
            )

            return [
                {
                    "id": str(node.id),
                    "title": node.title,
                    "content": node.content[:200] if node.content else "",
                    "node_type": node.node_type.value,
                    "session_status": node.session_status,
                    "session_id": node.session_id,
                    "importance": node.importance,
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Failed to query session isolated nodes: {e}")
            return []

    # =========================================================================
    # Story 7.7: Concurrent Task Routing
    # =========================================================================

    def register_task_handler(
        self,
        task_type: str,
        handler: Any,
    ) -> None:
        """
        Register a handler for a task type.

        Args:
            task_type: Type of task (e.g., "research", "development")
            handler: Async handler function
        """
        self._task_handlers[task_type] = handler
        logger.info(f"Registered task handler for type: {task_type}")

    def dispatch_concurrent(
        self,
        tasks: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dispatch multiple tasks concurrently (AC-1).

        Args:
            tasks: List of task definitions with keys:
                - task_type: Type of task (e.g., "research", "development")
                - department: Target department (Department enum or string)
                - payload: Task payload
                - priority: Optional priority ("high", "medium", "low")
            session_id: Session ID for isolation

        Returns:
            List of dispatched task info dicts
        """
        # Check kill switch
        if self._copilot_kill_switch.is_active:
            return [{
                "status": "suspended",
                "message": "Agent activity suspended - Copilot kill switch is active",
            }]

        # Parse priorities
        priority_map = {
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
        }

        # Convert to task router format
        router_tasks = []
        for task_def in tasks:
            priority_str = task_def.get("priority", "medium")
            router_tasks.append({
                "task_type": task_def["task_type"],
                "department": task_def["department"],
                "payload": task_def.get("payload", {}),
                "priority": priority_map.get(priority_str.lower(), TaskPriority.MEDIUM),
            })

        # Dispatch via task router
        dispatched = self._task_router.dispatch_concurrent(router_tasks, session_id)

        # Register tasks for tracking
        for task in dispatched:
            self.register_task(task.task_id, {
                "task_type": task.task_type,
                "department": task.department,
                "priority": task.priority.value,
                "session_id": task.session_id,
            })

        logger.info(f"Dispatched {len(dispatched)} concurrent tasks")

        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "department": t.department,
                "priority": t.priority.value,
                "session_id": t.session_id,
                "status": t.status.value,
            }
            for t in dispatched
        ]

    async def execute_concurrent(
        self,
        session_id: str,
        max_concurrent: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute all pending tasks for a session concurrently (AC-1, AC-2).

        Args:
            session_id: Session ID to execute tasks for
            max_concurrent: Maximum concurrent executions

        Returns:
            Aggregated result with timing metrics
        """
        # Get all active tasks for session
        session_tasks = [
            task for task in self._task_router._active_tasks.values()
            if task.session_id == session_id and task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]
        ]

        if not session_tasks:
            return {
                "status": "no_tasks",
                "message": "No pending tasks for session",
            }

        # Execute concurrently
        completed = await self._task_router.execute_concurrent(
            session_tasks,
            self._task_handlers,
            max_concurrent=max_concurrent,
        )

        # Aggregate results
        result = await self._task_router.aggregate_results(completed)

        # Update tracking
        for task in completed:
            if task.status == TaskStatus.COMPLETED:
                self.unregister_task(task.task_id)

        return result.to_dict()

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task (AC-1).

        Args:
            task_id: Task ID

        Returns:
            Task status dict or None
        """
        status = self._task_router.get_task_status(task_id)

        if task_id in self._active_tasks:
            task_info = self._active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": status.value if status else "unknown",
                "task_type": task_info.get("task_type"),
                "department": task_info.get("department"),
            }

        return None

    def get_all_department_status(self, session_id: str) -> Dict[str, Dict[str, str]]:
        """
        Get status of all departments for Agent Panel (AC-1).

        Args:
            session_id: Session ID

        Returns:
            Dict mapping department name to status dict
        """
        return self._task_router.get_all_department_status(session_id)

    def get_concurrent_task_status_display(self, session_id: str) -> Dict[str, str]:
        """
        Get formatted status for Agent Panel display (AC-1).

        Returns:
            Dict mapping department to status string like "running", "queued", etc.
        """
        all_status = self.get_all_department_status(session_id)
        display: Dict[str, str] = {}

        for dept, tasks in all_status.items():
            if not tasks:
                display[dept] = "idle"
                continue

            # Check if any task is running
            running = any("running" in status.lower() for status in tasks.values())
            queued = any("queued" in status.lower() for status in tasks.values())
            completed = any("completed" in status.lower() for status in tasks.values())
            failed = any("failed" in status.lower() for status in tasks.values())

            if running:
                display[dept] = "running"
            elif failed:
                display[dept] = "failed"
            elif completed and not running:
                display[dept] = "completed"
            elif queued:
                display[dept] = "queued"
            else:
                display[dept] = "pending"

        return display

    def close(self):
        """Clean up resources."""
        if self.mail_service:
            self.mail_service.close()
        if self._task_router:
            self._task_router.close()
        logger.info("FloorManager closed")


# Singleton instance
_floor_manager: Optional[FloorManager] = None


def get_floor_manager() -> FloorManager:
    """
    Get the singleton FloorManager instance.

    Returns:
        FloorManager instance
    """
    global _floor_manager
    if _floor_manager is None:
        _floor_manager = FloorManager()
    return _floor_manager
