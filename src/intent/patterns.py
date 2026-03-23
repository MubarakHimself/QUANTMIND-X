"""Command pattern matching for intent classification.

Story 5.7: NL System Commands & Context-Aware Canvas Binding
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class CommandIntent(Enum):
    """Supported command intents for natural language commands."""

    STRATEGY_PAUSE = "strategy_pause"
    STRATEGY_RESUME = "strategy_resume"
    POSITION_CLOSE = "position_close"
    POSITION_INFO = "position_info"
    REGIME_QUERY = "regime_query"
    ACCOUNT_INFO = "account_info"
    GENERAL_QUERY = "general_query"
    CLARIFICATION_NEEDED = "clarification_needed"
    # Story 10.4: Audit Query intents
    AUDIT_TIMELINE_QUERY = "audit_timeline_query"
    AUDIT_REASONING_QUERY = "audit_reasoning_query"
    # Story 11.3: Node Update intents
    NODE_UPDATE = "node_update"
    # Story 11.4: Backup/Restore intents
    BACKUP_SYSTEM = "backup_system"
    RESTORE_BACKUP = "restore_backup"
    BACKUP_QUERY = "backup_query"


# Destructive commands that require confirmation
DESTRUCTIVE_INTENTS = {
    CommandIntent.STRATEGY_PAUSE,
    CommandIntent.STRATEGY_RESUME,
    CommandIntent.POSITION_CLOSE,
    CommandIntent.NODE_UPDATE,  # Node update can trigger rollbacks
    CommandIntent.RESTORE_BACKUP,  # Restore is destructive - requires confirmation
}

# Intent to command pattern mappings
INTENT_PATTERNS = {
    CommandIntent.STRATEGY_PAUSE: [
        r"\bpause\b.*\b(strategy|trading)\b",
        r"\bstop\b.*\b(strategy|trading)\b",
        r"\bhalt\b.*\b(strategy|trading)\b",
        r"\bsuspend\b.*\b(strategy|trading)\b",
    ],
    CommandIntent.STRATEGY_RESUME: [
        r"\bresume\b.*\b(strategy|trading)\b",
        r"\bstart\b.*\b(strategy|trading)\b",
        r"\bcontinue\b.*\b(strategy|trading)\b",
        r"\breactivate\b.*\b(strategy|trading)\b",
    ],
    CommandIntent.POSITION_CLOSE: [
        r"\bclose\b.*\b(position|trade)\b",
        r"\bexit\b.*\b(position|trade)\b",
        r"\bliquidate\b.*\b(position|trade)\b",
    ],
    CommandIntent.POSITION_INFO: [
        r"\b(show|get|list|display)\b.*\b(positions|trades|open positions|my positions)\b",
        r"\bwhat\s+(are\s+)?(my\s+)?(open\s+)?positions\b",
        r"\bposition\s+info\b",
    ],
    CommandIntent.REGIME_QUERY: [
        r"\b(regime|market regime|trend|market state)\b",
        r"\bwhat\s+is\s+(the\s+)?(market\s+)?regime\b",
    ],
    CommandIntent.ACCOUNT_INFO: [
        r"\b(account|balance|equity)\b.*\b(info|status|check)\b",
        r"\bwhat\s+is\s+(my\s+)?(account|balance|equity)\b",
    ],
    # Story 10.4: Audit Timeline Query - "Why was X paused?" queries
    CommandIntent.AUDIT_TIMELINE_QUERY: [
        r"\bwhy\b.*\b(paused|stopped|halted|suspended)\b",
        r"\bwhy\b.*\b(closed|liquidated)\b",
        r"\bwhat\s+happened\b.*\b(yesterday|last\s+\w+)\b",
        r"\bshow\s+me\s+the\s+(timeline|history)\b.*\b(yesterday|last\s+\w+)\b",
        r"\bwhat\s+caused\b.*\b(pause|stop|close)\b",
        r"\bexplain\b.*\b(why|what)\b.*\b(paused|stopped)\b",
        r"\bwas\b.*\bpaused\b.*\b(at\s+)?\d{1,2}:\d{2}\b",
    ],
    # Story 10.4: Audit Reasoning Query - "Show reasoning" queries
    CommandIntent.AUDIT_REASONING_QUERY: [
        r"\bshow\s+me\s+the\s+reasoning\b",
        r"\bshow\s+me\s+(the\s+)?(reasoning|decision)\s+(chain|logic)\b",
        r"\bwhy\s+did\s+(the\s+)?(\w+\s+)?department\b.*\brecommend\b",
        r"\bexplain\s+(the\s+)?reasoning\b",
        r"\bwhat\s+was\s+the\s+reasoning\b.*\b(recommendation|decision)\b",
        r"\bshow\s+(me\s+)?(the\s+)?opinion\b.*\b(chain|nodes)\b",
    ],
    # Story 11.3: Node Update - "update all nodes", "deploy new version"
    CommandIntent.NODE_UPDATE: [
        r"\bupdate\s+all\s+node[sz]\b",
        r"\bdeploy\s+(new\s+)?version\b",
        r"\bupdate\s+(all\s+)?server[sz]?\b",
        r"\brollout\s+(new\s+)?version\b",
        r"\bupdate\s+contabo\b.*\bcloudzy\b.*\bdesktop\b",
    ],
    # Story 11.4: Backup intents - "backup system", "create backup"
    CommandIntent.BACKUP_SYSTEM: [
        r"\bbackup\s+(the\s+)?system\b",
        r"\bcreate\s+(a\s+)?backup\b",
        r"\bfull\s+backup\b",
        r"\bbackup\s+now\b",
        r"\brun\s+backup\b",
    ],
    # Story 11.4: Restore intents - "restore from backup", "restore system"
    CommandIntent.RESTORE_BACKUP: [
        r"\brestore\s+(from\s+)?backup\b",
        r"\brestore\s+(the\s+)?system\b",
        r"\brecover\s+from\s+backup\b",
    ],
    # Story 11.4: Backup query - "show backups", "list backups", "backup status"
    CommandIntent.BACKUP_QUERY: [
        r"\bshow\s+(me\s+)?backup[sz]\b",
        r"\blist\s+backup[sz]\b",
        r"\bbackup\s+status\b",
        r"\bwhat.*backup\b",
        r"\bbackup\s+info\b",
        r"\bbackup\s+manifest\b",
    ],
}


# Entity extraction patterns
ENTITY_PATTERNS = {
    "symbol": r"\b[A-Z]{3,6}(?:USD|EUR|GBP|JPY)?\b",  # Currency pairs like GBPUSD, EUR
    "all": r"\ball\b|\beverything\b|\ball\s+positions\b",
}


@dataclass
class IntentClassification:
    """Result of intent classification."""

    intent: CommandIntent
    entities: List[str]
    confidence: float
    raw_command: str
    requires_confirmation: bool


class CommandPatternMatcher:
    """Pattern-based command matcher for fast intent classification."""

    def __init__(self):
        """Initialize the matcher with compiled patterns."""
        self._compiled_patterns: dict[CommandIntent, List[re.Pattern]] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def match(self, message: str) -> IntentClassification:
        """
        Match a message against command patterns.

        Args:
            message: User message to classify

        Returns:
            IntentClassification with intent, entities, and confidence
        """
        message_lower = message.lower().strip()

        # Try each intent pattern
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    # Extract entities
                    entities = self._extract_entities(message)

                    # Calculate confidence based on pattern match quality
                    confidence = self._calculate_confidence(message, intent, pattern)

                    return IntentClassification(
                        intent=intent,
                        entities=entities,
                        confidence=confidence,
                        raw_command=message,
                        requires_confirmation=intent in DESTRUCTIVE_INTENTS,
                    )

        # No pattern matched - return general query
        return IntentClassification(
            intent=CommandIntent.GENERAL_QUERY,
            entities=[],
            confidence=0.5,
            raw_command=message,
            requires_confirmation=False,
        )

    def _extract_entities(self, message: str) -> List[str]:
        """Extract entities from message (symbols, etc.)."""
        entities = []

        # Extract currency pair symbols
        symbol_matches = re.findall(ENTITY_PATTERNS["symbol"], message.upper())
        entities.extend(symbol_matches)

        # Check for "all" entities
        if re.search(ENTITY_PATTERNS["all"], message.lower()):
            entities.append("all")

        return list(set(entities))

    def _calculate_confidence(
        self, message: str, intent: CommandIntent, pattern: re.Pattern
    ) -> float:
        """Calculate confidence score for the match."""
        # Base confidence for pattern match
        base_confidence = 0.85

        # Boost confidence for exact keyword matches
        message_lower = message.lower()

        # Check for explicit intent keywords
        explicit_keywords = {
            CommandIntent.STRATEGY_PAUSE: ["pause", "stop", "halt", "suspend"],
            CommandIntent.STRATEGY_RESUME: ["resume", "start", "continue", "reactivate"],
            CommandIntent.POSITION_CLOSE: ["close", "exit", "liquidate"],
            CommandIntent.POSITION_INFO: ["show", "get", "list", "display", "positions"],
        }

        if intent in explicit_keywords:
            keywords = explicit_keywords[intent]
            if any(kw in message_lower for kw in keywords):
                base_confidence = min(base_confidence + 0.1, 1.0)

        return base_confidence


# Singleton instance
_matcher: Optional[CommandPatternMatcher] = None


def get_matcher() -> CommandPatternMatcher:
    """Get singleton pattern matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = CommandPatternMatcher()
    return _matcher
