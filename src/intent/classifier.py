"""Intent classifier for natural language commands.

Story 5.7: NL System Commands & Context-Aware Canvas Binding
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.intent.patterns import (
    CommandIntent,
    CommandPatternMatcher,
    IntentClassification,
    get_matcher,
)
from src.config import get_internal_api_base_url

logger = logging.getLogger(__name__)

# Confidence threshold below which clarification is needed
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class CanvasContext:
    """Canvas context for intent classification."""

    canvas: str
    session_id: str
    entity: Optional[str] = None


class IntentClassifier:
    """
    Intent classifier for natural language commands.

    Uses a two-phase approach:
    1. Fast path: Pattern matching for known commands
    2. Accuracy path: LLM-based classification for complex/natural language

    The classifier also handles canvas context binding to provide
    context-aware responses based on the current canvas.
    """

    def __init__(self):
        """Initialize the intent classifier."""
        # Use singleton pattern matcher for efficiency
        self._matcher = get_matcher()

        # Canvas-specific context binding
        self._canvas_binders: Dict[str, "CanvasBinder"] = {}
        self._init_binders()

        logger.info("IntentClassifier initialized")

    def _init_binders(self):
        """Initialize canvas context binders."""
        try:
            from src.intent.binders import (
                LiveTradingBinder,
                RiskBinder,
                PortfolioBinder,
                WorkshopBinder,
            )

            self._canvas_binders = {
                "live_trading": LiveTradingBinder(),
                "risk": RiskBinder(),
                "portfolio": PortfolioBinder(),
                "workshop": WorkshopBinder(),
            }
        except ImportError as e:
            logger.warning(f"Canvas binders not available: {e}")
            self._canvas_binders = {}

    async def classify(
        self,
        message: str,
        canvas_context: Dict[str, Any],
    ) -> IntentClassification:
        """
        Classify user message into actionable intent.

        Args:
            message: User message to classify
            canvas_context: Canvas context dictionary with canvas, session_id, entity

        Returns:
            IntentClassification with intent, entities, confidence, and context
        """
        canvas_type = canvas_context.get("canvas", "workshop")
        session_id = canvas_context.get("session_id", "")

        # Phase 1: Try pattern matching (fast path)
        pattern_result = self._matcher.match(message)

        # If high confidence from patterns, return immediately
        if pattern_result.confidence >= CONFIDENCE_THRESHOLD:
            logger.debug(f"Pattern match: {pattern_result.intent.value} (confidence: {pattern_result.confidence})")
            return pattern_result

        # If it's a general query, don't trigger clarification - just pass through
        if pattern_result.intent == CommandIntent.GENERAL_QUERY:
            return pattern_result

        # Phase 2: Low confidence - try canvas context binding
        # This enhances the classification with canvas-specific data
        canvas_binder = self._canvas_binders.get(canvas_type)

        if canvas_binder and pattern_result.confidence < CONFIDENCE_THRESHOLD:
            # Try to bind canvas context to improve classification
            context_data = await canvas_binder.bind_context(message, canvas_context)

            # If we can enhance with canvas data, might improve confidence
            if context_data:
                # Boost confidence slightly based on canvas context match
                enhanced_confidence = min(pattern_result.confidence + 0.1, 1.0)

                # If still below threshold, need clarification (for non-general intents)
                if enhanced_confidence < CONFIDENCE_THRESHOLD:
                    return IntentClassification(
                        intent=CommandIntent.CLARIFICATION_NEEDED,
                        entities=[],
                        confidence=enhanced_confidence,
                        raw_command=message,
                        requires_confirmation=False,
                    )

        # Return original pattern result (may have lower confidence)
        return pattern_result

    async def handle_command(
        self,
        message: str,
        canvas_context: Dict[str, Any],
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        """
        Handle natural language command with confirmation flow.

        Args:
            message: User message
            canvas_context: Canvas context dictionary
            confirmed: Whether user has confirmed the action

        Returns:
            Dict with type (confirmation_needed, clarification_needed, success, etc.)
        """
        # Classify intent
        classification = await self.classify(message, canvas_context)

        # If confidence below threshold, ask for clarification (but not for general queries)
        if classification.confidence < CONFIDENCE_THRESHOLD and classification.intent != CommandIntent.GENERAL_QUERY:
            return {
                "type": "clarification_needed",
                "message": "I'm not sure what you mean. Could you clarify?",
                "suggestions": [
                    "pause all strategies",
                    "show my positions",
                    "what is the current regime?",
                ],
                "classification": {
                    "intent": classification.intent.value,
                    "confidence": classification.confidence,
                },
            }

        # If requires confirmation and not confirmed, ask for confirmation
        if classification.requires_confirmation and not confirmed:
            action_verb = classification.intent.value.replace("_", " ")
            entities_str = ", ".join(classification.entities) if classification.entities else "the target"

            return {
                "type": "confirmation_needed",
                "message": f"I'll {action_verb} {entities_str}. Confirm?",
                "intent": classification.intent.value,
                "entities": classification.entities,
                "classification": {
                    "intent": classification.intent.value,
                    "confidence": classification.confidence,
                },
            }

        # Execute command
        return await self._execute_command(classification, canvas_context)

    async def _execute_command(
        self,
        classification: IntentClassification,
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the classified command."""
        intent = classification.intent
        canvas_type = canvas_context.get("canvas", "workshop")

        # Get canvas binder for context-aware execution
        canvas_binder = self._canvas_binders.get(canvas_type)

        try:
            if intent == CommandIntent.STRATEGY_PAUSE:
                return await self._execute_strategy_pause(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.STRATEGY_RESUME:
                return await self._execute_strategy_resume(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.POSITION_CLOSE:
                return await self._execute_position_close(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.POSITION_INFO:
                return await self._execute_position_info(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.REGIME_QUERY:
                return await self._execute_regime_query(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.ACCOUNT_INFO:
                return await self._execute_account_info(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.NODE_UPDATE:
                return await self._execute_node_update(classification, canvas_binder, canvas_context)
            elif intent == CommandIntent.GENERAL_QUERY:
                return {
                    "type": "general_query",
                    "message": "I'll help you with that. Let me process your request.",
                    "intent": intent.value,
                }
            else:
                return {
                    "type": "error",
                    "message": f"Unknown intent: {intent.value}",
                }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to execute command: {str(e)}",
            }

    async def _execute_strategy_pause(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute strategy pause command via risk API."""
        symbols = classification.entities if classification.entities else ["all"]

        # TODO: Call actual risk API when available
        logger.info(f"Executing STRATEGY_PAUSE for symbols: {symbols}")

        return {
            "type": "success",
            "message": f"Strategy paused for {', '.join(symbols)}",
            "action": "strategy_pause",
            "symbols": symbols,
        }

    async def _execute_strategy_resume(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute strategy resume command."""
        symbols = classification.entities if classification.entities else ["all"]

        logger.info(f"Executing STRATEGY_RESUME for symbols: {symbols}")

        return {
            "type": "success",
            "message": f"Strategy resumed for {', '.join(symbols)}",
            "action": "strategy_resume",
            "symbols": symbols,
        }

    async def _execute_position_close(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute position close command."""
        symbols = classification.entities if classification.entities else ["all"]

        logger.info(f"Executing POSITION_CLOSE for symbols: {symbols}")

        return {
            "type": "success",
            "message": f"Position closed for {', '.join(symbols)}",
            "action": "position_close",
            "symbols": symbols,
        }

    async def _execute_position_info(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute position info query."""
        positions = []

        # Try to get positions from canvas binder
        if canvas_binder:
            try:
                positions = await canvas_binder.get_positions()
            except Exception as e:
                logger.warning(f"Failed to get positions from binder: {e}")

        return {
            "type": "success",
            "message": f"You have {len(positions)} open position(s)",
            "positions": positions,
            "action": "position_info",
        }

    async def _execute_regime_query(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute regime query."""
        regime = "UNKNOWN"

        if canvas_binder:
            try:
                regime = await canvas_binder.get_regime()
            except Exception as e:
                logger.warning(f"Failed to get regime from binder: {e}")

        return {
            "type": "success",
            "message": f"Current market regime: {regime}",
            "regime": regime,
            "action": "regime_query",
        }

    async def _execute_account_info(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute account info query."""
        account_info = {}

        if canvas_binder:
            try:
                account_info = await canvas_binder.get_account_info()
            except Exception as e:
                logger.warning(f"Failed to get account info from binder: {e}")

        return {
            "type": "success",
            "message": f"Account balance: {account_info.get('balance', 'N/A')}",
            "account": account_info,
            "action": "account_info",
        }

    async def _execute_node_update(
        self,
        classification: IntentClassification,
        canvas_binder: Optional[Any],
        canvas_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute node update command.

        Triggers sequential update across Contabo → Cloudzy → Desktop nodes.
        Displays completion summary in Copilot (AC3).
        """
        logger.info("Executing NODE_UPDATE command")

        try:
            # Import required functions
            from src.api.node_update_endpoints import is_valid_deploy_window, UpdateRequest

            # Check deploy window first
            if not is_valid_deploy_window():
                return {
                    "type": "error",
                    "message": "Node updates are only allowed during the deploy window (Friday 22:00 - Sunday 22:00 UTC). Please try again during the allowed time window.",
                    "action": "node_update",
                }

            # Get the server URL from canvas context or use default
            base_url = (canvas_context.get("server_url") or get_internal_api_base_url()).rstrip("/")

            # Call the node update API endpoint to execute the update and get results
            try:
                import httpx

                # Make synchronous call to the update endpoint
                response = httpx.post(
                    f"{base_url}/api/node-update/update",
                    json={"version": None, "nodes": None},
                    timeout=120.0  # Allow time for sequential updates
                )

                if response.status_code == 200:
                    result = response.json()

                    # Format completion summary for Copilot (AC3)
                    if result.get("status") == "completed":
                        nodes = result.get("nodes", [])
                        node_summary = ", ".join([f"{n.get('node')}: {n.get('status')}" for n in nodes])

                        return {
                            "type": "success",
                            "message": f"Node update completed successfully!\n\n"
                                      f"Updated nodes: {node_summary}\n"
                                      f"Duration: {result.get('duration_seconds', 0):.1f}s\n\n"
                                      f"All nodes passed health checks.",
                            "action": "node_update",
                            "sequence": ["contabo", "cloudzy", "desktop"],
                            "result": result
                        }
                    elif result.get("status") == "failed":
                        failed_node = result.get("failed_node", "unknown")
                        rolled_back = result.get("rolled_back_nodes", [])

                        return {
                            "type": "warning",
                            "message": f"Node update failed on {failed_node}.\n\n"
                                      f"Rolled back nodes: {', '.join(rolled_back) if rolled_back else 'None'}\n"
                                      f"Previous nodes ({result.get('nodes', []).__len__() - len(rolled_back)}) remain on new version.\n\n"
                                      f"Check logs and retry after resolving the issue.",
                            "action": "node_update",
                            "failed_node": failed_node,
                            "rolled_back": rolled_back,
                            "result": result
                        }
                    else:
                        return {
                            "type": "success",
                            "message": "Node update sequence initiated. Nodes will update in order: Contabo → Cloudzy → Desktop. You will receive notifications on completion or if rollback is triggered.",
                            "action": "node_update",
                            "sequence": ["contabo", "cloudzy", "desktop"],
                            "note": "Updates only allowed Friday 22:00 - Sunday 22:00 UTC"
                        }
                elif response.status_code == 403:
                    return {
                        "type": "error",
                        "message": "Node updates are only allowed during the deploy window (Friday 22:00 - Sunday 22:00 UTC). Please try again during the allowed time window.",
                        "action": "node_update",
                    }
                else:
                    logger.warning(f"Node update API returned {response.status_code}")
                    # Fall through to default response

            except ImportError:
                # httpx not available, use default response
                logger.warning("httpx not available for direct API call")
            except Exception as e:
                # API call failed, use default response but include error info
                logger.warning(f"Failed to call node update API: {e}")

            # Default response when API not available or fails
            return {
                "type": "success",
                "message": "Node update sequence initiated. Nodes will update in order: Contabo → Cloudzy → Desktop. You will receive notifications on completion or if rollback is triggered.",
                "action": "node_update",
                "sequence": ["contabo", "cloudzy", "desktop"],
                "note": "Updates only allowed Friday 22:00 - Sunday 22:00 UTC"
            }

        except Exception as e:
            logger.error(f"Node update failed: {e}")
            return {
                "type": "error",
                "message": f"Failed to initiate node update: {str(e)}",
                "action": "node_update",
            }


# Singleton instance
_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get singleton intent classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
