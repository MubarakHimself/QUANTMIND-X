"""
Router State Management
Shared memory for the Strategy Router components.
"""

from typing import Dict, Any

class RouterState:
    """
    Central State Manager.
    In V1: In-Memory Dictionary.
    In V2: Redis Client Wrapper.
    """
    def __init__(self):
        self._state: Dict[str, Any] = {
            "current_regime": None,
            "active_bots": [],
            "risk_mandate": {},
            "last_tick_time": 0.0
        }

    def update_regime(self, report):
        self._state["current_regime"] = report

    def get_regime(self):
        return self._state.get("current_regime")

    def register_bot(self, bot_id: str):
        if bot_id not in self._state["active_bots"]:
             self._state["active_bots"].append(bot_id)

    def set_risk_mandate(self, mandate):
        self._state["risk_mandate"] = mandate
