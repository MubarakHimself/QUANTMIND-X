"""
QuantMind Agent Hook Manager.
The interface for "Service-Oriented" Agent communication.
Ensures that Co-pilot interacts with Agents via standardized protocols.
"""

import logging
import asyncio
from typing import Dict, Any, Callable
from src.agents.core.database import sys_db

logger = logging.getLogger(__name__)

# Registry of active hook handlers (local execution for now)
# In V3, these could be HTTP endpoints.
_LOCAL_HOOK_REGISTRY: Dict[str, Callable] = {}

class AgentHookManager:
    @staticmethod
    def register_agent(agent_name: str, handler_func: Callable):
        """Register a local function as a hook handler."""
        _LOCAL_HOOK_REGISTRY[agent_name] = handler_func
        logger.info(f"Hook registered for agent: {agent_name}")

    @staticmethod
    async def submit_job(agent_name: str, task_type: str, payload: Dict[str, Any], mission_id: str = "orphan") -> Dict[str, Any]:
        """
        Submit a job to an agent hook.
        
        Args:
            agent_name: Target agent (e.g., 'analyst').
            task_type: The intent (e.g., 'DESIGN', 'BUILD').
            payload: The data context.
            mission_id: The global mission ID this task belongs to.
            
        Returns:
            The result dictionary from the agent.
        """
        # 1. Log the intent to the System DB
        task_id = sys_db.submit_task(mission_id, agent_name, {
            "type": task_type,
            "raw_payload": payload
        })
        
        # 2. Check for "Pull the Plug" or "Circuit Break" status
        # (TODO: Implement DB check for agent_status == 'PAUSED')

        # 3. Route to Handler
        handler = _LOCAL_HOOK_REGISTRY.get(agent_name)
        if not handler:
            raise ValueError(f"No active hook found for agent: {agent_name}")
        
        logger.info(f"HookManager: Routing {task_type} to {agent_name} (Task {task_id})")
        
        try:
            # Execute the agent logic
            result = await handler(payload)
            
            # 4. Update Success Status
            # sys_db.update_task(task_id, status="COMPLETED", result=result) 
            return result
        except Exception as e:
            logger.error(f"Hook execution failed for {agent_name}: {e}")
            # sys_db.update_task(task_id, status="FAILED", error=str(e))
            raise e

hook_manager = AgentHookManager()
