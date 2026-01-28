"""
QuantMind System Database (Cloud Ledger)
Powered by Supabase (PostgreSQL)

Acts as the Single Source of Truth for:
- Agent Missions & Status
- Webhook/Task Tracking
- System Prompts & Configurations
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    from supabase import create_client, Client
except ImportError:
    Client = Any # type mocking for dev without deps installed

logger = logging.getLogger(__name__)

class QuantMindSysDB:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantMindSysDB, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            logger.warning("Supabase credentials not found. DB features will be disabled.")
            self.client: Optional[Client] = None
            return

        try:
            self.client = create_client(url, key)
            logger.info("Connected to QuantMind System DB (Supabase)")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None

    def create_mission(self, user_id: str, objective: str) -> str:
        """Create a new high-level mission."""
        if not self.client: return "mock_mission_id"
        
        data = {
            "user_id": user_id,
            "objective": objective,
            "status": "PLANNING",
            "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}
        }
        res = self.client.table("missions").insert(data).execute()
        return res.data[0]['id']

    def update_mission_status(self, mission_id: str, status: str, step: str = None):
        if not self.client: return
        data = {"status": status}
        if step: data["current_step"] = step
        self.client.table("missions").update(data).eq("id", mission_id).execute()

    def submit_task(self, mission_id: str, agent_id: str, payload: Dict[str, Any]) -> str:
        """Submit a task to the queue (for the Agent Hook to pick up)."""
        if not self.client: return "mock_task_id"
        
        data = {
            "mission_id": mission_id,
            "agent_id": agent_id,
            "payload": payload,
            "status": "PENDING",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        res = self.client.table("tasks").insert(data).execute()
        return res.data[0]['id']

    def get_agent_prompt(self, agent_name: str) -> str:
        """Fetch the dynamic system prompt for an agent."""
        if not self.client: return None
        
        res = self.client.table("agents").select("system_prompt").eq("name", agent_name).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]['system_prompt']
        return None

# Global Singleton Accessor
sys_db = QuantMindSysDB()
