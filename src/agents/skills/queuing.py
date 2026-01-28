"""
Queuing Skill
Provides asynchronous task management and queuing capabilities.
"""

import os
import json
import time
from typing import List, Optional
from langchain_core.tools import tool
from src.agents.skills.base import AgentSkill

class TaskQueueSkill(AgentSkill):
    def __init__(self, queue_dir: str = "data/queues"):
        super().__init__(
            name="TaskQueueSkill",
            description="Ability to manage a task queue for long-running or distributed jobs."
        )
        self.queue_dir = queue_dir
        os.makedirs(self.queue_dir, exist_ok=True)
        
        @tool
        def push_to_queue(task_name: str, payload: dict, priority: int = 1) -> str:
            """
            Push a task into the global execution queue. 
            Useful for handovers to QuantCode or long-running compilation tasks.
            """
            task_id = f"task_{int(time.time())}_{task_name}"
            task_file = os.path.join(self.queue_dir, f"{task_id}.json")
            data = {
                "id": task_id,
                "name": task_name,
                "payload": payload,
                "priority": priority,
                "timestamp": time.time(),
                "status": "PENDING"
            }
            with open(task_file, "w") as f:
                json.dump(data, f, indent=2)
            return f"Task '{task_name}' pushed to queue with ID: {task_id}"

        @tool
        def list_queue() -> str:
            """
            List all pending tasks in the queue.
            """
            tasks = []
            for f in os.listdir(self.queue_dir):
                if f.endswith(".json"):
                    with open(os.path.join(self.queue_dir, f), "r") as tf:
                        tasks.append(json.load(tf))
            
            if not tasks:
                return "The queue is currently empty."
            
            summary = "\n".join([f"- [{t['status']}] {t['name']} (ID: {t['id']})" for t in tasks])
            return f"Current Queue:\n{summary}"

        @tool
        def pop_task() -> str:
            """
            Fetch the next pending task from the queue and mark it as IN_PROGRESS.
            """
            tasks = []
            for f in os.listdir(self.queue_dir):
                if f.endswith(".json"):
                    with open(os.path.join(self.queue_dir, f), "r") as tf:
                        tasks.append(json.load(tf))
            
            pending = [t for t in tasks if t['status'] == "PENDING"]
            if not pending:
                return "No pending tasks found."
            
            # Sort by priority and timestamp
            next_task = sorted(pending, key=lambda x: (-x['priority'], x['timestamp']))[0]
            next_task['status'] = "IN_PROGRESS"
            
            task_file = os.path.join(self.queue_dir, f"{next_task['id']}.json")
            with open(task_file, "w") as f:
                json.dump(next_task, f, indent=2)
                
            return f"Task Popped: {json.dumps(next_task, indent=2)}"

        self.tools.extend([push_to_queue, list_queue, pop_task])

        self.system_prompt_addition = """
        You have access to a Task Queue. 
        - Use 'push_to_queue' to hand off tasks to other agents (e.g., Analyst pushing a TRD to QuantCode).
        - Use 'list_queue' to see what's pending.
        - Use 'pop_task' when you are in a worker cycle to pick up new work.
        """
