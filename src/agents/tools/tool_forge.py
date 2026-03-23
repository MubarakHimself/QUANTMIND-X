"""
Tool Forge — Dynamic Tool Creation System

Allows agents to create, register, and share new tools at runtime.
Tools are stored in SQLite and loaded on demand.
"""
import json
import sqlite3
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DB_PATH = Path(".quantmind/tool_forge.db")

@dataclass
class ForgeTool:
    tool_id: str
    name: str
    description: str
    department: str  # which dept created it, or "shared"
    input_schema: Dict[str, Any]
    implementation: str  # Python code as string (sandboxed exec)
    created_by: str  # agent_id or dept name
    created_at: str
    usage_count: int = 0
    active: bool = True

class ToolForge:
    """Runtime tool creation and registry."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    tool_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    department TEXT DEFAULT 'shared',
                    input_schema TEXT,
                    implementation TEXT,
                    created_by TEXT,
                    created_at TEXT,
                    usage_count INTEGER DEFAULT 0,
                    active INTEGER DEFAULT 1
                )
            """)

    def register_tool(self, name: str, description: str, input_schema: dict,
                      implementation: str, department: str = "shared", created_by: str = "system") -> ForgeTool:
        tool = ForgeTool(
            tool_id=str(uuid.uuid4()),
            name=name,
            description=description,
            department=department,
            input_schema=input_schema,
            implementation=implementation,
            created_by=created_by,
            created_at=datetime.utcnow().isoformat(),
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tools
                (tool_id, name, description, department, input_schema, implementation, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (tool.tool_id, tool.name, tool.description, tool.department,
                  json.dumps(tool.input_schema), tool.implementation, tool.created_by, tool.created_at))
        logger.info(f"Tool registered: {name} by {created_by}")
        return tool

    def list_tools(self, department: Optional[str] = None) -> List[ForgeTool]:
        with sqlite3.connect(self.db_path) as conn:
            if department:
                rows = conn.execute(
                    "SELECT * FROM tools WHERE active=1 AND (department=? OR department='shared')",
                    (department,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tools WHERE active=1").fetchall()
        return [self._row_to_tool(r) for r in rows]

    def get_tool(self, name: str) -> Optional[ForgeTool]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM tools WHERE name=? AND active=1", (name,)).fetchone()
        return self._row_to_tool(row) if row else None

    def execute_tool(self, name: str, inputs: dict) -> dict:
        """Execute a forged tool in a restricted sandbox."""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}

        # Increment usage
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE tools SET usage_count=usage_count+1 WHERE name=?", (name,))

        # Sandboxed execution
        sandbox = {
            "__builtins__": {
                "print": print, "len": len, "str": str, "int": int,
                "float": float, "list": list, "dict": dict, "json": json
            },
            "inputs": inputs,
            "result": None
        }
        try:
            exec(tool.implementation, sandbox)
            return {"result": sandbox.get("result"), "tool": name}
        except Exception as e:
            return {"error": str(e), "tool": name}

    def deactivate_tool(self, name: str) -> bool:
        """Deactivate a tool by name. Returns True if found and deactivated."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("UPDATE tools SET active=0 WHERE name=? AND active=1", (name,))
            affected = cursor.rowcount
        if affected:
            logger.info(f"Tool deactivated: {name}")
        return bool(affected)

    def _row_to_tool(self, row) -> ForgeTool:
        return ForgeTool(
            tool_id=row[0], name=row[1], description=row[2], department=row[3],
            input_schema=json.loads(row[4] or "{}"), implementation=row[5],
            created_by=row[6], created_at=row[7], usage_count=row[8], active=bool(row[9])
        )


_forge = None


def get_tool_forge() -> ToolForge:
    global _forge
    if _forge is None:
        _forge = ToolForge()
    return _forge
