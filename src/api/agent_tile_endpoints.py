"""
Agent Tile System

Allows agents to create new information tiles on the trading floor UI.
Tiles appear in the relevant canvas when created by an agent and are
persisted in SQLite for durability across restarts.
"""
import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent-tiles", tags=["agent-tiles"])

DB_PATH = Path(".quantmind/agent_tiles.db")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AgentTile(BaseModel):
    """Full tile representation returned by the API."""

    tile_id: str
    canvas: str  # "research" | "development" | "risk" | "trading" | "portfolio" | "workshop"
    tile_type: str  # "insight" | "alert" | "metric" | "hypothesis" | "report"
    title: str
    content: str  # markdown or JSON string
    department: str
    created_by: str
    created_at: str
    expires_at: Optional[str] = None
    pinned: bool = False
    metadata: dict = {}


class CreateTileRequest(BaseModel):
    """Request body for creating a new agent tile."""

    canvas: str
    tile_type: str
    title: str
    content: str
    department: str
    created_by: str = "agent"
    expires_at: Optional[str] = None
    metadata: dict = {}


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def _init_tile_db() -> None:
    """Create the tiles table if it does not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tiles (
                tile_id    TEXT PRIMARY KEY,
                canvas     TEXT NOT NULL,
                tile_type  TEXT NOT NULL,
                title      TEXT NOT NULL,
                content    TEXT NOT NULL,
                department TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                pinned     INTEGER DEFAULT 0,
                metadata   TEXT DEFAULT '{}',
                is_read    INTEGER DEFAULT 0
            )
            """
        )
        # Migrate existing tables that lack is_read
        try:
            conn.execute("ALTER TABLE tiles ADD COLUMN is_read INTEGER DEFAULT 0")
        except Exception:
            pass


_init_tile_db()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _row_to_tile(row: tuple) -> dict:
    """Convert a SQLite row tuple to a tile dict."""
    tile_id = row[0]
    return {
        "id": tile_id,        # frontend expects 'id'
        "tile_id": tile_id,   # keep for backwards compat
        "canvas": row[1],
        "tile_type": row[2],
        "title": row[3],
        "content": row[4],
        "department": row[5],
        "created_by": row[6],
        "created_at": row[7],
        "expires_at": row[8],
        "pinned": bool(row[9]),
        "metadata": json.loads(row[10] or "{}"),
        "is_read": bool(row[11]) if len(row) > 11 else False,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
def list_tiles(
    canvas: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 100,
):
    """List tiles, optionally filtered by canvas and/or department.

    Returns the most recent tiles (default 100, configurable via limit param).
    Returns a flat array — the Svelte UI expects AgentTile[] directly.
    """
    with sqlite3.connect(DB_PATH) as conn:
        q = "SELECT * FROM tiles WHERE 1=1"
        params: list = []
        if canvas:
            q += " AND canvas=?"
            params.append(canvas)
        if department:
            q += " AND department=?"
            params.append(department)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(min(limit, 500))
        rows = conn.execute(q, params).fetchall()
    return [_row_to_tile(r) for r in rows]


@router.post("/")
def create_tile(req: CreateTileRequest):
    """Create a new agent tile and persist it to SQLite.

    Returns the new tile_id and a status of "created".
    """
    tile_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO tiles VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                tile_id,
                req.canvas,
                req.tile_type,
                req.title,
                req.content,
                req.department,
                req.created_by,
                now,
                req.expires_at,
                0,
                json.dumps(req.metadata),
            ),
        )
    logger.info(f"Agent tile created: {tile_id} on canvas={req.canvas} by {req.created_by}")
    return {"tile_id": tile_id, "status": "created"}


@router.get("/{tile_id}")
def get_tile(tile_id: str):
    """Retrieve a single tile by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT * FROM tiles WHERE tile_id=?", (tile_id,)
        ).fetchone()
    if not row:
        return {"error": "not found"}
    return _row_to_tile(row)


@router.delete("/{tile_id}")
def delete_tile(tile_id: str):
    """Delete a tile by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM tiles WHERE tile_id=?", (tile_id,))
    return {"status": "deleted"}


@router.post("/{tile_id}/read")
def mark_tile_read(tile_id: str):
    """Mark a tile as read."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE tiles SET is_read=1 WHERE tile_id=?", (tile_id,))
    return {"status": "read", "tile_id": tile_id}
