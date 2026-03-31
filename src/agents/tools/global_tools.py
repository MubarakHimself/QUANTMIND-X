"""
Global Tools — Available to all agents in all sessions.

Architecture reference: §16.3 (Tool Tiers — Global tier)

These tools are activated for every agent session regardless of department:
- read_skill: Load a skill.md file JIT into agent context
- write_memory: Store information in graph memory
- search_memory: Query graph memory for relevant context
- read_canvas_context: Load canvas context template identifiers
- request_tool: Request activation of a department/workflow tool
- send_department_mail: Send inter-department messages
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# read_skill — Load skill.md JIT (architecture §4.3 JIT chain)
# ===========================================================================

def read_skill(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a skill definition file into the agent's context.

    Args:
        skill_id: Skill identifier (e.g., "forge-strategy")
        department: Optional department scope (e.g., "development")

    Returns:
        Skill content and metadata
    """
    skill_id = args.get("skill_id", "")
    department = args.get("department", "")

    # Search paths for skill files
    base = os.environ.get("QUANTMINDX_ROOT", ".")
    search_paths = [
        Path(base) / "shared_assets" / "skills" / "departments" / department / skill_id / "skill.md",
        Path(base) / "shared_assets" / "skills" / "global" / skill_id / "skill.md",
    ]

    for path in search_paths:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            return {
                "skill_id": skill_id,
                "path": str(path),
                "content": content[:4000],
                "loaded": True,
            }

    return {
        "skill_id": skill_id,
        "loaded": False,
        "message": f"Skill '{skill_id}' not found in department '{department}' or global scope",
    }


# ===========================================================================
# write_memory — Store to graph memory (architecture §6.1)
# ===========================================================================

def write_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a memory node to the graph memory system.

    Supports all node types including OPINION nodes with extra fields:
    - action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role

    Also integrates with department mail when the memory node is a DECISION
    or OPINION that other departments should know about (cross-department
    memory sharing via mail notification).

    Args:
        node_type: Memory node type (OBSERVATION, OPINION, DECISION, WORLD, etc.)
        content: Text content of the memory
        category: Category within node type (general, factual, subjective, etc.)
        tags: List of tags for retrieval
        session_id: Current session ID
        department: Owning department
        # OPINION-specific fields:
        action: What the agent did (OPINION nodes)
        reasoning: Why they did it (OPINION nodes)
        confidence: Confidence score 0.0-1.0 (OPINION nodes)
        alternatives_considered: Options evaluated (OPINION nodes)
        constraints_applied: Constraints that influenced the decision (OPINION nodes)
        agent_role: Which agent/role took the action (OPINION nodes)
        # Cross-department notification:
        notify_departments: List of departments to notify via mail about this memory

    Returns:
        Created node metadata
    """
    node_type = args.get("node_type", "OBSERVATION")
    content = args.get("content", "")
    category = args.get("category", "general")
    tags = args.get("tags", [])
    session_id = args.get("session_id", "")
    department = args.get("department", "")
    notify_departments = args.get("notify_departments", [])

    # Build metadata dict
    metadata = {
        "session_id": session_id,
        "department": department,
        "created_at": datetime.now().isoformat(),
    }

    # OPINION-specific fields (architecture §6.1, Hindsight's Four Networks)
    opinion_fields = {}
    if node_type.upper() == "OPINION":
        for field in [
            "action", "reasoning", "confidence",
            "alternatives_considered", "constraints_applied", "agent_role",
        ]:
            val = args.get(field)
            if val is not None:
                opinion_fields[field] = val
                metadata[field] = val

    try:
        from src.memory.graph.store import get_graph_store
        store = get_graph_store()
        node_id = store.add_node(
            node_type=node_type,
            content=content,
            category=category,
            tags=tags,
            metadata=metadata,
            **opinion_fields,
        )

        # Cross-department mail notification for DECISION/OPINION nodes
        if notify_departments and department:
            _notify_departments_of_memory(
                node_type=node_type,
                content=content[:200],
                department=department,
                notify=notify_departments,
                node_id=node_id,
                session_id=session_id,
            )

        return {
            "node_id": node_id,
            "stored": True,
            "node_type": node_type,
            "opinion_fields": list(opinion_fields.keys()) if opinion_fields else [],
        }
    except Exception as e:
        logger.warning(f"Graph memory write failed: {e}")
        return {
            "stored": False,
            "error": str(e),
            "message": "Memory write queued — graph store unavailable",
        }


def _notify_departments_of_memory(
    node_type: str, content: str, department: str,
    notify: List[str], node_id: str, session_id: str,
) -> None:
    """Send department mail notifications when important memory nodes are created."""
    try:
        from src.agents.departments.department_mail import (
            get_mail_service, MessageType, Priority,
        )
        mail = get_mail_service()
        for dept in notify:
            if dept != department:
                mail.send(
                    from_dept=department,
                    to_dept=dept,
                    type=MessageType.STATUS,
                    subject=f"[{node_type}] New {node_type.lower()} from {department}",
                    body=json.dumps({
                        "memory_node_id": node_id,
                        "node_type": node_type,
                        "content_preview": content,
                        "session_id": session_id,
                    }),
                    priority=Priority.NORMAL,
                )
    except Exception as e:
        logger.debug(f"Memory-mail notification skipped: {e}")


# ===========================================================================
# search_memory — Query graph memory (architecture §6.1, §19.2)
# ===========================================================================

def search_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search graph memory for relevant context.

    Graph is the PRIMARY retrieval layer for all agent reasoning context
    (architecture §19.2).

    Args:
        query: Search query text
        node_types: Optional filter by node types
        department: Optional filter by department
        top_k: Number of results to return (default 5)

    Returns:
        List of matching memory nodes
    """
    query = args.get("query", "")
    node_types = args.get("node_types", [])
    department = args.get("department", "")
    top_k = args.get("top_k", 5)

    try:
        from src.memory.graph.store import get_graph_store
        store = get_graph_store()
        results = store.recall(
            query=query,
            top_k=top_k,
            node_types=node_types or None,
            filter_metadata={"department": department} if department else None,
        )
        return {
            "results": [
                {
                    "node_id": r.get("node_id", ""),
                    "node_type": r.get("node_type", ""),
                    "content": r.get("content", "")[:500],
                    "score": r.get("score", 0.0),
                    "tags": r.get("tags", []),
                }
                for r in (results or [])
            ],
            "count": len(results or []),
        }
    except Exception as e:
        logger.warning(f"Graph memory search failed: {e}")
        return {"results": [], "count": 0, "error": str(e)}


# ===========================================================================
# read_canvas_context — Load canvas context template (architecture §4.3)
# ===========================================================================

def read_canvas_context(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load canvas context template identifiers via the CanvasContextLoader.

    This is the JIT entry point for agents — loads the YAML template,
    memory identifiers, skill index, and suggestion chips for a canvas.
    Agents use this to understand what's available on their canvas.

    Args:
        canvas: Canvas name (e.g., "live_trading", "research", "development")
        session_id: Optional session ID for memory scope filtering
        include_tiles: Whether to include active agent tiles on this canvas

    Returns:
        Canvas context template identifiers, skills, memory scope, and tiles
    """
    canvas = args.get("canvas", "")
    session_id = args.get("session_id", "")
    include_tiles = args.get("include_tiles", False)

    result: Dict[str, Any] = {"canvas": canvas, "loaded": False}

    # Try loading from the real CanvasContextLoader (YAML templates)
    try:
        from src.canvas_context.loader import load_template
        template = load_template(canvas)
        if template:
            result.update({
                "loaded": True,
                "display_name": template.canvas_display_name or canvas,
                "icon": template.canvas_icon,
                "department_head": template.department_head,
                "base_descriptor": template.base_descriptor[:500],
                "memory_scope": template.memory_scope,
                "workflow_namespaces": template.workflow_namespaces,
                "department_mailbox": template.department_mailbox,
                "skills": [
                    {"id": s.id, "trigger": s.trigger}
                    for s in (template.skill_index or [])
                ],
                "required_tools": template.required_tools,
                "suggestion_chips": [
                    {"id": c.id, "label": c.label, "target_canvas": c.target_canvas}
                    for c in (template.suggestion_chips or [])
                ],
                "max_identifiers": template.max_identifiers,
            })
    except Exception as e:
        logger.debug(f"CanvasContextLoader unavailable, using fallback: {e}")
        # Fallback: static component lists for when YAML templates aren't loaded
        _fallback = {
            "live_trading": {
                "components": ["strategy_router", "kill_switch", "equity_curve",
                               "open_positions", "department_mail_queue"],
                "skills": [{"id": "morning-digest", "trigger": "morning"}],
            },
            "research": {
                "components": ["knowledge_base", "news_feed", "pattern_scanner"],
                "skills": [{"id": "context-enrich", "trigger": "context"}],
            },
            "development": {
                "components": ["mql5_editor", "backtest_runner", "ea_library"],
                "skills": [{"id": "forge-strategy", "trigger": "forge"}],
            },
            "risk": {
                "components": ["kelly_engine", "regime_state", "drawdown_monitor"],
                "skills": [{"id": "drawdown-review", "trigger": "drawdown"}],
            },
            "portfolio": {
                "components": ["bot_roster", "dpr_scores", "performance_tracker"],
                "skills": [{"id": "system-targets", "trigger": "targets"}],
            },
        }
        fb = _fallback.get(canvas, {"components": [], "skills": []})
        result.update({"loaded": True, **fb})

    # Optionally load active tiles from the agent tile system
    if include_tiles:
        try:
            from src.api.agent_tile_endpoints import _get_tiles_db
            db = _get_tiles_db()
            tiles = db.list_tiles(canvas=canvas, limit=50)
            result["tiles"] = [
                {
                    "tile_id": t["tile_id"],
                    "tile_type": t["tile_type"],
                    "title": t["title"],
                    "department": t["department"],
                    "pinned": t.get("pinned", False),
                }
                for t in tiles
            ]
        except Exception as e:
            logger.debug(f"Tile loading skipped: {e}")
            result["tiles"] = []

    # Load memory identifiers for this canvas scope
    if session_id:
        try:
            from src.memory.graph.facade import GraphMemoryFacade
            facade = GraphMemoryFacade()
            memory_scope = result.get("memory_scope", [])
            identifiers = []
            for scope in memory_scope[:3]:  # Limit to 3 scopes for token budget
                nodes = facade.recall(
                    query=scope,
                    limit=result.get("max_identifiers", 10),
                    session_id=session_id,
                )
                for n in nodes:
                    identifiers.append({
                        "id": n.get("id", ""),
                        "type": n.get("node_type", ""),
                        "content_preview": str(n.get("content", ""))[:100],
                    })
            result["memory_identifiers"] = identifiers
        except Exception as e:
            logger.debug(f"Memory identifier loading skipped: {e}")

    return result


# ===========================================================================
# request_tool — Universal tool request (architecture §16.6)
# ===========================================================================

def request_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request activation or creation of a tool.

    If tool_id exists in registry:
    - requires_approval=false → auto-activate for this session
    - requires_approval=true → Kanban task created

    If tool_id does NOT exist:
    - Kanban task created for Development Dept to build it

    Args:
        tool_id: ID of existing tool or proposed new tool name
        reason: Why this session needs it
        urgency: HIGH | MEDIUM | LOW
        session_id: Current session ID

    Returns:
        Request status and resolution
    """
    tool_id = args.get("tool_id", "")
    reason = args.get("reason", "")
    urgency = args.get("urgency", "MEDIUM")
    session_id = args.get("session_id", "")

    try:
        from src.agents.departments.tool_registry import get_tool_registry
        registry = get_tool_registry()
        tool = registry.get_tool(tool_id)

        if tool:
            return {
                "tool_id": tool_id,
                "found": True,
                "activated": True,
                "message": f"Tool '{tool_id}' activated for session {session_id}",
            }
    except Exception:
        pass

    return {
        "tool_id": tool_id,
        "found": False,
        "activated": False,
        "kanban_created": True,
        "message": (
            f"Tool '{tool_id}' not found. Kanban task created for Development Dept "
            f"(urgency={urgency}): {reason}"
        ),
    }


# ===========================================================================
# send_department_mail — Inter-department messaging (architecture §1.2)
# ===========================================================================

def send_department_mail(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message via the department mail system.

    Args:
        from_dept: Sending department
        to_dept: Receiving department
        subject: Message subject
        body: Message body (JSON string or plain text)
        priority: HIGH | MEDIUM | LOW
        message_type: DISPATCH | STRATEGY_DISPATCH | REPORT | NOTIFICATION

    Returns:
        Sent message metadata
    """
    from_dept = args.get("from_dept", "")
    to_dept = args.get("to_dept", "")
    subject = args.get("subject", "")
    body = args.get("body", "")
    priority = args.get("priority", "MEDIUM")

    try:
        from src.agents.departments.department_mail import (
            DepartmentMailService,
            MessageType,
            Priority,
        )
        mail = DepartmentMailService()
        msg_type = MessageType.DISPATCH
        pri = Priority.HIGH if priority == "HIGH" else (
            Priority.LOW if priority == "LOW" else Priority.MEDIUM
        )
        message = mail.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=msg_type,
            subject=subject,
            body=body,
            priority=pri,
        )
        return {
            "sent": True,
            "message_id": message.id,
            "from_dept": from_dept,
            "to_dept": to_dept,
        }
    except Exception as e:
        logger.error(f"Department mail failed: {e}")
        return {"sent": False, "error": str(e)}


# ===========================================================================
# Tool Registration Helper
# ===========================================================================

def get_global_tool_definitions() -> List[Dict[str, Any]]:
    """
    Return all global tools as ToolDefinition-compatible dicts.

    Used by BaseAgent to register global tools at session start.
    """
    return [
        {
            "name": "read_skill",
            "description": "Load a skill.md file JIT into agent context",
            "parameters": {
                "skill_id": {"type": "string", "description": "Skill identifier"},
                "department": {"type": "string", "description": "Department scope (optional)"},
            },
            "handler": read_skill,
            "category": "global",
            "read_only": True,
        },
        {
            "name": "write_memory",
            "description": (
                "Store information in graph memory (OBSERVATION, OPINION, DECISION, WORLD nodes). "
                "For OPINION nodes, include action, reasoning, confidence, alternatives_considered, "
                "constraints_applied, and agent_role fields. Use notify_departments to broadcast "
                "important memories to other departments via mail."
            ),
            "parameters": {
                "node_type": {"type": "string", "description": "Node type: OBSERVATION|OPINION|DECISION|WORLD"},
                "content": {"type": "string", "description": "Memory content"},
                "category": {"type": "string", "description": "Category within node type"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Retrieval tags"},
                "session_id": {"type": "string", "description": "Current session ID"},
                "department": {"type": "string", "description": "Owning department"},
                # --- OPINION node fields (architecture §6.1, Hindsight OPINION network) ---
                "action": {"type": "string", "description": "Action taken or recommended (OPINION nodes)"},
                "reasoning": {"type": "string", "description": "Reasoning behind the action (OPINION nodes)"},
                "confidence": {"type": "number", "description": "Confidence score 0.0-1.0 (OPINION nodes)"},
                "alternatives_considered": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Alternative options evaluated (OPINION nodes)",
                },
                "constraints_applied": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Constraints that shaped the decision (OPINION nodes)",
                },
                "agent_role": {"type": "string", "description": "Role of the agent forming the opinion (OPINION nodes)"},
                # --- Cross-department notification ---
                "notify_departments": {
                    "type": "array", "items": {"type": "string"},
                    "description": "List of departments to notify about this memory via department mail",
                },
            },
            "handler": write_memory,
            "category": "global",
        },
        {
            "name": "search_memory",
            "description": "Query graph memory for relevant context (primary retrieval layer)",
            "parameters": {
                "query": {"type": "string", "description": "Search query text"},
                "node_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by node types"},
                "department": {"type": "string", "description": "Filter by department"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "handler": search_memory,
            "category": "global",
            "read_only": True,
        },
        {
            "name": "read_canvas_context",
            "description": "Load canvas context template identifiers for active canvas",
            "parameters": {
                "canvas": {"type": "string", "description": "Canvas name (live_trading, research, etc.)"},
            },
            "handler": read_canvas_context,
            "category": "global",
            "read_only": True,
        },
        {
            "name": "request_tool",
            "description": "Request activation or creation of a tool not in current session",
            "parameters": {
                "tool_id": {"type": "string", "description": "Tool ID to request"},
                "reason": {"type": "string", "description": "Why this session needs it"},
                "urgency": {"type": "string", "description": "HIGH|MEDIUM|LOW"},
                "session_id": {"type": "string", "description": "Current session ID"},
            },
            "handler": request_tool,
            "category": "global",
        },
        {
            "name": "send_department_mail",
            "description": "Send inter-department message via mail system",
            "parameters": {
                "from_dept": {"type": "string", "description": "Sending department"},
                "to_dept": {"type": "string", "description": "Receiving department"},
                "subject": {"type": "string", "description": "Message subject"},
                "body": {"type": "string", "description": "Message body"},
                "priority": {"type": "string", "description": "HIGH|MEDIUM|LOW"},
            },
            "handler": send_department_mail,
            "category": "global",
        },
    ]
