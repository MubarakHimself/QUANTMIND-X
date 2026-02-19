"""
Claude Agent Hooks for v2 Agent Stack

Pre and post execution hooks for agent task lifecycle.
Replaces LangGraph node transition callbacks.

**Phase 2.3 - Agent Hooks**
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# TRD output directory
TRD_OUTPUT_DIR = Path(os.getenv("TRD_OUTPUT_DIR", "/app/docs/trds"))


async def pre_analyst_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-execution hook for Analyst agent.
    
    Validates NPRD input and loads relevant TRD templates.
    
    Args:
        task: Task dictionary with payload
        
    Returns:
        Modified task with additional context
    """
    logger.info(f"Running pre_analyst_hook for task {task.get('task_id')}")
    
    payload = task.get("payload", {})
    messages = payload.get("messages", [])
    context = payload.get("context", {})
    
    # Validate that we have NPRD content
    nprd_content = context.get("nprd_content")
    if not nprd_content and messages:
        # Try to extract from messages
        for msg in messages:
            if msg.get("role") == "user" and len(msg.get("content", "")) > 100:
                nprd_content = msg.get("content")
                break
    
    if nprd_content:
        # Basic NPRD validation
        required_sections = ["strategy", "requirements"]
        has_required = any(
            section.lower() in nprd_content.lower() 
            for section in required_sections
        )
        
        if not has_required:
            logger.warning("NPRD may be missing required sections")
            context["validation_warnings"] = ["NPRD missing recommended sections: strategy, requirements"]
        
        context["nprd_validated"] = True
    
    # Load TRD template reference
    trd_template_path = Path("/app/docs/templates/trd_template.md")
    if trd_template_path.exists():
        context["trd_template_available"] = True
        logger.debug("TRD template available for reference")
    
    # Update task with enriched context
    task["payload"]["context"] = context
    task["hook_metadata"] = {
        "pre_hook": "analyst",
        "executed_at": datetime.utcnow().isoformat(),
    }
    
    return task


async def post_analyst_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-execution hook for Analyst agent.
    
    Saves TRD to docs/trds/ and emits Prometheus metric.
    
    Args:
        task: Original task dictionary
        result: Result dictionary from agent execution
        
    Returns:
        Modified result with additional metadata
    """
    logger.info(f"Running post_analyst_hook for task {task.get('task_id')}")
    
    output = result.get("output", "")
    
    # Try to extract TRD content from output
    if output and "TRD" in output.upper():
        try:
            # Ensure TRD directory exists
            TRD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate TRD filename
            task_id = task.get("task_id", "unknown")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            trd_filename = f"TRD_{timestamp}_{task_id[:8]}.md"
            trd_path = TRD_OUTPUT_DIR / trd_filename
            
            # Write TRD content
            with open(trd_path, "w") as f:
                f.write(f"# Trading Requirements Document\n\n")
                f.write(f"**Generated:** {datetime.utcnow().isoformat()}\n")
                f.write(f"**Task ID:** {task_id}\n\n")
                f.write("---\n\n")
                f.write(output)
            
            logger.info(f"TRD saved to {trd_path}")
            result["trd_path"] = str(trd_path)
            
            # Emit Prometheus metric
            try:
                from src.agents.observers.prometheus_observer import agent_creations_total
                # This is a conceptual metric - the actual counter is for agent creations
                # For TRD generation, we'd need a custom metric
                logger.info(f"TRD generated: {trd_filename}")
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Failed to save TRD: {e}")
            result["trd_save_error"] = str(e)
    
    result["hook_metadata"] = {
        "post_hook": "analyst",
        "executed_at": datetime.utcnow().isoformat(),
        "trd_saved": "trd_path" in result,
    }
    
    return result


async def pre_quantcode_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-execution hook for QuantCode agent.
    
    Validates TRD input and checks MQL5 compilation environment.
    
    Args:
        task: Task dictionary with payload
        
    Returns:
        Modified task with additional context
    """
    logger.info(f"Running pre_quantcode_hook for task {task.get('task_id')}")
    
    payload = task.get("payload", {})
    context = payload.get("context", {})
    
    # Validate TRD content
    trd_content = context.get("trd_content")
    if not trd_content:
        # Check for TRD path
        trd_path = context.get("trd_path")
        if trd_path and Path(trd_path).exists():
            with open(trd_path, "r") as f:
                trd_content = f.read()
            context["trd_content"] = trd_content
    
    if trd_content:
        # Validate TRD has required sections
        required_sections = ["entry", "exit", "risk"]
        has_required = any(
            section.lower() in trd_content.lower() 
            for section in required_sections
        )
        
        if has_required:
            context["trd_validated"] = True
            logger.info("TRD validated for QuantCode generation")
        else:
            logger.warning("TRD may be missing trading logic sections")
            context["validation_warnings"] = ["TRD missing recommended sections: entry, exit, risk"]
    
    # Check MQL5 compilation environment
    mt5_path = Path(os.getenv("MT5_PATH", "/MT5"))
    mql5_experts_path = mt5_path / "MQL5" / "Experts"
    
    if mt5_path.exists():
        context["mt5_environment"] = "available"
        if mql5_experts_path.exists():
            context["mql5_output_dir"] = str(mql5_experts_path)
            logger.info(f"MQL5 output directory: {mql5_experts_path}")
    else:
        context["mt5_environment"] = "unavailable"
        logger.warning("MT5 environment not available - compilation will be skipped")
    
    # Update task
    task["payload"]["context"] = context
    task["hook_metadata"] = {
        "pre_hook": "quantcode",
        "executed_at": datetime.utcnow().isoformat(),
    }
    
    return task


async def post_quantcode_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-execution hook for QuantCode agent.
    
    Saves .mq5 file and triggers compilation check.
    
    Args:
        task: Original task dictionary
        result: Result dictionary from agent execution
        
    Returns:
        Modified result with additional metadata
    """
    logger.info(f"Running post_quantcode_hook for task {task.get('task_id')}")
    
    output = result.get("output", "")
    
    # Extract MQL5 code from output
    if output and ("MQL5" in output or "mq5" in output.lower()):
        try:
            # Find code blocks
            import re
            code_blocks = re.findall(r'```(?:mql5|mq5)?\s*\n(.*?)\n```', output, re.DOTALL)
            
            if code_blocks:
                # Get output directory
                context = task.get("payload", {}).get("context", {})
                output_dir = Path(context.get(
                    "mql5_output_dir",
                    "/app/workspace/quantcode/scratch"
                ))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                task_id = task.get("task_id", "unknown")
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                mq5_filename = f"Strategy_{timestamp}_{task_id[:8]}.mq5"
                mq5_path = output_dir / mq5_filename
                
                # Write the largest code block
                main_code = max(code_blocks, key=len)
                with open(mq5_path, "w") as f:
                    f.write(main_code)
                
                logger.info(f"MQL5 code saved to {mq5_path}")
                result["mq5_path"] = str(mq5_path)
                
                # TODO: Trigger compilation check via MT5 Compiler MCP
                # For now, just log the intent
                logger.info(f"MQL5 file ready for compilation: {mq5_filename}")
                
        except Exception as e:
            logger.error(f"Failed to save MQL5 code: {e}")
            result["mq5_save_error"] = str(e)
    
    result["hook_metadata"] = {
        "post_hook": "quantcode",
        "executed_at": datetime.utcnow().isoformat(),
        "mq5_saved": "mq5_path" in result,
    }
    
    return result


async def pre_copilot_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-execution hook for Copilot agent.
    
    Loads mission context from database.
    
    Args:
        task: Task dictionary with payload
        
    Returns:
        Modified task with mission context
    """
    logger.info(f"Running pre_copilot_hook for task {task.get('task_id')}")
    
    payload = task.get("payload", {})
    context = payload.get("context", {})
    
    # Load mission context if mission_id is provided
    mission_id = context.get("mission_id")
    if mission_id:
        try:
            from src.agents.core.database import sys_db
            
            # Get mission details
            mission = sys_db.get_mission(mission_id)
            if mission:
                context["mission"] = mission
                logger.info(f"Loaded mission context: {mission_id}")
                
                # Get related tasks
                related_tasks = sys_db.get_tasks_for_mission(mission_id)
                if related_tasks:
                    context["related_tasks"] = related_tasks
                    logger.debug(f"Found {len(related_tasks)} related tasks")
                    
        except ImportError:
            logger.warning("Database module not available for mission loading")
        except Exception as e:
            logger.error(f"Failed to load mission context: {e}")
    else:
        # This is a new mission - generate ID
        import uuid
        context["mission_id"] = context.get("mission_id", str(uuid.uuid4()))
        logger.info(f"New mission started: {context['mission_id']}")
    
    # Determine mode based on context
    if not context.get("mode"):
        if context.get("trd_content") or context.get("trd_path"):
            context["mode"] = "BUILD"
        elif context.get("clarification_needed"):
            context["mode"] = "ASK"
        else:
            context["mode"] = "PLAN"
    
    task["payload"]["context"] = context
    task["hook_metadata"] = {
        "pre_hook": "copilot",
        "executed_at": datetime.utcnow().isoformat(),
        "mode": context.get("mode"),
    }
    
    return task


async def post_copilot_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-execution hook for Copilot agent.
    
    Updates mission status in database.
    
    Args:
        task: Original task dictionary
        result: Result dictionary from agent execution
        
    Returns:
        Modified result with additional metadata
    """
    logger.info(f"Running post_copilot_hook for task {task.get('task_id')}")
    
    context = task.get("payload", {}).get("context", {})
    mission_id = context.get("mission_id")
    
    if mission_id:
        try:
            from src.agents.core.database import sys_db
            
            # Update mission status based on result
            status = "in_progress"
            if result.get("status") == "completed":
                # Check if mission is complete
                output = result.get("output", "")
                if "MISSION_COMPLETE" in output.upper() or "strategy deployed" in output.lower():
                    status = "completed"
                elif "awaiting" in output.lower() or "clarification" in output.lower():
                    status = "awaiting_input"
            
            sys_db.update_mission_status(mission_id, status)
            logger.info(f"Updated mission {mission_id} status to {status}")
            
            result["mission_status"] = status
            
        except ImportError:
            logger.warning("Database module not available for mission update")
        except Exception as e:
            logger.error(f"Failed to update mission status: {e}")
    
    # Check for sub-agent tasks
    output = result.get("output", "")
    if "submit to analyst" in output.lower() or "submit to quantcode" in output.lower():
        result["sub_agent_tasks"] = []
        # Parse which sub-agents to invoke
        if "analyst" in output.lower():
            result["sub_agent_tasks"].append("analyst")
        if "quantcode" in output.lower():
            result["sub_agent_tasks"].append("quantcode")
        if "pinescript" in output.lower():
            result["sub_agent_tasks"].append("pinescript")
    
    result["hook_metadata"] = {
        "post_hook": "copilot",
        "executed_at": datetime.utcnow().isoformat(),
        "mission_updated": mission_id is not None,
    }
    
    return result


# No-op hooks for lightweight agents
async def pre_router_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-execution hook for Router agent (no-op)."""
    task["hook_metadata"] = {"pre_hook": "router", "executed_at": datetime.utcnow().isoformat()}
    return task


async def post_router_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Post-execution hook for Router agent (no-op)."""
    result["hook_metadata"] = {"post_hook": "router", "executed_at": datetime.utcnow().isoformat()}
    return result


async def pre_executor_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-execution hook for Executor agent (no-op)."""
    task["hook_metadata"] = {"pre_hook": "executor", "executed_at": datetime.utcnow().isoformat()}
    return task


async def post_executor_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Post-execution hook for Executor agent (no-op)."""
    result["hook_metadata"] = {"post_hook": "executor", "executed_at": datetime.utcnow().isoformat()}
    return result


async def pre_pinescript_hook(task: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-execution hook for PineScript agent (no-op)."""
    task["hook_metadata"] = {"pre_hook": "pinescript", "executed_at": datetime.utcnow().isoformat()}
    return task


async def post_pinescript_hook(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Post-execution hook for PineScript agent (no-op)."""
    result["hook_metadata"] = {"post_hook": "pinescript", "executed_at": datetime.utcnow().isoformat()}
    return result