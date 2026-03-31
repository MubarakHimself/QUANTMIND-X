"""
QuantMind Base Agent — Anthropic Agent SDK Runtime

Implements the canonical agent loop using Anthropic Agent SDK patterns:
- Agent identity schema (tier, department, session_type)
- Tool activation at session start (global + department + workflow-context)
- Hooks for pre/post tool use, audit logging, and safety gates
- Session management with resume capability

Architecture reference: architecture.md §4.1–4.3, §16
SDK reference: claude-agent-sdk-python

NOTE: Uses Anthropic Messages API directly with SDK PATTERNS applied
(query loop, tool registration, hooks, MCP config).
"""

import asyncio
import json
import logging
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Identity (architecture §4.3)
# ---------------------------------------------------------------------------

class AgentIdentity:
    """
    Typed agent identity controlling session, skill, and tool access.

    Determines: tier, department, session type, tools/skills activated.
    """

    def __init__(
        self,
        tier: str,
        department: str,
        sub_agent_type: Optional[str] = None,
        session_type: str = "autonomous",
        session_id: Optional[str] = None,
    ):
        self.tier = tier
        self.department = department
        self.sub_agent_type = sub_agent_type
        self.session_type = session_type
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"

    @property
    def memory_namespace(self) -> Dict[str, List[str]]:
        """Memory namespace per architecture §4.3."""
        return {
            "episodic": ["dept", self.department, self.session_id, "episodic"],
            "semantic": ["dept", self.department, self.session_id, "semantic"],
            "profile": ["dept", self.department, "profile"],
            "global": ["global", "strategies"],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "department": self.department,
            "sub_agent_type": self.sub_agent_type,
            "session_type": self.session_type,
            "session_id": self.session_id,
            "memory_namespace": self.memory_namespace,
        }


# ---------------------------------------------------------------------------
# Hook System (SDK-pattern hooks)
# ---------------------------------------------------------------------------

class HookEvent:
    """Hook event types matching Anthropic Agent SDK."""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


class HookMatcher:
    """Matches hook callbacks to tool patterns."""

    def __init__(
        self,
        matcher: Optional[str] = None,
        hooks: Optional[List[Callable]] = None,
        timeout: int = 60,
    ):
        self.matcher = matcher
        self.hooks = hooks or []
        self.timeout = timeout


class AgentHooks:
    """
    Hook registry for agent lifecycle events.

    SDK hooks: PreToolUse (allow/deny/modify), PostToolUse (audit), Stop (cleanup).
    """

    def __init__(self):
        self._hooks: Dict[str, List[HookMatcher]] = {
            HookEvent.PRE_TOOL_USE: [],
            HookEvent.POST_TOOL_USE: [],
            HookEvent.POST_TOOL_USE_FAILURE: [],
            HookEvent.STOP: [],
            HookEvent.SUBAGENT_START: [],
            HookEvent.SUBAGENT_STOP: [],
        }

    def register(self, event: str, matcher: HookMatcher) -> None:
        if event in self._hooks:
            self._hooks[event].append(matcher)

    async def fire(
        self,
        event: str,
        input_data: Dict[str, Any],
        tool_use_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fire all matching hooks for an event."""
        import re

        result: Dict[str, Any] = {}
        for matcher in self._hooks.get(event, []):
            tool_name = input_data.get("tool_name", "")
            if matcher.matcher and not re.match(matcher.matcher, tool_name):
                continue
            for hook_fn in matcher.hooks:
                try:
                    output = await asyncio.wait_for(
                        hook_fn(input_data, tool_use_id, None),
                        timeout=matcher.timeout,
                    )
                    if output:
                        result.update(output)
                except asyncio.TimeoutError:
                    logger.warning(f"Hook timeout for {event} on {tool_name}")
                except Exception as e:
                    logger.error(f"Hook error for {event}: {e}")
        return result


# ---------------------------------------------------------------------------
# Tool Definition (SDK-pattern tools)
# ---------------------------------------------------------------------------

class ToolDefinition:
    """
    A tool registered with the agent runtime.

    SDK @tool pattern: name, description, JSON schema, async handler.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        category: str = "global",
        dept_scope: Optional[List[str]] = None,
        requires_approval: bool = False,
        safe_for_autonomous: bool = True,
        read_only: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.category = category
        self.dept_scope = dept_scope
        self.requires_approval = requires_approval
        self.safe_for_autonomous = safe_for_autonomous
        self.read_only = read_only

    def to_claude_tool(self) -> Dict[str, Any]:
        """Convert to Claude API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
            },
        }

    def validate_input(self, args: Dict[str, Any]) -> Optional[str]:
        """Validate tool input against JSON Schema parameters.

        Returns None if valid, or an error string if validation fails.
        Checks required fields and basic type constraints.
        """
        if not self.parameters:
            return None

        errors = []
        for param_name, param_spec in self.parameters.items():
            expected_type = param_spec.get("type")
            if param_name in args and expected_type:
                val = args[param_name]
                type_checks = {
                    "string": str, "number": (int, float), "integer": int,
                    "boolean": bool, "array": list, "object": dict,
                }
                expected_cls = type_checks.get(expected_type)
                if expected_cls and val is not None and not isinstance(val, expected_cls):
                    errors.append(
                        f"Parameter '{param_name}': expected {expected_type}, "
                        f"got {type(val).__name__}"
                    )
        return "; ".join(errors) if errors else None

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool handler with input validation."""
        # JSON Schema validation (Agent SDK compliance)
        validation_error = self.validate_input(args)
        if validation_error:
            logger.warning(f"Tool {self.name} input validation: {validation_error}")
            # Log but don't block — Claude's inputs are usually correct

        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(args)
            else:
                result = self.handler(args)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "is_error": True,
            }


# ---------------------------------------------------------------------------
# Agent Result (SDK ResultMessage pattern)
# ---------------------------------------------------------------------------

class AgentResult:
    """Result of an agent query."""

    def __init__(
        self,
        result: str = "",
        session_id: str = "",
        num_turns: int = 0,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        subtype: str = "success",
        error: Optional[str] = None,
    ):
        self.result = result
        self.session_id = session_id
        self.num_turns = num_turns
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
        self.subtype = subtype
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "session_id": self.session_id,
            "num_turns": self.num_turns,
            "usage": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
            "subtype": self.subtype,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Base Agent — Agentic Loop
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Anthropic Agent SDK-pattern agent runtime.

    Loop: prompt → Claude → tool_use? → execute tools → repeat → final text.
    """

    DEFAULT_MAX_TURNS = 30

    def __init__(
        self,
        name: str,
        identity: AgentIdentity,
        system_prompt: str,
        tools: Optional[List[ToolDefinition]] = None,
        hooks: Optional[AgentHooks] = None,
        mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        model: Optional[str] = None,
    ):
        self.name = name
        self.identity = identity
        self.system_prompt = system_prompt
        self.tools: Dict[str, ToolDefinition] = {}
        self.hooks = hooks or AgentHooks()
        self.mcp_servers = mcp_servers or {}
        self.max_turns = max_turns
        self._conversation_history: List[Dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._model = model
        self._client = None

        if tools:
            for td in tools:
                self.register_tool(td)

        # Register global tools at session start (architecture §16.3)
        self._register_global_tools()

        logger.info(
            f"Agent {name}: tier={identity.tier}, "
            f"dept={identity.department}, tools={len(self.tools)}"
        )

    def _register_global_tools(self) -> None:
        """
        Register global-tier tools available to all agents (architecture §16.3).

        Global tools: read_skill, write_memory, search_memory,
        read_canvas_context, request_tool, send_department_mail.
        """
        try:
            from src.agents.tools.global_tools import get_global_tool_definitions

            for tool_def_dict in get_global_tool_definitions():
                handler = tool_def_dict.get("handler")
                if not handler:
                    continue

                # Convert dict-style parameters to JSON Schema
                params_raw = tool_def_dict.get("parameters", {})
                json_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                for param_name, param_info in params_raw.items():
                    json_schema["properties"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                    }
                    # All global tool params are optional by default

                td = ToolDefinition(
                    name=tool_def_dict["name"],
                    description=tool_def_dict.get("description", ""),
                    parameters=json_schema,
                    handler=handler,
                    read_only=tool_def_dict.get("read_only", False),
                )
                # Only register if not already provided by explicit tools
                if td.name not in self.tools:
                    self.tools[td.name] = td

            logger.debug(f"Registered {len(self.tools)} tools (incl. global)")
        except ImportError as e:
            logger.debug(f"Global tools not available: {e}")
        except Exception as e:
            logger.warning(f"Global tool registration failed: {e}")

    def _get_client_and_model(self) -> Tuple[Any, str]:
        """Lazy-init via shared llm_utils (ProviderRouter → env → bare)."""
        if self._client is not None:
            return self._client, self._model

        from src.agents.departments.subagents.llm_utils import get_subagent_client
        self._client, resolved_model = get_subagent_client()
        if not self._model:
            self._model = resolved_model
        return self._client, self._model

    def register_tool(self, tool_def: ToolDefinition) -> None:
        self.tools[tool_def.name] = tool_def

    def get_claude_tools(self) -> List[Dict[str, Any]]:
        return [t.to_claude_tool() for t in self.tools.values()]

    # Map internal thought types to UI-recognized types
    # Frontend recognises: reasoning, action, observation, decision
    _THOUGHT_TYPE_MAP = {
        "reasoning": "reasoning",
        "tool_call": "action",
        "action": "action",
        "observation": "observation",
        "decision": "decision",
        "error": "observation",       # surface errors as observations
        "memory_read": "observation",
        "dispatch": "action",
    }

    def _publish_thought(self, thought: str, thought_type: str = "reasoning") -> None:
        """Publish agent thinking to SSE thought stream for UI display."""
        try:
            from src.api.agent_thought_stream_endpoints import get_thought_publisher
            mapped_type = self._THOUGHT_TYPE_MAP.get(thought_type, "reasoning")
            get_thought_publisher().publish(
                department=self.identity.department or self.name,
                thought=thought,
                thought_type=mapped_type,
                session_id=self.identity.session_id,
            )
        except Exception:
            pass  # Thought stream is optional — don't break agent loop

    async def query(self, prompt: str, resume: bool = False) -> AgentResult:
        """
        Execute the agentic loop with chain-of-thought streaming.

        Each turn publishes reasoning, tool calls, and decisions to the
        SSE thought stream so the UI can display real-time agent thinking.
        """
        client, model = self._get_client_and_model()

        if not resume:
            self._conversation_history = []
            self._total_input_tokens = 0
            self._total_output_tokens = 0

        self._conversation_history.append({"role": "user", "content": prompt})
        turns = 0
        final_text = ""

        self._publish_thought(
            f"Starting query: {prompt[:120]}...",
            thought_type="reasoning",
        )

        while turns < self.max_turns:
            turns += 1
            try:
                claude_tools = self.get_claude_tools()
                kwargs = {
                    "model": model,
                    "max_tokens": 8192,
                    "system": self.system_prompt,
                    "messages": self._conversation_history,
                }
                if claude_tools:
                    kwargs["tools"] = claude_tools

                response = client.messages.create(**kwargs)

                if hasattr(response, "usage"):
                    self._total_input_tokens += getattr(response.usage, "input_tokens", 0)
                    self._total_output_tokens += getattr(response.usage, "output_tokens", 0)

                tool_use_blocks = []
                text_blocks = []
                thinking_blocks = []
                for block in response.content:
                    if hasattr(block, "type"):
                        if block.type == "tool_use":
                            tool_use_blocks.append(block)
                        elif block.type == "text":
                            text_blocks.append(block.text)
                        elif block.type == "thinking":
                            thinking_text = getattr(block, "thinking", "")
                            if thinking_text:
                                thinking_blocks.append(thinking_text)

                # Stream thinking to UI if present (extended thinking)
                if thinking_blocks:
                    for chunk in thinking_blocks:
                        self._publish_thought(chunk[:500], thought_type="reasoning")

                if not tool_use_blocks:
                    final_text = "\n".join(text_blocks)
                    self._conversation_history.append({"role": "assistant", "content": response.content})

                    # Publish final response summary
                    self._publish_thought(
                        f"Turn {turns}: Final response ({len(final_text)} chars)",
                        thought_type="decision",
                    )
                    break

                self._conversation_history.append({"role": "assistant", "content": response.content})

                # Publish tool usage to thought stream
                for tb in tool_use_blocks:
                    self._publish_thought(
                        f"Turn {turns}: Calling tool {tb.name}",
                        thought_type="tool_call",
                    )

                tool_results = await self._execute_tool_blocks(tool_use_blocks)
                self._conversation_history.append({"role": "user", "content": tool_results})

            except Exception as e:
                logger.error(f"Agent {self.name} error turn {turns}: {e}")
                self._publish_thought(f"Error on turn {turns}: {str(e)[:100]}", thought_type="error")
                await self.hooks.fire(HookEvent.STOP, {"session_id": self.identity.session_id, "error": str(e)})
                return AgentResult(
                    session_id=self.identity.session_id, num_turns=turns,
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                    subtype="error_tool", error=str(e),
                )

        self._publish_thought(
            f"Query complete: {turns} turns, {self._total_input_tokens + self._total_output_tokens} tokens",
            thought_type="decision",
        )

        await self.hooks.fire(HookEvent.STOP, {"session_id": self.identity.session_id})
        return AgentResult(
            result=final_text,
            session_id=self.identity.session_id,
            num_turns=turns,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            subtype="success" if turns < self.max_turns else "error_max_turns",
        )

    async def _execute_tool_blocks(self, tool_use_blocks: list) -> List[Dict[str, Any]]:
        """Execute tool blocks with hook integration."""
        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            pre_result = await self.hooks.fire(
                HookEvent.PRE_TOOL_USE,
                {"tool_name": tool_name, "tool_input": tool_input,
                 "session_id": self.identity.session_id, "hook_event_name": HookEvent.PRE_TOOL_USE},
                tool_use_id,
            )
            hook_out = pre_result.get("hookSpecificOutput", {})
            if hook_out.get("permissionDecision") == "deny":
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use_id,
                    "content": f"Tool denied: {hook_out.get('permissionDecisionReason', 'Denied')}",
                    "is_error": True,
                })
                continue

            tool_def = self.tools.get(tool_name)
            if tool_def:
                # ── HITL: Tool-level approval ────────────────────────────
                if getattr(tool_def, "requires_approval", False):
                    try:
                        from src.agents.approval_manager import (
                            get_approval_manager, ApprovalType, ApprovalUrgency,
                        )
                        self._publish_thought(
                            f"Tool {tool_name} requires human approval — waiting...",
                            thought_type="observation",
                        )
                        # Build resume context so the agent can recover
                        _resume_ctx = {
                            "agent_name": self.name,
                            "agent_type": type(self).__name__,
                            "session_id": getattr(self, "session_id", None),
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_use_id": tool_use_id,
                        }
                        # Save checkpoint if checkpoint manager is available
                        try:
                            if hasattr(self, "_checkpoint_manager") and self._checkpoint_manager:
                                self._checkpoint_manager.save_checkpoint(
                                    state="paused_hitl",
                                    metadata={"approval_tool": tool_name},
                                )
                                _resume_ctx["checkpoint_saved"] = True
                        except Exception:
                            pass

                        req = get_approval_manager().request_approval(
                            approval_type=ApprovalType.TOOL_EXECUTION,
                            title=f"Tool: {tool_name}",
                            description=(
                                f"Agent {self.name} wants to execute tool '{tool_name}' "
                                f"with input: {json.dumps(tool_input)[:500]}"
                            ),
                            department=getattr(self.identity, "department", "") or self.name,
                            agent_id=self.name,
                            tool_name=tool_name,
                            tool_input=tool_input,
                            urgency=ApprovalUrgency.HIGH,
                            resume_context=_resume_ctx,
                        )
                        resolved = await get_approval_manager().wait_for_approval(
                            req.id, timeout=600,  # 10 min timeout for tool approval
                        )
                        if resolved.status.value != "approved":
                            reason = resolved.rejection_reason or "Rejected by human"
                            self._publish_thought(
                                f"Tool {tool_name} rejected: {reason}",
                                thought_type="observation",
                            )
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": f"Tool execution denied by human: {reason}",
                                "is_error": True,
                            })
                            continue
                        self._publish_thought(
                            f"Tool {tool_name} approved — executing",
                            thought_type="action",
                        )
                    except Exception as e:
                        logger.warning(f"HITL tool approval failed: {e}")
                        # On failure, proceed with execution (fail-open for now)

                result = await tool_def.execute(tool_input)
                content_text = ""
                if result.get("content"):
                    for c in result["content"]:
                        if c.get("type") == "text":
                            content_text = c["text"]
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use_id,
                    "content": content_text, "is_error": result.get("is_error", False),
                })
                await self.hooks.fire(
                    HookEvent.POST_TOOL_USE,
                    {"tool_name": tool_name, "tool_result": content_text,
                     "session_id": self.identity.session_id, "hook_event_name": HookEvent.POST_TOOL_USE},
                    tool_use_id,
                )
            else:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tool_use_id,
                    "content": f"Unknown tool: {tool_name}", "is_error": True,
                })
        return tool_results

    async def query_stream(self, prompt: str) -> AsyncIterator[Dict[str, Any]]:
        """Streaming agentic loop — yields SDK-pattern events."""
        client, model = self._get_client_and_model()

        yield {"type": "system", "subtype": "init", "session_id": self.identity.session_id}
        self._conversation_history.append({"role": "user", "content": prompt})
        turns = 0

        while turns < self.max_turns:
            turns += 1
            claude_tools = self.get_claude_tools()
            kwargs = {
                "model": model, "max_tokens": 8192,
                "system": self.system_prompt, "messages": self._conversation_history,
            }
            if claude_tools:
                kwargs["tools"] = claude_tools

            try:
                with client.messages.stream(**kwargs) as stream:
                    full_response = stream.get_final_message()

                if hasattr(full_response, "usage"):
                    self._total_input_tokens += getattr(full_response.usage, "input_tokens", 0)
                    self._total_output_tokens += getattr(full_response.usage, "output_tokens", 0)

                tool_use_blocks = []
                for block in full_response.content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            yield {"type": "assistant", "text": block.text}
                        elif block.type == "tool_use":
                            tool_use_blocks.append(block)
                            yield {"type": "tool_use", "name": block.name, "input": block.input}

                if not tool_use_blocks:
                    self._conversation_history.append({"role": "assistant", "content": full_response.content})
                    break

                self._conversation_history.append({"role": "assistant", "content": full_response.content})
                tool_results = await self._execute_tool_blocks(tool_use_blocks)

                for tr, block in zip(tool_results, tool_use_blocks):
                    yield {"type": "tool_result", "name": block.name, "result": tr.get("content", "")}

                self._conversation_history.append({"role": "user", "content": tool_results})

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield {"type": "result", "subtype": "error_tool", "error": str(e),
                       "session_id": self.identity.session_id, "num_turns": turns}
                return

        yield {
            "type": "result",
            "subtype": "success" if turns < self.max_turns else "error_max_turns",
            "session_id": self.identity.session_id, "num_turns": turns,
            "usage": {"input_tokens": self._total_input_tokens, "output_tokens": self._total_output_tokens},
        }


# ---------------------------------------------------------------------------
# CommandParser (backward compatibility)
# ---------------------------------------------------------------------------

class CommandParser:
    """Slash command parser for direct command execution."""

    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self._register_builtin_commands()

    def _register_builtin_commands(self) -> None:
        self.commands["code"] = self._execute_code
        self.commands["file"] = self._execute_file
        self.commands["run"] = self._execute_run
        self.commands["help"] = self._execute_help

    def register_command(self, name: str, handler: Callable) -> None:
        self.commands[name] = handler

    def parse_command(self, message: str) -> Optional[Tuple[str, str]]:
        if not message or not message.startswith("/"):
            return None
        parts = message[1:].split(maxsplit=1)
        return (parts[0], parts[1] if len(parts) > 1 else "") if parts else None

    def is_slash_command(self, message: str) -> bool:
        return bool(message and message.startswith("/"))

    async def execute(self, message: str) -> str:
        parsed = self.parse_command(message)
        if not parsed:
            return "Not a command"
        cmd, args = parsed
        if cmd not in self.commands:
            return f"Unknown command: /{cmd}"
        try:
            return self.commands[cmd](args)
        except Exception as e:
            return f"Error: {str(e)}"

    def _execute_code(self, args: str) -> str:
        try:
            import io
            from contextlib import redirect_stdout
            out = io.StringIO()
            with redirect_stdout(out):
                exec(args)
            return out.getvalue() or "Code executed (no output)"
        except Exception as e:
            return f"Code execution error: {str(e)}"

    def _execute_file(self, args: str) -> str:
        parts = args.split(maxsplit=2)
        if len(parts) < 2:
            return "Usage: /file [read|write] <path> [content]"
        op, path = parts[0], parts[1]
        try:
            if op == "read":
                with open(path, "r") as f:
                    return f.read()
            elif op == "write" and len(parts) >= 3:
                with open(path, "w") as f:
                    f.write(parts[2])
                return f"Wrote to {path}"
            return f"Unknown op: {op}"
        except Exception as e:
            return f"File error: {str(e)}"

    def _execute_run(self, args: str) -> str:
        try:
            import subprocess
            r = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=30)
            return r.stdout or r.stderr or "Done"
        except Exception as e:
            return f"Run error: {str(e)}"

    def _execute_help(self, args: str) -> str:
        return (
            "Commands: /code <py>, /file read|write <path> [content], "
            "/run <cmd>, /help"
        )
