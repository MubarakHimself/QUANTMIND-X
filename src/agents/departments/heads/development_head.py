"""
Development Department Head

Responsible for:
- Building and maintaining trading bots (Python, PineScript, MQL5)
- EA lifecycle management (create, test, deploy)
- Strategy implementation and optimization

Workers:
- python_dev: Python-based strategy implementation
- pinescript_dev: TradingView PineScript development
- mql5_dev: MetaTrader 5 EA development
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config, SubAgentType
from src.agents.departments.tool_registry import ToolRegistry
from src.agents.departments.tool_access import ToolPermission
from src.trd.schema import TRDDocument
from src.trd.parser import TRDParser
from src.trd.validator import TRDValidator, ValidationResult
from src.mql5.generator import MQL5Generator
from src.mql5.compiler.service import get_compilation_service, COMPILE_STATUS_PENDING, CompilationServiceResult
from src.strategy.output import EAOutputStorage

logger = logging.getLogger(__name__)


# Task input schema
@dataclass
class DevelopmentTask:
    """Development task input."""
    task_type: str  # "generate_ea", "parse_trd", "validate_trd"
    trd_data: Optional[Dict[str, Any]] = None
    strategy_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class EAGenerationResult:
    """Result of EA generation."""
    success: bool
    strategy_id: str
    version: int
    file_path: str
    validation_result: Optional[ValidationResult] = None
    clarification_needed: bool = False
    clarification_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DevelopmentHead(DepartmentHead):
    """Development Department Head for EA/Bot building."""

    # Ambiguity threshold for requiring FloorManager clarification
    CLARIFICATION_SEVERITY_THRESHOLD = "high"

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.DEVELOPMENT)
        super().__init__(config=config, mail_db_path=mail_db_path)

        # Initialize TRD components
        self.parser = TRDParser()
        self.validator = TRDValidator()
        self.mql5_generator = MQL5Generator()
        self.ea_storage = EAOutputStorage()
        self.compilation_service = get_compilation_service()

        # Track current session
        self._current_session_id: Optional[str] = None

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "create_ea",
                "description": "Create new Expert Advisor (EA)",
                "parameters": {
                    "name": "EA name",
                    "strategy_type": "Strategy type (trend, range, breakout)",
                    "language": "mql5, pinescript, or python",
                },
            },
            {
                "name": "write_code",
                "description": "Write trading bot code",
                "parameters": {
                    "file_type": "File type (.mq5, .py, .pine)",
                    "content": "Trading logic code",
                },
            },
            {
                "name": "test_ea",
                "description": "Test EA on historical data",
                "parameters": {
                    "ea_name": "EA to test",
                    "symbol": "Trading symbol",
                    "timeframe": "Timeframe (M1, M5, H1, etc.)",
                },
            },
            {
                "name": "deploy_ea",
                "description": "Deploy EA to paper trading",
                "parameters": {
                    "ea_name": "EA to deploy",
                    "symbol": "Trading symbol",
                    "parameters": "EA parameters",
                },
            },
            {
                "name": "generate_mql5_ea",
                "description": "Generate MQL5 EA from TRD document",
                "parameters": {
                    "trd_data": "TRD document data (JSON)",
                },
            },
            {
                "name": "validate_trd",
                "description": "Validate TRD document for completeness",
                "parameters": {
                    "trd_data": "TRD document data (JSON)",
                },
            },
        ]

    def get_tool_instances(self) -> Dict[str, Any]:
        """Get actual tool instances for Development department."""
        tools = {}
        dept = Department.DEVELOPMENT

        # Development gets full access to coding tools
        tool_names = [
            "mql5_tools",
            "pinescript_tools",
            "backtest_tools",
            "ea_lifecycle",
            "memory_tools",
            "knowledge_tools",
        ]

        for tool_name in tool_names:
            tool = ToolRegistry.get_tool(tool_name, dept)
            if tool:
                tools[tool_name] = tool

        return tools

    def _format_tools_for_anthropic(self) -> list:
        """Convert self._tools to Anthropic tool definition format."""
        tools = []
        for tool_name, tool_obj in (self._tools or {}).items():
            try:
                tools.append({
                    "name": tool_name,
                    "description": getattr(tool_obj, "description", f"{tool_name} tool"),
                    "input_schema": getattr(
                        tool_obj,
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                })
            except Exception:
                pass
        return tools

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a development task via Claude SDK.

        Args:
            task: Development task description string
            context: Optional canvas/session context dict

        Returns:
            Dict with status, department, content, tool_calls, and optionally
            generated_code / has_code if MQL5 code was produced.
        """
        import os
        import re
        import anthropic

        # Build system prompt with department persona + graph memory
        dept_system = self.system_prompt

        memory_ctx = ""
        try:
            if hasattr(self, "_read_relevant_memory"):
                nodes = await self._read_relevant_memory(task)
                if nodes:
                    memory_ctx = "\n\n## Relevant Memory\n" + "\n".join(
                        f"- {n['content']}" for n in nodes
                    )
        except Exception:
            pass

        full_system = dept_system + memory_ctx

        # Get tools formatted for Anthropic
        tools = self._format_tools_for_anthropic()

        # Call Claude
        try:
            if hasattr(self, "_invoke_claude"):
                result = await self._invoke_claude(
                    task=task,
                    canvas_context=context,
                    tools=tools if tools else None,
                )
            else:
                client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                kwargs = {
                    "model": os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6"),
                    "max_tokens": 4096,
                    "system": full_system,
                    "messages": [{"role": "user", "content": task}],
                }
                if tools:
                    kwargs["tools"] = tools
                resp = await client.messages.create(**kwargs)
                content = "".join(b.text for b in resp.content if b.type == "text")
                result = {"content": content, "tool_calls": []}
        except Exception as e:
            logger.error(f"{self.department.value} Claude call failed: {e}")
            return {"status": "error", "error": str(e), "department": self.department.value}

        # Extract MQL5/C++ code blocks if present
        if "```mql5" in result.get("content", "") or "```cpp" in result.get("content", ""):
            code_blocks = re.findall(r"```(?:mql5|cpp)(.*?)```", result["content"], re.DOTALL)
            if code_blocks:
                result["generated_code"] = code_blocks[0].strip()
                result["has_code"] = True

        # Write opinion to graph memory
        try:
            if hasattr(self, "_write_opinion_node") and result.get("content"):
                await self._write_opinion_node(
                    content=f"Task: {task[:200]}\nResult: {result['content'][:500]}",
                    confidence=0.7,
                    tags=[self.department.value],
                )
        except Exception:
            pass

        return {
            "status": "success",
            "department": self.department.value,
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
            "generated_code": result.get("generated_code"),
            "has_code": result.get("has_code", False),
        }

    def process_development_task(self, task: DevelopmentTask) -> EAGenerationResult:
        """
        Process a structured development task through the TRD/EA pipeline.

        Args:
            task: Development task with type and data

        Returns:
            EAGenerationResult with generation outcome
        """
        logger.info(f"Processing development task: {task.task_type}")

        if task.task_type == "generate_ea":
            return self._generate_ea_from_trd(task.trd_data)
        elif task.task_type == "validate_trd":
            return self._validate_trd_task(task.trd_data)
        elif task.task_type == "parse_trd":
            return self._parse_trd_task(task.trd_data)
        else:
            return EAGenerationResult(
                success=False,
                strategy_id="",
                version=0,
                file_path="",
                error=f"Unknown task type: {task.task_type}",
            )

    def process_trd(self, trd_data: Dict[str, Any]) -> EAGenerationResult:
        """
        Process a TRD document end-to-end: parse, validate, generate.

        Args:
            trd_data: TRD document as dictionary

        Returns:
            EAGenerationResult with generation outcome
        """
        return self._generate_ea_from_trd(trd_data)

    def _parse_trd_task(self, trd_data: Dict[str, Any]) -> EAGenerationResult:
        """Parse TRD data into a TRDDocument."""
        try:
            trd = self.parser.parse_dict(trd_data)
            return EAGenerationResult(
                success=True,
                strategy_id=trd.strategy_id,
                version=trd.version,
                file_path="",
                validation_result=None,
            )
        except Exception as e:
            logger.error(f"TRD parsing failed: {e}")
            return EAGenerationResult(
                success=False,
                strategy_id="",
                version=0,
                file_path="",
                error=str(e),
            )

    def _validate_trd_task(self, trd_data: Dict[str, Any]) -> EAGenerationResult:
        """Validate TRD document."""
        try:
            trd = self.parser.parse_dict(trd_data)
            validation_result = self.validator.validate(trd)

            if validation_result.has_blocking_issues():
                return EAGenerationResult(
                    success=False,
                    strategy_id=trd.strategy_id,
                    version=trd.version,
                    file_path="",
                    validation_result=validation_result,
                    error="TRD has validation errors",
                )

            return EAGenerationResult(
                success=True,
                strategy_id=trd.strategy_id,
                version=trd.version,
                file_path="",
                validation_result=validation_result,
            )
        except Exception as e:
            logger.error(f"TRD validation failed: {e}")
            return EAGenerationResult(
                success=False,
                strategy_id="",
                version=0,
                file_path="",
                error=str(e),
            )

    def _generate_ea_from_trd(self, trd_data: Dict[str, Any]) -> EAGenerationResult:
        """
        Generate MQL5 EA from TRD document.

        This is the main workflow:
        1. Parse TRD
        2. Validate TRD
        3. Check for ambiguities (flag for clarification if needed)
        4. Generate MQL5 code
        5. Save to output directory
        """
        try:
            # Step 1: Parse TRD
            logger.info("Step 1: Parsing TRD document")
            trd = self.parser.parse_dict(trd_data)

            # Step 2: Validate TRD
            logger.info("Step 2: Validating TRD document")
            validation_result = self.validator.validate(trd)

            # Check for blocking errors
            if validation_result.has_blocking_issues():
                error_msg = "; ".join([e.error for e in validation_result.errors])
                logger.error(f"TRD validation failed: {error_msg}")
                return EAGenerationResult(
                    success=False,
                    strategy_id=trd.strategy_id,
                    version=trd.version,
                    file_path="",
                    validation_result=validation_result,
                    error=f"Validation errors: {error_msg}",
                )

            # Step 3: Check for ambiguities
            logger.info("Step 3: Checking for ambiguities")
            clarification_request = self.validator.get_clarification_request(trd)

            if clarification_request.get("needs_clarification"):
                # Check if we need high-severity clarification
                has_high_severity = any(
                    a.get("severity") == "high"
                    for a in clarification_request.get("ambiguous_parameters", [])
                )

                if has_high_severity or clarification_request.get("missing_parameters"):
                    logger.warning(f"TRD requires clarification: {clarification_request.get('message')}")
                    return EAGenerationResult(
                        success=False,
                        strategy_id=trd.strategy_id,
                        version=trd.version,
                        file_path="",
                        validation_result=validation_result,
                        clarification_needed=True,
                        clarification_details=clarification_request,
                        error="TRD requires clarification from FloorManager",
                    )

            # Step 3.5: Build SDD spec directive from TRD
            logger.info("Step 3.5: Building SDD spec directive")
            try:
                from src.agents.skills.builtin_skills import sdd_spec_builder
                import json as _json
                trd_source_text = _json.dumps(trd.to_dict(), default=str)
                sdd_directive = sdd_spec_builder(
                    source=trd_source_text,
                    source_type="trd",
                    strategy_name=trd.strategy_name,
                )
                # Attach directive to generation context
                generation_context = {"sdd_directive": sdd_directive, "trd": trd}
                # Make the directive accessible to the generator via trd.parameters
                trd.parameters["sdd_directive"] = sdd_directive
                logger.info(f"SDD directive built for strategy: {trd.strategy_name}")
            except Exception:
                logger.warning("SDD spec builder failed — proceeding without directive", exc_info=True)
                generation_context = {"trd": trd}

            # Step 4: Generate MQL5 code
            logger.info("Step 4: Generating MQL5 EA code")
            mql5_code = self.mql5_generator.generate(trd)

            # Validate syntax
            is_valid, syntax_error = self.mql5_generator.validate_mql5_syntax(mql5_code)
            if not is_valid:
                logger.error(f"MQL5 syntax validation failed: {syntax_error}")
                return EAGenerationResult(
                    success=False,
                    strategy_id=trd.strategy_id,
                    version=trd.version,
                    file_path="",
                    validation_result=validation_result,
                    error=f"MQL5 syntax error: {syntax_error}",
                )

            # Step 5: Save to output directory
            logger.info("Step 5: Saving EA to storage")
            trd_snapshot = trd.to_dict()
            ea_output = self.ea_storage.save_ea(
                strategy_id=trd.strategy_id,
                strategy_name=trd.strategy_name,
                mql5_code=mql5_code,
                trd_snapshot=trd_snapshot,
            )

            # Update compile status to pending
            self.ea_storage.update_compile_status(
                strategy_id=ea_output.strategy_id,
                version=ea_output.version,
                compile_status=COMPILE_STATUS_PENDING,
            )

            # Trigger compilation (Story 7.3) - Subtask 4.1
            compile_result = self.compilation_service.compile_ea(
                strategy_id=ea_output.strategy_id,
                version=ea_output.version,
            )

            # Log compilation result
            if compile_result.success:
                logger.info(
                    f"Compilation succeeded for {ea_output.strategy_id} v{ea_output.version}"
                )
            else:
                logger.warning(
                    f"Compilation failed for {ea_output.strategy_id} v{ea_output.version}: "
                    f"{compile_result.errors}"
                )
                # Escalate to FloorManager if needed
                if compile_result.escalated_to_floor_manager:
                    self._escalate_compilation_failure(
                        strategy_id=ea_output.strategy_id,
                        version=ea_output.version,
                        compile_result=compile_result,
                    )

            logger.info(f"EA generated successfully: {ea_output.file_path}")

            return EAGenerationResult(
                success=True,
                strategy_id=ea_output.strategy_id,
                version=ea_output.version,
                file_path=ea_output.file_path,
                validation_result=validation_result,
            )

        except Exception as e:
            logger.error(f"EA generation failed: {e}")
            return EAGenerationResult(
                success=False,
                strategy_id="",
                version=0,
                file_path="",
                error=str(e),
            )

    def request_clarification(
        self,
        trd: TRDDocument,
        validation_result: ValidationResult,
    ) -> Dict[str, Any]:
        """
        Request clarification from FloorManager for ambiguous parameters.

        Args:
            trd: The TRD document
            validation_result: Validation result with ambiguities

        Returns:
            Formatted clarification request
        """
        clarification = self.validator.get_clarification_request(trd)

        # Add department context
        clarification["source_department"] = Department.DEVELOPMENT.value
        clarification["target_department"] = "floor_manager"
        clarification["subject"] = f"TRD Clarification Request: {trd.strategy_name}"

        return clarification

    def send_to_compilation(self, ea_output) -> Dict[str, Any]:
        """
        Send generated EA to compilation step (Story 7.3).

        Args:
            ea_output: The generated EA output

        Returns:
            Result of sending to compilation
        """
        try:
            # Send to compilation department via mail
            result = self.send_result(
                to_dept=Department.DEVELOPMENT,
                subject=f"EA Ready for Compilation: {ea_output.strategy_id} v{ea_output.version}",
                body=f"""EA Generation Complete

Strategy: {ea_output.strategy_name}
Version: {ea_output.version}
File: {ea_output.file_path}

TRD Snapshot:
- Symbol: {ea_output.trd_snapshot.get('symbol')}
- Timeframe: {ea_output.trd_snapshot.get('timeframe')}
- Strategy Type: {ea_output.trd_snapshot.get('strategy_type')}

Ready for compilation (Story 7.3).
""",
                priority="normal",
            )

            return {
                "status": "sent",
                "message_id": result.get("message_id"),
                "next_step": "compilation",
            }

        except Exception as e:
            logger.error(f"Failed to send to compilation: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def _escalate_compilation_failure(
        self,
        strategy_id: str,
        version: int,
        compile_result,
    ) -> None:
        """
        Escalate compilation failure to FloorManager.

        Args:
            strategy_id: Strategy identifier
            version: Version number
            compile_result: Compilation result from compilation service
        """
        try:
            escalation = self.compilation_service.escalate_to_floor_manager(
                strategy_id=strategy_id,
                version=version,
                reason=compile_result.escalation_reason or "Compilation failed",
                errors=compile_result.errors,
            )

            # Send escalation to FloorManager
            self.send_result(
                to_dept="floor_manager",
                subject=escalation["subject"],
                body=escalation["body"],
                priority=escalation["priority"],
            )

            logger.info(f"Escalated compilation failure for {strategy_id} v{version} to FloorManager")

        except Exception as e:
            logger.error(f"Failed to escalate compilation failure: {e}")
