const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak
} = require("docx");

// ─── Helpers ──────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const hdrShade = { fill: "1B3A5C", type: ShadingType.CLEAR };
const altShade = { fill: "F2F7FB", type: ShadingType.CLEAR };
const cellPad = { top: 60, bottom: 60, left: 100, right: 100 };

function hdr(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ heading: level, children: [new TextRun({ text, bold: true })] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts,
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts.run })],
  });
}
function bold(text) {
  return new TextRun({ text, bold: true, size: 22, font: "Arial" });
}
function code(text) {
  return new TextRun({ text, font: "Consolas", size: 20 });
}
function codeBlock(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Consolas", size: 18 })],
  });
}
function richP(runs) {
  return new Paragraph({ spacing: { after: 120 }, children: runs });
}

// table helper
function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const hdrRow = new TableRow({
    children: headers.map((h, i) =>
      new TableCell({
        borders, shading: hdrShade, width: { size: colWidths[i], type: WidthType.DXA },
        margins: cellPad,
        children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 20, font: "Arial" })] })],
      })
    ),
  });
  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          borders, width: { size: colWidths[ci], type: WidthType.DXA },
          margins: cellPad,
          shading: ri % 2 === 1 ? altShade : undefined,
          children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20, font: "Arial" })] })],
        })
      ),
    })
  );
  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths: colWidths, rows: [hdrRow, ...dataRows] });
}

// Bullet config
const numbering = {
  config: [
    {
      reference: "bullets", levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "numbers", levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    },
    {
      reference: "subbullets", levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 360 } } },
      }],
    },
  ],
};

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 22, font: "Arial" })],
  });
}
function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 22, font: "Arial" })],
  });
}
function subbullet(text) {
  return new Paragraph({
    numbering: { reference: "subbullets", level: 0 },
    spacing: { after: 40 },
    children: [new TextRun({ text, size: 20, font: "Arial" })],
  });
}

// ─── DOCUMENT CONTENT ────────────────────────────────────────────

const children = [];

// ============ TITLE PAGE ============
children.push(new Paragraph({ spacing: { before: 3000 } }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "QuantMindX", size: 56, bold: true, font: "Arial", color: "1B3A5C" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Agentic System Migration Specification", size: 36, font: "Arial", color: "1B3A5C" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Full Migration to Claude Agent SDK", size: 28, font: "Arial", color: "555555" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "Version 1.0 \u2014 March 30, 2026", size: 22, font: "Arial", color: "888888" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "CONFIDENTIAL \u2014 For Coding Agent Use Only", size: 20, font: "Arial", color: "CC0000", bold: true })],
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ TABLE OF CONTENTS (manual) ============
children.push(hdr("Table of Contents"));
const tocEntries = [
  "1. Executive Summary",
  "2. Current State Audit",
  "3. Claude Agent SDK Architecture",
  "4. Migration Plan: Phase 1 \u2014 Core Agent Infrastructure",
  "5. Migration Plan: Phase 2 \u2014 Department Heads & Tool System",
  "6. Migration Plan: Phase 3 \u2014 Harness Infrastructure",
  "7. Migration Plan: Phase 4 \u2014 Workflow Implementations (WF1\u2013WF4)",
  "8. Migration Plan: Phase 5 \u2014 UI/Frontend Integration",
  "9. Migration Plan: Phase 6 \u2014 Multi-Provider & Settings",
  "10. Dead Code Removal Checklist",
  "11. File Change Registry",
  "12. Acceptance Criteria",
  "13. Post-Migration: Skill Writing",
];
tocEntries.forEach(e => children.push(p(e, { indent: { left: 360 } })));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 1. EXECUTIVE SUMMARY ============
children.push(hdr("1. Executive Summary"));
children.push(p("This document specifies the complete migration of the QuantMindX agentic system from its current mixed-framework state (LangChain stubs, LangGraph remnants, raw Anthropic SDK calls) to a unified Claude Agent SDK architecture. The migration covers 175 Python files across the src/agents/ directory, the frontend chat UI in quantmind-ide/, and the API layer in src/api/."));
children.push(p("The coding agent receiving this document should treat it as the single source of truth for all changes. Every file that needs modification is listed with its exact path, what needs to change, code snippets showing the target pattern, and acceptance criteria."));

children.push(hdr("1.1 Scope", HeadingLevel.HEADING_2));
children.push(bullet("Remove ALL LangChain, LangGraph, CrewAI, and AutoGen imports and dependencies"));
children.push(bullet("Rewrite BaseAgent using Claude Agent SDK query() pattern"));
children.push(bullet("Migrate all 7 department heads to SDK subagent pattern"));
children.push(bullet("Convert all tools to SDK MCP tool format"));
children.push(bullet("Wire harness infrastructure (checkpointing, crash recovery, session resume)"));
children.push(bullet("Implement all 4 PRD workflows end-to-end"));
children.push(bullet("Fix UI chat so users can talk to agents"));
children.push(bullet("Add multi-provider support via base_url (OpenRouter, etc.)"));
children.push(bullet("Remove all dead code and stubs"));
children.push(bullet("Consolidate memory to single SQLite system"));

children.push(hdr("1.2 Non-Goals", HeadingLevel.HEADING_2));
children.push(bullet("Writing new skills (deferred until after verification)"));
children.push(bullet("Changing the HMM/MS-GARCH/BOCPD model training pipeline"));
children.push(bullet("Modifying the risk framework (3/5/7 stays as-is)"));
children.push(bullet("UI redesign beyond fixing the chat flow"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 2. CURRENT STATE AUDIT ============
children.push(hdr("2. Current State Audit"));
children.push(p("The codebase has 175 Python files in src/agents/ totaling ~65,800 lines. Of these, 88% are framework-independent. The remaining 12% contain stubs, legacy imports, or non-functional code."));

children.push(hdr("2.1 Framework Dependency Map", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Framework", "Status", "Files Affected", "Action"],
  [
    ["Anthropic SDK (raw)", "ACTIVE", "14 files", "Migrate to Agent SDK query()"],
    ["LangChain", "COMMENTED OUT", "7 files", "Remove all imports"],
    ["LangGraph", "COMMENTED OUT", "1 file (base_agent.py)", "Delete entirely"],
    ["CrewAI", "Never adopted", "0 files", "No action"],
    ["AutoGen", "Never adopted", "0 files", "No action"],
    ["OpenAI", "Fallback stubs", "2 files (llm_provider.py, providers/)", "Remove"],
  ],
  [2200, 1800, 2200, 3160],
));

children.push(hdr("2.2 Critical Broken Components", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Component", "File", "Problem", "Severity"],
  [
    ["BaseAgent", "src/agents/core/base_agent.py", "All methods raise NotImplementedError", "CRITICAL"],
    ["SimpleLLM", "src/agents/llm_provider.py", "Returns mock 'LLM stub response'", "CRITICAL"],
    ["Provider Router", "src/agents/providers/router.py", "Multi-provider but unused", "HIGH"],
    ["WorkflowCoordinator", "src/agents/departments/workflow_coordinator.py", "In-memory state, lost on restart", "HIGH"],
    ["CodingSkill", "src/agents/skills/coding.py", "File ops raise NotImplementedError", "HIGH"],
    ["DI Container", "src/agents/di_container.py", "Injects stubs", "MEDIUM"],
    ["Queue Manager", "src/agents/queue_manager.py", "Commented LangChain imports", "MEDIUM"],
  ],
  [1800, 2800, 3000, 1760],
));

children.push(hdr("2.3 What Actually Works Today", HeadingLevel.HEADING_2));
children.push(bullet("DepartmentHead._invoke_claude() in heads/base.py (lines 551-699) \u2014 the ONLY functional agentic loop"));
children.push(bullet("6 department head implementations (Research, Development, Trading, Risk, Portfolio, Analysis) using AsyncAnthropic"));
children.push(bullet("8 sub-agent implementations using Anthropic SDK"));
children.push(bullet("Redis-backed DepartmentMailService for inter-department communication"));
children.push(bullet("Redis Streams TaskRouter with priority queues"));
children.push(bullet("Memory system: DepartmentMemoryManager (Markdown), AgentMemory (SQLite), GraphMemory"));
children.push(bullet("MCP integration framework: discovery, loader, SSE handler"));
children.push(bullet("40+ domain-specific tools (backtest, trading, risk, knowledge, MQL5, PineScript)"));
children.push(bullet("Session checkpoint infrastructure (GitCheckpointManager, SessionRoutine) \u2014 EXISTS but NOT WIRED"));
children.push(bullet("System prompts for all 5 departments defined in types.py lines 165-503"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 3. CLAUDE AGENT SDK ARCHITECTURE ============
children.push(hdr("3. Claude Agent SDK Architecture"));
children.push(p("The Claude Agent SDK (Python: claude-agent-sdk-python) replaces all LLM orchestration with a single entry point: query(). Below is the target architecture that ALL QuantMindX agents must conform to."));

children.push(hdr("3.1 Core Concepts", HeadingLevel.HEADING_2));

children.push(richP([bold("query() Function: "), code("The single entry point for all agent invocations.")]));
children.push(codeBlock("async for message in query(prompt='...', options=ClaudeAgentOptions(...)): ..."));
children.push(p("Every agent call \u2014 whether a department head processing a task, a sub-agent doing research, or the floor manager routing \u2014 goes through query(). No more raw client.messages.create() loops."));

children.push(richP([bold("Message Types: "), code("SystemMessage, AssistantMessage, UserMessage, ResultMessage")]));
children.push(bullet("SystemMessage (subtype='init'): MCP server connection status, available tools"));
children.push(bullet("AssistantMessage: Claude's response with text and/or tool_use blocks"));
children.push(bullet("UserMessage: Tool results fed back into the loop"));
children.push(bullet("ResultMessage: Final result with cost, usage, session_id, subtype (success/error_*)"));

children.push(richP([bold("Tool Definition Pattern (MCP format):")]));
children.push(codeBlock(`@tool("tool_name", "Description", {"param": str, "num": float})
async def handler(args: dict) -> dict:
    return {"content": [{"type": "text", "text": "result"}], "is_error": False}`));

children.push(richP([bold("Subagent Definition:")]));
children.push(codeBlock(`AgentDefinition(
    description="What this agent does",
    prompt="System prompt for the agent",
    tools=["Read", "Grep", "Glob", "mcp__backtest__run_backtest"],
    model="sonnet",
    mcp_servers=["backtest-server"]
)`));

children.push(richP([bold("Session Management:")]));
children.push(codeBlock(`# Capture session_id from ResultMessage
# Resume: query(prompt='...', options=ClaudeAgentOptions(resume=session_id))
# Fork: query(prompt='...', options=ClaudeAgentOptions(resume=session_id, fork_session=True))`));

children.push(richP([bold("Cost Control:")]));
children.push(codeBlock(`ClaudeAgentOptions(max_turns=30, max_budget_usd=10.0, effort="high")`));

children.push(hdr("3.2 Multi-Provider via base_url", HeadingLevel.HEADING_2));
children.push(p("The SDK respects ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY environment variables. For OpenRouter or other providers, simply set these at runtime. No code changes needed."));
children.push(codeBlock(`# OpenRouter
ANTHROPIC_BASE_URL="https://openrouter.ai/api/v1"
ANTHROPIC_API_KEY="sk-or-v1-..."

# Azure AI Foundry
ANTHROPIC_BASE_URL="https://{resource}.services.ai.azure.com"

# AWS Bedrock
CLAUDE_CODE_USE_BEDROCK=1  (+ AWS credentials via boto3)`));

children.push(p("The UI settings page should let users configure: provider name, base_url, api_key, and default model. These map directly to env vars at runtime."));

children.push(hdr("3.3 Hooks for Observability", HeadingLevel.HEADING_2));
children.push(codeBlock(`ClaudeAgentOptions(hooks={
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[log_fn])],
    "PostToolUse": [HookMatcher(matcher="*", hooks=[metrics_fn])],
    "Stop": [HookMatcher(matcher="*", hooks=[cost_logger])],
})`));
children.push(p("Use hooks for: tool audit logging, cost tracking per department, FlowForge event emission, and security gating."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 4. PHASE 1: CORE AGENT INFRASTRUCTURE ============
children.push(hdr("4. Phase 1 \u2014 Core Agent Infrastructure"));
children.push(p("Priority: CRITICAL. This phase replaces the broken core with Claude Agent SDK primitives."));

children.push(hdr("4.1 Rewrite BaseAgent", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/core/base_agent.py (409 lines)"],
    ["Current State", "All methods raise NotImplementedError. Dead LangGraph stubs."],
    ["Target", "Thin wrapper around Claude Agent SDK query()"],
    ["Key Changes", "Remove all LangChain/LangGraph imports. Remove SimpleTool stubs. Implement invoke() using query(). Accept ClaudeAgentOptions. Expose session_id, cost, usage on result."],
  ],
  [2000, 7360],
));

children.push(richP([bold("Target Code Pattern:")]));
children.push(codeBlock(`class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.options = ClaudeAgentOptions(
            model=config.model,
            system_prompt=config.system_prompt,
            max_turns=config.max_turns or 20,
            max_budget_usd=config.max_budget_usd or 5.0,
            mcp_servers=config.mcp_servers or {},
            allowed_tools=config.allowed_tools or [],
            agents=config.sub_agents or {},
            hooks=config.hooks or {},
        )
        self.last_session_id = None
        self.last_cost = None

    async def invoke(self, prompt: str, resume_session: str = None) -> AgentResult:
        opts = self.options
        if resume_session:
            opts = ClaudeAgentOptions(**{**opts.__dict__, "resume": resume_session})
        result = None
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage):
                result = message
        self.last_session_id = result.session_id
        self.last_cost = result.total_cost_usd
        return AgentResult(
            content=result.result,
            session_id=result.session_id,
            cost=result.total_cost_usd,
            usage=result.usage,
            status=result.subtype,
        )`));

children.push(hdr("4.2 Delete SimpleLLM and LLM Provider", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/llm_provider.py (305 lines)"],
    ["Action", "DELETE ENTIRELY"],
    ["Reason", "SimpleLLM returns mock responses. get_llm_for_agent() injects stubs. All provider logic now handled by ANTHROPIC_BASE_URL env var."],
    ["Impact", "Update 3 files that import from llm_provider: di_container.py, queue_manager.py, test files"],
  ],
  [2000, 7360],
));

children.push(hdr("4.3 Delete Provider Router", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/providers/router.py (274 lines)"],
    ["Action", "DELETE ENTIRELY"],
    ["Reason", "Multi-provider routing now handled by env var ANTHROPIC_BASE_URL. No need for OpenRouter/Zhipu/MiniMax abstraction."],
    ["Also Delete", "src/agents/providers/client.py (364 lines) \u2014 same reason"],
  ],
  [2000, 7360],
));

children.push(hdr("4.4 Refactor DI Container", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/di_container.py (512 lines)"],
    ["Current", "Injects SimpleLLM stubs, provider router, LangChain tools"],
    ["Target", "Inject ClaudeAgentOptions factory, MCP server configs, tool registrations"],
    ["Pattern", "Container.get_agent_options(department) returns ClaudeAgentOptions pre-configured for that department"],
  ],
  [2000, 7360],
));

children.push(hdr("4.5 Agent Configuration Model", HeadingLevel.HEADING_2));
children.push(p("Replace the current DepartmentHeadConfig (types.py lines 120-159) with an SDK-native config:"));
children.push(codeBlock(`@dataclass
class AgentConfig:
    agent_id: str
    department: Department
    system_prompt: str
    model: str = "claude-sonnet-4-6"       # Configurable via UI settings
    max_turns: int = 20
    max_budget_usd: float = 5.0
    effort: str = "high"                    # low|medium|high|max
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    sub_agents: Dict[str, AgentDefinition] = field(default_factory=dict)
    hooks: Dict[str, List[HookMatcher]] = field(default_factory=dict)
    base_url: Optional[str] = None          # Override ANTHROPIC_BASE_URL
    api_key: Optional[str] = None           # Override ANTHROPIC_API_KEY`));

children.push(p("The model, base_url, and api_key fields are populated from the UI settings store (see Phase 5)."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 5. PHASE 2: DEPARTMENT HEADS & TOOLS ============
children.push(hdr("5. Phase 2 \u2014 Department Heads & Tool System"));

children.push(hdr("5.1 Migrate Department Heads", HeadingLevel.HEADING_2));
children.push(p("Currently, DepartmentHead._invoke_claude() in heads/base.py (lines 551-699) manually implements a 10-iteration tool loop using AsyncAnthropic. This must be replaced with the SDK's query() which handles the loop automatically."));

children.push(richP([bold("Files to change:")]));
children.push(makeTable(
  ["File", "Lines", "Change"],
  [
    ["src/agents/departments/heads/base.py", "551-699", "Replace _invoke_claude() with query() call via BaseAgent.invoke()"],
    ["src/agents/departments/heads/base.py", "403-454", "_build_system_prompt() stays but output feeds into AgentConfig.system_prompt"],
    ["src/agents/departments/heads/base.py", "61-111", "Standard tools (send_mail, read_memory, write_opinion) convert to @tool MCP format"],
    ["src/agents/departments/heads/research_head.py", "Full", "Inherit from new BaseAgent, configure research-specific tools"],
    ["src/agents/departments/heads/development_head.py", "Full", "Same pattern"],
    ["src/agents/departments/heads/trading_head.py", "Full", "Same pattern"],
    ["src/agents/departments/heads/risk_head.py", "Full", "Same pattern"],
    ["src/agents/departments/heads/portfolio_head.py", "Full", "Same pattern"],
    ["src/agents/departments/heads/analysis_head.py", "Full", "Same pattern"],
  ],
  [3800, 1000, 4560],
));

children.push(richP([bold("Target Pattern for Department Head:")]));
children.push(codeBlock(`class ResearchHead(BaseAgent):
    def __init__(self, config: AgentConfig):
        # Add department-specific MCP tools
        config.mcp_servers["backtest"] = backtest_mcp_config
        config.mcp_servers["knowledge"] = knowledge_mcp_config
        config.allowed_tools = [
            "send_mail", "read_memory", "write_opinion",
            "mcp__backtest__run_backtest",
            "mcp__knowledge__search",
        ]
        config.sub_agents = {
            "strategy-researcher": AgentDefinition(
                description="Researches strategy hypotheses",
                prompt=STRATEGY_RESEARCHER_PROMPT,
                tools=["mcp__knowledge__search", "mcp__backtest__run_backtest"],
                model="sonnet",
            ),
        }
        super().__init__(config)`));

children.push(hdr("5.2 Convert Tools to MCP Format", HeadingLevel.HEADING_2));
children.push(p("All tools must be converted from their current class-method style to Claude Agent SDK @tool decorator format. Tools become in-process MCP servers."));

children.push(makeTable(
  ["Current Tool File", "Tools Defined", "Target MCP Server Name"],
  [
    ["src/agents/tools/backtest_tools.py", "run_backtest, get_backtest_status, get_backtest_results", "backtest"],
    ["src/agents/tools/trading_tools.py", "execute_trade, analyze_market, calculate_risk", "trading"],
    ["src/agents/tools/risk_tools.py", "calculate_position_size, check_drawdown, calculate_var", "risk"],
    ["src/agents/tools/knowledge_tools.py", "search_knowledge_base, index_document", "knowledge"],
    ["src/agents/tools/mql5_tools.py", "compile_mql5, validate_syntax", "mql5"],
    ["src/agents/tools/pinescript_tools.py", "generate_pinescript, validate_pine", "pinescript"],
    ["src/agents/tools/memory_tools.py", "read_memory, write_memory, search_memory", "memory"],
    ["src/agents/tools/broker_tools.py", "get_account_info, get_positions", "broker"],
  ],
  [3200, 3600, 2560],
));

children.push(richP([bold("Target Tool Pattern:")]));
children.push(codeBlock(`# src/agents/tools/sdk/backtest_server.py
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("run_backtest", "Execute a strategy backtest", {
    "strategy_code": str,
    "symbol": str,
    "timeframe": str,
    "start_date": str,
    "end_date": str,
})
async def run_backtest(args: dict) -> dict:
    config = BacktestConfig(**args)
    result = await backtest_engine.run(config)
    return {
        "content": [{"type": "text", "text": json.dumps(result.to_dict())}],
        "is_error": False,
    }

backtest_mcp = create_sdk_mcp_server("backtest", tools=[run_backtest])`));

children.push(hdr("5.3 Standard Tools (send_mail, read_memory, write_opinion)", HeadingLevel.HEADING_2));
children.push(p("These three tools (defined in heads/base.py lines 61-111) are available to ALL department heads. Convert them to a shared MCP server:"));
children.push(codeBlock(`# src/agents/tools/sdk/department_tools_server.py
@tool("send_mail", "Send a message to another department", {
    "to": str,        # department name
    "subject": str,
    "body": str,
})
async def send_mail(args: dict) -> dict:
    mail_service = get_mail_service()
    await mail_service.send(from_dept=context.department, **args)
    return {"content": [{"type": "text", "text": f"Mail sent to {args['to']}"}]}

@tool("read_memory", "Search department memory for relevant context", {
    "query": str,
})
async def read_memory(args: dict) -> dict:
    memory = get_memory_facade()
    results = await memory.search(args["query"], namespace=context.department)
    return {"content": [{"type": "text", "text": json.dumps(results)}]}

@tool("write_opinion", "Write an insight to long-term memory", {
    "content": str,
    "confidence": float,  # 0.0 to 1.0
})
async def write_opinion(args: dict) -> dict:
    memory = get_memory_facade()
    await memory.store(args["content"], confidence=args["confidence"])
    return {"content": [{"type": "text", "text": "Opinion stored."}]}`));

children.push(hdr("5.4 Tool Access Control", HeadingLevel.HEADING_2));
children.push(p("The existing ToolAccessController (src/agents/departments/tool_access.py) maps departments to READ/WRITE permissions. In the SDK, this is enforced via the allowed_tools list on ClaudeAgentOptions:"));
children.push(codeBlock(`# Risk department: read-only
risk_config.allowed_tools = [
    "read_memory", "send_mail", "write_opinion",
    "mcp__risk__calculate_position_size",   # read-only annotation
    "mcp__risk__check_drawdown",
    "mcp__risk__calculate_var",
]
# Trading department: can execute
trading_config.allowed_tools = [
    "read_memory", "send_mail", "write_opinion",
    "mcp__trading__execute_trade",
    "mcp__trading__analyze_market",
    "mcp__risk__calculate_position_size",  # read-only cross-dept
]`));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 6. PHASE 3: HARNESS INFRASTRUCTURE ============
children.push(hdr("6. Phase 3 \u2014 Harness Infrastructure"));
children.push(p("The harness is the crash-recovery, checkpointing, and session-management layer that keeps long-running agents alive across restarts. QuantMindX has the primitives (GitCheckpointManager, SessionRoutine) but they are NOT wired to anything."));

children.push(hdr("6.1 Wire Checkpointing into Agent Loop", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File to Change", "src/agents/core/base_agent.py (new version from Phase 1)"],
    ["What to Add", "After each query() call that returns ResultMessage, call checkpoint_stage() with the session_id and partial results"],
    ["Recovery", "On agent init, call SessionRoutine.run_startup_routine(). If can_resume=True, pass session_id to query() via resume parameter"],
    ["Checkpoint Files", "src/agents/session/checkpoint.py (GitCheckpointManager) \u2014 already implemented, just needs wiring"],
    ["Startup Routine", "src/agents/session/routine.py (SessionRoutine) \u2014 already implemented, just needs wiring"],
  ],
  [2200, 7160],
));

children.push(richP([bold("Target Pattern:")]));
children.push(codeBlock(`class BaseAgent:
    async def invoke(self, prompt, resume_session=None):
        # 1. Check for resumable session
        if not resume_session:
            startup = await self.session_routine.run_startup_routine()
            if startup["can_resume"]:
                resume_session = startup.get("session_id")

        # 2. Run query with optional resume
        opts = self.options
        if resume_session:
            opts = ClaudeAgentOptions(**{**opts.__dict__, "resume": resume_session})

        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage):
                # 3. Checkpoint on completion
                await self.checkpoint.checkpoint_stage(
                    stage=f"agent_{self.config.agent_id}",
                    result={"session_id": message.session_id, "cost": message.total_cost_usd}
                )
                return AgentResult(...)`));

children.push(hdr("6.2 Persist Workflow State to Redis", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/departments/workflow_coordinator.py"],
    ["Current", "self._workflows = {} (in-memory dict, lost on restart)"],
    ["Target", "Store workflow state in Redis: workflow:{id} as JSON with 7-day TTL"],
    ["Pattern", "On create: redis.set(f'workflow:{id}', json.dumps(state)). On update: redis.set(...). On resume: redis.get(f'workflow:{id}')"],
    ["Recovery", "On startup, scan Redis for workflow:* keys, resume any with status=RUNNING or status=WAITING"],
  ],
  [2000, 7360],
));

children.push(hdr("6.3 Fix Task Router Deadlock", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Item", "Detail"],
  [
    ["File", "src/agents/departments/task_router.py"],
    ["Bug", "_wait_for_dependencies() polls every 0.1s in busy-wait loop with no DAG validation"],
    ["Fix 1", "Add DAG validation before dispatch_concurrent() \u2014 detect circular deps, raise TaskDependencyError"],
    ["Fix 2", "Replace busy-wait with Redis pub/sub: task completion publishes to channel, dependents subscribe"],
    ["Fix 3", "Add dead-letter queue: tasks failing 3 times move to task:dead_letter:{id}"],
  ],
  [2000, 7360],
));

children.push(hdr("6.4 Cost and Context Guards", HeadingLevel.HEADING_2));
children.push(p("Add to BaseAgent (Phase 1):"));
children.push(bullet("max_budget_usd on ClaudeAgentOptions (already supported by SDK)"));
children.push(bullet("max_turns on ClaudeAgentOptions (already supported by SDK)"));
children.push(bullet("Track cumulative cost per workflow: sum all ResultMessage.total_cost_usd across agents in a workflow"));
children.push(bullet("On error_max_budget_usd or error_max_turns ResultMessage subtypes: checkpoint, log, and alert via FlowForge"));
children.push(bullet("Context compaction: SDK handles automatically, but add to CLAUDE.md instructions for what to preserve during compaction"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 7. PHASE 4: WORKFLOWS ============
children.push(hdr("7. Phase 4 \u2014 Workflow Implementations (WF1\u2013WF4)"));
children.push(p("The PRD defines 4 workflows. Currently only WF1 (AlphaForge) has a partial linear pipeline implementation. All 4 must be implemented as SDK-native harness sessions."));

children.push(hdr("7.1 Workflow 1: AlphaForge (Bot Factory)", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Stage", "Department", "Input", "Output", "SDK Agent"],
  [
    ["1. Video Ingest", "Research", "YouTube URL or document", "Extracted trading rules, hypothesis", "research-ingest subagent"],
    ["2. Research", "Research", "Hypothesis", "TRD (Trading Requirements Document)", "strategy-researcher subagent"],
    ["3. Development", "Development", "TRD", "MQL5 EA code + validation", "mql5-dev subagent"],
    ["4. Backtesting", "Risk", "EA code", "6-mode backtest results, Sharpe/DD/WR", "backtest-runner subagent"],
    ["5. Paper Trading", "Trading", "Approved EA", "Paper trade session monitoring", "trade-monitor subagent"],
    ["6. Completion", "Portfolio", "Paper results", "Bot added to portfolio roster", "allocation-manager subagent"],
  ],
  [1600, 1200, 2000, 2400, 2160],
));

children.push(richP([bold("Implementation:")]));
children.push(codeBlock(`# src/agents/workflows/alphaforge.py
async def run_alphaforge(input_data: dict) -> WorkflowResult:
    checkpoint = get_checkpoint_manager()
    await checkpoint.start_workflow("alphaforge", ["ingest","research","dev","backtest","paper","complete"])

    for stage in ALPHAFORGE_STAGES:
        await checkpoint.current_progress.mark_stage_started(stage.name)
        agent = get_department_agent(stage.department)
        result = await agent.invoke(stage.prompt.format(**input_data))
        await checkpoint.checkpoint_stage(stage.name, result=result.to_dict())
        input_data.update(result.output)  # Feed output to next stage

    await checkpoint.end_workflow("completed")`));

children.push(hdr("7.2 Workflow 2: Iteration (Live Learning Loop)", HeadingLevel.HEADING_2));
children.push(p("Triggered when a live bot underperforms. Research Head analyzes why, proposes refinements, Development modifies EA parameters, Risk re-validates."));
children.push(bullet("Trigger: DPR score drops below threshold OR SESSION_CONCERN flag raised"));
children.push(bullet("Stages: Analyze Performance -> Identify Improvement -> Modify EA -> Re-Backtest -> Deploy Update"));
children.push(bullet("Key constraint: Only runs on weekends per PRD (Workflow 4 cadence)"));

children.push(hdr("7.3 Workflow 3: Performance Intelligence (Dead Zone)", HeadingLevel.HEADING_2));
children.push(p("Agent-driven daily analysis running 16:15-18:00 GMT in the Dead Zone. Produces EOD reports, DPR scores, SESSION_SPECIALIST tags."));
children.push(makeTable(
  ["Step", "Time (GMT)", "Action", "Output"],
  [
    ["1. EOD Report", "16:15", "Analyze day's trades, generate session summary", "EOD report JSON"],
    ["2. Session Performer ID", "16:45", "Identify SESSION_SPECIALIST bots", "Tags on bot manifests"],
    ["3. DPR Update", "17:00", "Calculate composite 0-100 scores (WR25%/PnL30%/consist20%/EV25%)", "DPR scores per bot"],
    ["4. Queue Re-rank", "17:30", "Re-order bot queue based on DPR + tier remix (T1/T3/T2)", "Updated queue order"],
    ["5. Fortnight Accumulation", "18:00", "Append to rolling 14-day intelligence file", "Fortnight file updated"],
  ],
  [1800, 1400, 3400, 2760],
));
children.push(p("Implementation: Scheduled harness session triggered by cron at 16:15 GMT. Each step is a stage in the checkpoint system."));

children.push(hdr("7.4 Workflow 4: Weekend Update Cycle", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["Day", "Action", "Agent"],
  [
    ["Friday Night", "Trigger WF3 final summary + plan weekend analysis", "Floor Manager"],
    ["Saturday", "Run WF2 refinement cycles, Walk-Forward Analysis", "Research + Development"],
    ["Sunday", "Pre-market preparation, bot roster finalization", "Portfolio + Risk"],
    ["Monday", "Deploy new paper entries, fresh roster", "Trading + Portfolio"],
  ],
  [2000, 4360, 3000],
));
children.push(p("Hard rule from PRD: NO bot parameter changes on weekdays. All modifications happen Saturday-Sunday only."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 8. PHASE 5: UI/FRONTEND ============
children.push(hdr("8. Phase 5 \u2014 UI/Frontend Integration"));
children.push(p("The user reports they CANNOT chat with any agent. Analysis reveals multiple issues in the frontend SvelteKit app and the FastAPI backend."));

children.push(hdr("8.1 Critical UI Fixes", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Issue", "Fix"],
  [
    ["quantmind-ide/src/lib/stores/departmentChatStore.ts:133", "Hardcoded API_BASE = 'http://localhost:8000/api'", "Use env var PUBLIC_API_BASE or derive from window.location"],
    ["quantmind-ide/src/lib/api/chatApi.ts:1", "Inconsistent API_BASE = '/api/chat' (relative)", "Unify with departmentChatStore to use same base"],
    ["src/api/chat_endpoints.py:221-254", "Workshop Copilot returns hardcoded placeholder", "Wire to actual Claude Agent SDK query() via FloorManager"],
    ["src/api/chat_endpoints.py:337-343", "dept_system_prompt silently falls back to None", "Require system prompt, throw clear error if missing"],
    ["src/api/chat_endpoints.py:358-360", "Errors yielded to SSE but not logged", "Add logger.error() before yielding error SSE"],
    ["src/agents/departments/floor_manager.py:149", "Dynamic import of dept heads can silently fail", "Add startup health check logging for each department"],
  ],
  [3200, 3200, 2960],
));

children.push(hdr("8.2 Chat Flow End-to-End", HeadingLevel.HEADING_2));
children.push(p("Target flow after migration:"));
children.push(numbered("User types message in DepartmentChatPanel.svelte"));
children.push(numbered("departmentChatStore.sendMessage() POSTs to /api/chat/departments/{dept}/message"));
children.push(numbered("chat_endpoints.py gets department head from FloorManager"));
children.push(numbered("Department head's BaseAgent.invoke() calls query() with system_prompt + user message"));
children.push(numbered("SDK runs agentic loop (tools, sub-agents, memory)"));
children.push(numbered("Results stream back via SSE to frontend"));
children.push(numbered("DepartmentChatPanel renders response with tool call indicators"));

children.push(hdr("8.3 UI Settings for Agent Configuration", HeadingLevel.HEADING_2));
children.push(p("The user wants a settings page (like claude.ai) where they can configure:"));
children.push(bullet("Model provider (Anthropic, OpenRouter, Azure, Bedrock)"));
children.push(bullet("Base URL for the provider"));
children.push(bullet("API key"));
children.push(bullet("Default model (claude-opus-4-6, claude-sonnet-4-6, etc.)"));
children.push(bullet("Per-department model override"));
children.push(bullet("Max budget per workflow"));
children.push(bullet("Max turns per agent invocation"));

children.push(p("These settings should be stored in a new API endpoint:"));
children.push(codeBlock(`# src/api/settings_endpoints.py
POST /api/settings/agent-config
GET  /api/settings/agent-config
{
    "default_provider": "anthropic",
    "default_base_url": null,         // null = use Anthropic default
    "default_model": "claude-sonnet-4-6",
    "api_keys": {                     // Encrypted at rest
        "anthropic": "sk-ant-...",
        "openrouter": "sk-or-...",
    },
    "department_overrides": {
        "research": {"model": "claude-opus-4-6"},
        "risk": {"model": "claude-sonnet-4-6"},
    },
    "max_budget_per_workflow": 10.0,
    "max_turns_per_agent": 30,
}`));

children.push(hdr("8.4 Chat Input Features", HeadingLevel.HEADING_2));
children.push(p("Like claude.ai, the chat input should support:"));
children.push(bullet("Model selector dropdown (reads from settings)"));
children.push(bullet("File attachment (upload to /api/chat/upload, pass as tool result to agent)"));
children.push(bullet("Canvas context toggle (attach current canvas state to system prompt)"));
children.push(bullet("Slash commands (/research, /backtest, /deploy) that route to specific workflows"));
children.push(bullet("Department selector (research, development, trading, risk, portfolio, floor-manager)"));

children.push(hdr("8.5 FlowForge Observability UI", HeadingLevel.HEADING_2));
children.push(p("The FlowForge component should display:"));
children.push(bullet("Active workflows with stage progress (from checkpoint system)"));
children.push(bullet("Agent cost tracking per department"));
children.push(bullet("Tool call logs (from PostToolUse hooks)"));
children.push(bullet("Inter-department mail messages"));
children.push(bullet("Error alerts and retry status"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 9. PHASE 6: MULTI-PROVIDER & SETTINGS ============
children.push(hdr("9. Phase 6 \u2014 Multi-Provider & Settings"));

children.push(hdr("9.1 Remove Hardcoded Providers", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Current Hardcoded Values", "Replace With"],
  [
    ["src/agents/departments/heads/base.py:600", "MiniMax-M2.7 fallback model", "Settings-driven default model"],
    ["src/agents/llm_provider.py", "OpenRouter/Zhipu/MiniMax base URLs, API key env vars", "DELETE FILE \u2014 use ANTHROPIC_BASE_URL"],
    ["src/agents/providers/router.py", "Provider priority chain with hardcoded URLs", "DELETE FILE \u2014 env var driven"],
    ["src/agents/departments/types.py:506-538", "model: 'claude-sonnet-4-20250514' hardcoded", "Read from AgentConfig which reads from settings API"],
    ["src/agents/claude_config.py", "6 agent configs with hardcoded timeouts", "Merge into AgentConfig, read from settings"],
  ],
  [3000, 3200, 3160],
));

children.push(hdr("9.2 Runtime Provider Switching", HeadingLevel.HEADING_2));
children.push(p("When the user changes provider in settings, the backend should:"));
children.push(numbered("Update the stored config via PUT /api/settings/agent-config"));
children.push(numbered("Set os.environ['ANTHROPIC_BASE_URL'] and os.environ['ANTHROPIC_API_KEY'] at runtime"));
children.push(numbered("All subsequent query() calls automatically use the new provider"));
children.push(numbered("No restart required \u2014 env vars are read per-call by the SDK"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 10. DEAD CODE REMOVAL ============
children.push(hdr("10. Dead Code Removal Checklist"));
children.push(p("These files should be DELETED as part of the migration:"));

children.push(makeTable(
  ["File", "Reason", "Lines"],
  [
    ["src/agents/llm_provider.py", "SimpleLLM returns mocks, all provider logic moves to env vars", "305"],
    ["src/agents/providers/router.py", "Multi-provider abstraction replaced by ANTHROPIC_BASE_URL", "274"],
    ["src/agents/providers/client.py", "Abstraction layer no longer needed", "364"],
    ["src/agents/core/base_agent.py", "REWRITE (not delete) \u2014 but remove all LangChain/LangGraph stubs", "409"],
    ["src/agents/hooks.py", "Deprecated, functionality consolidated in department_hooks.py", "447"],
    ["src/agents/queue_manager.py", "Has commented LangChain imports, consolidate with task_router", "572"],
  ],
  [3200, 4400, 1760],
));

children.push(p("Additionally, remove ALL commented-out imports from these files:"));
children.push(bullet("Any file with '# from langchain' or '# from langgraph' \u2014 find with: grep -r 'langchain\\|langgraph' src/agents/"));
children.push(bullet("Any file with 'NotImplementedError(\"pending migration' \u2014 find with: grep -r 'pending migration' src/agents/"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 11. FILE CHANGE REGISTRY ============
children.push(hdr("11. File Change Registry"));
children.push(p("Complete list of every file that needs modification, organized by phase:"));

children.push(hdr("Phase 1 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["src/agents/core/base_agent.py", "REWRITE", "Remove LangChain/LangGraph. Implement query()-based BaseAgent."],
    ["src/agents/llm_provider.py", "DELETE", "All functionality replaced by SDK + env vars."],
    ["src/agents/providers/router.py", "DELETE", "Provider routing via ANTHROPIC_BASE_URL."],
    ["src/agents/providers/client.py", "DELETE", "Abstraction layer removed."],
    ["src/agents/di_container.py", "REFACTOR", "Inject ClaudeAgentOptions factory instead of stubs."],
    ["src/agents/departments/types.py", "UPDATE", "Replace DepartmentHeadConfig with AgentConfig. Keep system prompts."],
    ["src/agents/hooks.py", "DELETE", "Consolidate into department_hooks.py."],
    ["requirements.txt or pyproject.toml", "UPDATE", "Add claude-agent-sdk-python. Remove langchain, langgraph deps."],
  ],
  [3200, 1200, 4960],
));

children.push(hdr("Phase 2 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["src/agents/departments/heads/base.py", "REWRITE", "Replace _invoke_claude() with BaseAgent.invoke(). Convert std tools to @tool."],
    ["src/agents/departments/heads/research_head.py", "REFACTOR", "Inherit new BaseAgent. Configure MCP tools."],
    ["src/agents/departments/heads/development_head.py", "REFACTOR", "Same pattern."],
    ["src/agents/departments/heads/trading_head.py", "REFACTOR", "Same pattern."],
    ["src/agents/departments/heads/risk_head.py", "REFACTOR", "Same pattern."],
    ["src/agents/departments/heads/portfolio_head.py", "REFACTOR", "Same pattern."],
    ["src/agents/departments/heads/analysis_head.py", "REFACTOR", "Same pattern."],
    ["src/agents/tools/sdk/ (NEW DIR)", "CREATE", "New MCP server files for each tool domain."],
    ["src/agents/departments/tool_access.py", "UPDATE", "Map permissions to allowed_tools lists."],
    ["src/agents/departments/tool_registry.py", "UPDATE", "Register SDK MCP servers instead of class instances."],
  ],
  [3500, 1200, 4660],
));

children.push(hdr("Phase 3 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["src/agents/core/base_agent.py", "ADD", "Wire GitCheckpointManager and SessionRoutine into invoke()."],
    ["src/agents/session/checkpoint.py", "KEEP", "Already implemented. No changes needed."],
    ["src/agents/session/routine.py", "KEEP", "Already implemented. No changes needed."],
    ["src/agents/departments/workflow_coordinator.py", "REFACTOR", "Replace self._workflows dict with Redis persistence."],
    ["src/agents/departments/task_router.py", "FIX", "Add DAG validation. Replace busy-wait with pub/sub. Add DLQ."],
  ],
  [3500, 1200, 4660],
));

children.push(hdr("Phase 4 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["src/agents/workflows/ (NEW DIR)", "CREATE", "New workflow harness files."],
    ["src/agents/workflows/alphaforge.py", "CREATE", "WF1: Video->Research->Dev->Backtest->Paper->Complete"],
    ["src/agents/workflows/iteration.py", "CREATE", "WF2: Analyze->Identify->Modify->Re-Backtest->Deploy"],
    ["src/agents/workflows/performance_intel.py", "CREATE", "WF3: EOD->SessionPerf->DPR->QueueRerank->Fortnight"],
    ["src/agents/workflows/weekend_update.py", "CREATE", "WF4: FridayPlan->SaturdayWFA->SundayPrep->MondayDeploy"],
    ["src/agents/departments/floor_manager.py", "REFACTOR", "Wire workflows. Add health check on startup."],
  ],
  [3500, 1200, 4660],
));

children.push(hdr("Phase 5 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["quantmind-ide/src/lib/stores/departmentChatStore.ts", "FIX", "Replace hardcoded localhost:8000 with env-driven API_BASE."],
    ["quantmind-ide/src/lib/api/chatApi.ts", "FIX", "Unify API_BASE with departmentChatStore."],
    ["src/api/chat_endpoints.py", "REFACTOR", "Wire to BaseAgent.invoke(). Fix error logging. Fix workshop stub."],
    ["src/api/floor_manager_endpoints.py", "REFACTOR", "Add startup health check logging. Wire streaming."],
    ["src/api/settings_endpoints.py (NEW)", "CREATE", "Agent config CRUD endpoint."],
    ["quantmind-ide/src/lib/components/agent-panel/ (NEW)", "CREATE", "Settings UI, model selector, slash commands."],
  ],
  [3800, 1200, 4360],
));

children.push(hdr("Phase 6 Files", HeadingLevel.HEADING_2));
children.push(makeTable(
  ["File", "Action", "What Changes"],
  [
    ["src/agents/departments/heads/base.py", "UPDATE", "Remove MiniMax-M2.7 fallback. Read model from AgentConfig."],
    ["src/agents/departments/types.py", "UPDATE", "Remove hardcoded model strings. Use settings."],
    ["src/agents/claude_config.py", "REFACTOR", "Merge into AgentConfig. Remove hardcoded timeouts."],
  ],
  [3500, 1200, 4660],
));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 12. ACCEPTANCE CRITERIA ============
children.push(hdr("12. Acceptance Criteria"));
children.push(p("Each phase must meet these criteria before moving to the next:"));

children.push(hdr("Phase 1 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("Zero LangChain/LangGraph/OpenAI imports in codebase (grep returns empty)"));
children.push(numbered("BaseAgent.invoke() successfully calls query() and returns AgentResult"));
children.push(numbered("SimpleLLM and llm_provider.py are deleted"));
children.push(numbered("Provider router files are deleted"));
children.push(numbered("All existing tests that import deleted modules are updated or removed"));

children.push(hdr("Phase 2 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("All 6 department heads inherit from new BaseAgent"));
children.push(numbered("_invoke_claude() method is removed from heads/base.py"));
children.push(numbered("All tools are registered as @tool MCP format"));
children.push(numbered("send_mail, read_memory, write_opinion work via MCP server"));
children.push(numbered("Tool access control enforced via allowed_tools (Risk dept cannot call execute_trade)"));

children.push(hdr("Phase 3 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("Agent invoke() creates checkpoint after completion"));
children.push(numbered("On simulated crash + restart, agent resumes from last checkpoint"));
children.push(numbered("WorkflowCoordinator state survives Redis restart (persisted)"));
children.push(numbered("Task router rejects circular dependencies with clear error"));
children.push(numbered("Busy-wait polling replaced with pub/sub (no CPU spin)"));

children.push(hdr("Phase 4 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("WF1 (AlphaForge): can run Video Ingest through to Paper Trading completion"));
children.push(numbered("WF3 (Performance Intel): runs on schedule, produces DPR scores"));
children.push(numbered("Each workflow stage creates a checkpoint"));
children.push(numbered("Workflow resumes from mid-stage after simulated crash"));

children.push(hdr("Phase 5 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("User can type a message in the chat UI and receive a response from any department"));
children.push(numbered("Settings page allows changing model and provider"));
children.push(numbered("Slash commands route to correct workflows"));
children.push(numbered("File attachments are passed to agent as context"));
children.push(numbered("FlowForge shows active workflows and their progress"));

children.push(hdr("Phase 6 Acceptance", HeadingLevel.HEADING_2));
children.push(numbered("Changing provider in settings immediately affects next agent call"));
children.push(numbered("No hardcoded model strings or API URLs in any Python file"));
children.push(numbered("All model/provider config comes from settings API or env vars"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 13. POST-MIGRATION: SKILL WRITING ============
children.push(hdr("13. Post-Migration: Skill Writing"));
children.push(p("After the coding agent completes Phases 1-6 and the user verifies the implementation, the next step is writing production skills. This is NOT part of the current migration \u2014 it happens AFTER verification."));

children.push(hdr("13.1 Skills to Write (After Verification)", HeadingLevel.HEADING_2));
children.push(bullet("Research skills: knowledge_search, hypothesis_writer, market_scanner"));
children.push(bullet("Development skills: mql5_generator, pinescript_generator, ea_validator"));
children.push(bullet("Trading skills: order_executor, fill_tracker, session_monitor"));
children.push(bullet("Risk skills: backtest_evaluator (6-mode), position_sizer (fractional Kelly), drawdown_monitor"));
children.push(bullet("Portfolio skills: allocation_optimizer, rebalancer, performance_reporter"));

children.push(hdr("13.2 Skill Format", HeadingLevel.HEADING_2));
children.push(p("Skills in the SDK are just specialized system prompts + tool selections for sub-agents:"));
children.push(codeBlock(`# Skills = AgentDefinition with specialized prompt + tools
skills = {
    "backtest-evaluator": AgentDefinition(
        description="Evaluates strategy backtests across 6 modes",
        prompt=BACKTEST_EVALUATOR_PROMPT,
        tools=["mcp__backtest__run_backtest", "mcp__risk__calculate_var"],
        model="sonnet",
    ),
}`));

children.push(p("The existing skill schema (src/agents/skills/skill_schema.py) can be preserved as metadata, but the actual execution uses AgentDefinition."));

children.push(new Paragraph({ spacing: { before: 600 } }));
children.push(new Paragraph({
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1B3A5C", space: 1 } },
  spacing: { after: 200 },
  children: [],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "End of Specification", size: 20, font: "Arial", color: "888888", italics: true })],
}));

// ============ BUILD DOCUMENT ============
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1B3A5C" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  numbering,
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1200, right: 1200, bottom: 1200, left: 1200 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "QuantMindX \u2014 Claude Agent SDK Migration Spec", size: 16, font: "Arial", color: "999999", italics: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 16, font: "Arial", color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: "999999" }),
            new TextRun({ text: " \u2014 CONFIDENTIAL", size: 16, font: "Arial", color: "999999" }),
          ],
        })],
      }),
    },
    children,
  }],
});

const outputPath = process.argv[2] || "/home/mubarkahimself/Desktop/QUANTMINDX/docs/QuantMindX_Claude_Agent_SDK_Migration_Spec.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Document written to: ${outputPath}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
});
