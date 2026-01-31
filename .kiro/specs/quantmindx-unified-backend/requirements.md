# Requirements Document

## Introduction

QuantMindX is an AI-powered trading system with multi-agent architecture that requires implementation of the Backend Implementation Plan v7 "Hybrid Core" components. Based on the approved v7 specification, this document consolidates the critical missing backend components: Agent Workspaces & Queues, Database Layer (SQLite + ChromaDB), QuantMind Standard Library (QSL) modular MQL5 components, and PropFirm Module database integration. This specification focuses exclusively on backend implementation, with frontend user stories documented separately but out of scope for this implementation phase.

## Glossary

- **QuantMindX**: AI-powered trading system with multi-agent architecture
- **QSL**: QuantMind Standard Library - modular MQL5 library system replacing monolithic QuantMind_Risk.mqh
- **Agent_Workspace**: Isolated directory structure for analyst/, quant/, and executor/ agents
- **FIFO_Queue**: File-based JSON task queues for asynchronous agent coordination
- **ChromaDB**: Vector database for semantic search replacing Qdrant
- **SQLite**: Local-first relational database for PropFirm accounts and daily snapshots
- **PropManager**: MQL5 module for daily drawdown tracking and news guard functionality
- **RiskClient**: MQL5 module for Python bridge communication with heartbeat system
- **KellySizer**: MQL5 module for position sizing calculations
- **JSON_Parser**: MQL5 utility module (CJAVal) for parsing risk_matrix.json
- **Sockets_Module**: MQL5 utility for WebSocket bridge communication
- **Quadratic_Throttle**: Risk reduction algorithm based on distance to daily loss limit
- **Kelly_Filter**: A+ setup filter (KellyScore >= 0.8) for preservation mode
- **Coin_Flip_Bot**: Minimal risk bot for meeting minimum trading day requirements
- **Migration_Strategy**: Backward-compatible transition from monolithic to modular architecture
- **LangGraph**: State-based agent workflow framework for building multi-step agent processes
- **StateGraph**: LangGraph's graph construction pattern with nodes, edges, and conditional routing
- **AgentState**: TypedDict with message accumulation for maintaining conversation context
- **MCP_Tool**: Model Context Protocol tool decorator for exposing agent capabilities
- **LangMem**: Long-term memory SDK for semantic, episodic, and procedural memory management
- **ReflectionExecutor**: Deferred memory processing system to avoid redundant work
- **Memory_Namespace**: Hierarchical organization pattern for user/team/project memory isolation

## Requirements

### Requirement 1: Agent Workspaces and Queue System

**User Story:** As an AI agent, I want dedicated workspace environments with task queues, so that I can operate independently while coordinating with other agents without file collisions.

#### Acceptance Criteria

1. THE System SHALL create workspaces/ directory with analyst/, quant/, and executor/ subdirectories
2. THE Analyst workspace SHALL contain specs/, logs/, and inputs/ directories for TRD documents and research
3. THE Quant workspace SHALL contain strategies/ and backtests/ directories for Python strategy drafts and artifacts
4. THE Executor workspace SHALL contain deployment/ and heartbeat/ directories for manifests and live status logs
5. WHEN agents submit tasks, THE System SHALL queue them in data/queues/ using JSON-based FIFO format
6. THE System SHALL provide separate queue files for each agent type (quant_tasks.json, analyst_tasks.json)
7. WHEN multiple agents access workspaces concurrently, THE System SHALL prevent file conflicts through proper isolation
8. THE Workspace system SHALL support "Cloud Code" style asynchronous task processing

### Requirement 2: Database Layer Implementation (Local-First Persistence)

**User Story:** As a system administrator, I want local-first database persistence that mimics cloud architecture, so that PropFirm accounts and trading data are reliably stored and accessible across system restarts.

#### Acceptance Criteria

1. THE System SHALL implement SQLite database at data/quantmind.db for structured data storage
2. THE System SHALL implement ChromaDB at data/chromadb/ for vector embeddings replacing Qdrant
3. THE SQLite schema SHALL include PropFirmAccounts table with id, firm_name, and daily_loss_limit fields
4. THE SQLite schema SHALL include DailySnapshots table with account_id, high_water_mark, and current_equity fields
5. THE DailySnapshots table SHALL drive the Quadratic Throttle calculations for risk management
6. THE System SHALL use SQLAlchemy ORM for Python database operations
7. WHEN database connections fail, THE System SHALL implement automatic reconnection with proper error handling
8. THE ChromaDB SHALL store embeddings for articles and strategy DNA for semantic search capabilities

### Requirement 3: QuantMind Standard Library (QSL) Modular Architecture

**User Story:** As a trading system developer, I want modular MQL5 "Lego Block" components that replace the monolithic QuantMind_Risk.mqh, so that agents can discover and build with reusable components.

#### Acceptance Criteria

1. THE QSL SHALL be organized under src/mql5/Include/QuantMind/ with Core/, Risk/, Signals/, and Utils/ subdirectories
2. THE Risk/PropManager.mqh SHALL implement DailyDrawdownLock with 4.5% hard stop and NewsGuard functionality
3. THE Risk/RiskClient.mqh SHALL implement Python bridge communication with recursive file watcher on risk_matrix.json
4. THE Risk/KellySizer.mqh SHALL provide position sizing logic using Kelly criterion calculations
5. THE Utils/JSON.mqh SHALL implement CJAVal parser for robust JSON parsing capabilities
6. THE Utils/Sockets.mqh SHALL provide WebSocket bridge communication utilities
7. THE Signals/Indicators/ modules SHALL utilize CRiBuff (Ring Buffer) for O(1) indicator memory performance
8. WHEN QSL modules are imported, THE modules SHALL be self-contained with minimal cross-dependencies

### Requirement 4: PropFirm Module Python Implementation

**User Story:** As a prop firm trader, I want the "Offensive Brain" Kelly Filter and Quadratic Throttle logic implemented in Python, so that only the best trades are executed with appropriate risk scaling.

#### Acceptance Criteria

1. THE PropCommander in src/router/prop/commander.py SHALL implement Kelly Filter with KellyScore >= 0.8 threshold for A+ setups
2. WHEN PropCommander is in preservation mode, THE System SHALL reject all trades below the Kelly threshold
3. WHEN minimum trading days are not met, THE PropCommander SHALL activate Coin Flip Bot with MIN_DAYS_TICKER strategy
4. THE PropGovernor in src/router/prop/governor.py SHALL implement Quadratic Throttle using formula: Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss) ^ 2
5. THE PropGovernor SHALL apply hard stop when daily loss reaches effective limit (4% with 1% buffer)
6. THE PropState SHALL retrieve account metrics from database instead of in-memory storage
7. WHEN news events occur (KILL_ZONE), THE PropGovernor SHALL implement hard stop with allocation_scalar = 0.0
8. THE PropFirm module SHALL integrate with existing Commander and Governor base classes

### Requirement 5: Migration Strategy (v6 to v7 Refactoring)

**User Story:** As a system maintainer, I want a structured migration from the monolithic QuantMind_Risk.mqh to modular QSL components, so that existing functionality is preserved while enabling the new architecture.

#### Acceptance Criteria

1. THE Migration SHALL preserve existing QuantMind_Risk.mqh file during transition period with deprecation warnings
2. THE System SHALL deconstruct monolithic components: CJAVal to Utils/JSON.mqh, Heartbeat to Risk/RiskClient.mqh
3. THE Migration SHALL create new PropManager.mqh with enhanced PropFirm logic not present in legacy version
4. THE System SHALL initialize data/quantmind.db and run database migrations for persistent storage
5. WHEN legacy components are accessed, THE System SHALL log deprecation warnings but maintain functionality
6. THE Migration SHALL provide rollback capability to monolithic architecture if issues are discovered
7. THE System SHALL validate that all existing EA functionality works with new modular components
8. THE Migration process SHALL be reversible without data loss or system downtime

### Requirement 6: Integration and Communication Layer

**User Story:** As a system architect, I want robust MQL5-Python integration with heartbeat monitoring, so that the multi-language architecture operates reliably with proper health monitoring.

#### Acceptance Criteria

1. THE Risk/RiskClient.mqh SHALL implement REST heartbeat using WebRequest POST to http://localhost:8000/heartbeat
2. THE Heartbeat system SHALL send JSON payload with ea_name, symbol, magic_number, risk_multiplier, and timestamp
3. THE System SHALL implement fast path risk retrieval via GlobalVariableGet("QM_RISK_MULTIPLIER")
4. WHEN GlobalVariable is unavailable, THE System SHALL fallback to reading risk_matrix.json from MQL5/Files/
5. THE DiskSyncer in src/router/sync.py SHALL write risk_matrix.json atomically using temp file then rename
6. THE System SHALL implement recursive file watcher for risk_matrix.json changes
7. WHEN communication failures occur, THE System SHALL implement retry mechanisms with exponential backoff
8. THE Integration layer SHALL support multiple concurrent EA connections without performance degradation

### Requirement 7: LangChain Agent Implementation Patterns

**User Story:** As an agent developer, I want to implement agents using proven LangChain patterns and best practices, so that the system follows established conventions for tool usage, state management, and agent coordination.

#### Acceptance Criteria

1. THE System SHALL use create_react_agent pattern for single agents with tools following ReAct methodology
2. THE Agent tools SHALL use @tool decorator with clear descriptions and Pydantic schema validation
3. THE System SHALL implement init_chat_model with "anthropic:claude-3-5-sonnet" for consistent model usage
4. THE Agent creation SHALL follow the pattern: model + tools + prompt → create_react_agent
5. THE System SHALL implement proper error handling in tools with actionable error messages
6. THE Agent invocation SHALL support both single-turn (invoke) and streaming (stream) execution modes
7. THE System SHALL implement agent checkpointing for multi-turn conversations and state persistence
8. THE Tool runtime SHALL implement proper schema validation and type checking for all inputs
9. THE System SHALL use LangChain's memory patterns for cross-session agent state management
10. THE Agent architecture SHALL support both synchronous and asynchronous execution patterns

### Requirement 8: Dependencies and Environment Setup

**User Story:** As a system administrator, I want proper dependency management and environment setup, so that the v7 backend components can be installed and configured correctly.

#### Acceptance Criteria

1. THE System SHALL add sqlalchemy and chromadb to requirements.txt for database functionality
2. THE System SHALL ensure mcp-metatrader5-server is active for MQL5 integration
3. THE System SHALL create data/queues/ directory for FIFO task queue storage
4. THE System SHALL initialize data/chromadb/ directory for vector database storage
5. WHEN dependencies are installed, THE System SHALL verify database connections are functional
6. THE System SHALL provide setup scripts for initializing workspace directories and database schema
7. THE System SHALL validate that all required Python packages are compatible with existing codebase
8. THE Environment setup SHALL support both development and production deployment configurations

### Requirement 8: Dependencies and Environment Setup

**User Story:** As a system administrator, I want proper dependency management and environment setup, so that the v7 backend components can be installed and configured correctly.

#### Acceptance Criteria

1. THE System SHALL add sqlalchemy, chromadb, langchain, langgraph, and langmem to requirements.txt
2. THE System SHALL ensure mcp-metatrader5-server is active for MQL5 integration
3. THE System SHALL create data/queues/ directory for FIFO task queue storage
4. THE System SHALL initialize data/chromadb/ directory for vector database storage
5. THE System SHALL install langgraph-cli[inmem] for development and debugging capabilities
6. WHEN dependencies are installed, THE System SHALL verify database connections are functional
7. THE System SHALL provide setup scripts for initializing workspace directories and database schema
8. THE System SHALL validate that all required Python packages are compatible with existing codebase
9. THE Environment setup SHALL support both development and production deployment configurations
10. THE System SHALL configure LangSmith integration for agent monitoring and debugging

### Requirement 9: Documentation and Knowledge Management

**User Story:** As a developer, I want comprehensive documentation of the QSL modules and database schema, so that I can effectively use and maintain the modular architecture.

#### Acceptance Criteria

1. THE System SHALL create docs/knowledge/mql5_asset_index.md documenting all QSL modules and their interfaces
2. THE Documentation SHALL include usage examples for each QSL module with code snippets
3. THE System SHALL document database schema with table relationships and field descriptions
4. THE Documentation SHALL include migration guide from monolithic to modular architecture
5. THE System SHALL provide API documentation for PropCommander and PropGovernor classes
6. THE Documentation SHALL include troubleshooting guide for common integration issues
7. THE System SHALL maintain changelog documenting differences between v6 and v7 implementations
8. THE Documentation SHALL be automatically updated when QSL modules or database schema change

### Requirement 9: Documentation and Knowledge Management

**User Story:** As a developer, I want comprehensive documentation of the QSL modules and database schema, so that I can effectively use and maintain the modular architecture.

#### Acceptance Criteria

1. THE System SHALL create docs/knowledge/mql5_asset_index.md documenting all QSL modules and their interfaces
2. THE Documentation SHALL include usage examples for each QSL module with code snippets
3. THE System SHALL document database schema with table relationships and field descriptions
4. THE Documentation SHALL include migration guide from monolithic to modular architecture
5. THE System SHALL provide API documentation for PropCommander and PropGovernor classes
6. THE Documentation SHALL include troubleshooting guide for common integration issues
7. THE System SHALL maintain changelog documenting differences between v6 and v7 implementations
8. THE Documentation SHALL include LangGraph agent workflow diagrams and state transition documentation
9. THE Documentation SHALL provide LangMem memory management patterns and namespace organization guides
10. THE Documentation SHALL be automatically updated when QSL modules or database schema change

### Requirement 10: Verification and Testing

**User Story:** As a quality assurance engineer, I want comprehensive verification steps to ensure all v7 backend components are working correctly, so that the system meets the approved specification requirements.

#### Acceptance Criteria

1. THE System SHALL provide agent verification by running Copilot with /run ls -la command
2. THE System SHALL verify skill generation by asking Copilot to create a Pivot Points calculation skill
3. THE System SHALL verify heartbeat functionality by compiling TestRisk.mq5 and checking localhost:8000 access logs
4. THE System SHALL verify database connectivity by creating test PropFirmAccount and DailySnapshot records
5. THE System SHALL verify QSL module imports by compiling test EA using new modular components
6. THE System SHALL verify PropCommander Kelly Filter logic with test trade proposals
7. THE System SHALL verify PropGovernor Quadratic Throttle calculations with test scenarios
8. THE System SHALL verify LangGraph agent workflows by executing test scenarios with state transitions
9. THE System SHALL verify LangMem memory operations by testing semantic, episodic, and procedural memory storage/retrieval
10. THE Verification process SHALL validate backward compatibility with existing QuantMind_Risk.mqh usage

### Requirement 10: LangGraph Agent Architecture Implementation

**User Story:** As an AI agent developer, I want LangGraph-based state management and workflow orchestration, so that agents can maintain conversation context and execute complex multi-step processes with proper routing and memory management.

#### Acceptance Criteria

1. THE System SHALL implement StateGraph pattern for Analyst, QuantCode, and Executor agents using LangGraph framework
2. THE AgentState SHALL use TypedDict with add_messages annotation for automatic message accumulation
3. THE Agent workflows SHALL use explicit node functions that return partial state updates as dictionaries
4. THE System SHALL implement conditional edges for dynamic routing based on agent state and context
5. THE Analyst Agent SHALL use LangGraph nodes for: research_planning, knowledge_extraction, strategy_synthesis, and validation
6. THE QuantCode Agent SHALL use LangGraph nodes for: planning, coding, backtesting, analysis, and reflection with retry logic
7. THE Executor Agent SHALL use LangGraph nodes for: deployment_planning, ea_compilation, risk_validation, and monitoring
8. WHEN agents need to coordinate, THE System SHALL use LangGraph's subagent pattern for delegation and handoffs
9. THE Agent graphs SHALL be compiled with proper START and END node connections
10. THE System SHALL support both synchronous (invoke) and asynchronous (ainvoke) agent execution patterns

### Requirement 11: MCP Tool Integration and Skill Management

**User Story:** As an agent, I want standardized MCP tool interfaces and dynamic skill loading capabilities, so that I can access external systems and acquire new capabilities at runtime.

#### Acceptance Criteria

1. THE System SHALL implement @mcp.tool() decorators for all agent-accessible functions following MCP specification
2. THE MCP tools SHALL include proper Pydantic schema validation for input parameters
3. THE System SHALL implement create_manage_memory_tool and create_search_memory_tool for agent memory access
4. THE Skill management system SHALL support dynamic skill loading using load_skill tool pattern
5. THE MCP servers SHALL implement FastMCP framework for tool exposure with proper logging
6. THE System SHALL provide MCP tools for: database queries, file operations, MT5 integration, and knowledge base search
7. THE Tool descriptions SHALL be clear and specific to enable proper LLM decision-making
8. WHEN MCP tools encounter errors, THE System SHALL return actionable error messages to agents
9. THE System SHALL implement tool result streaming for long-running operations
10. THE MCP integration SHALL support both local and remote tool execution patterns

### Requirement 12: LangMem Memory Management System

**User Story:** As an intelligent agent, I want long-term memory capabilities with semantic, episodic, and procedural memory types, so that I can learn from interactions and improve performance over time.

#### Acceptance Criteria

1. THE System SHALL implement LangMem memory management with semantic memory for facts and relationships
2. THE Semantic memory SHALL store Triple objects with subject, predicate, object, and context fields
3. THE System SHALL implement episodic memory for capturing agent experiences with observation, thoughts, action, and result
4. THE Episodic memory SHALL enable agents to learn from successful problem-solving approaches and adapt teaching styles
5. THE System SHALL implement procedural memory for storing and optimizing agent instructions and prompts
6. THE Memory system SHALL use hierarchical namespaces: ("memories", "user_id"), ("memories", "team_id"), ("memories", "project_id")
7. THE System SHALL implement ReflectionExecutor for deferred memory processing to avoid redundant work
8. THE Memory processing SHALL wait 30+ minutes before consolidating to capture complete conversation context
9. THE System SHALL integrate with ChromaDB for vector-based memory search and retrieval
10. THE Memory system SHALL support cross-session persistence using AsyncPostgresStore for production deployment

### Requirement 13: Agent Communication and Coordination Patterns

**User Story:** As a multi-agent system coordinator, I want standardized communication patterns and handoff mechanisms, so that agents can collaborate effectively on complex trading tasks.

#### Acceptance Criteria

1. THE System SHALL implement agent handoff patterns using LangGraph's multi-agent coordination capabilities
2. THE Agent communication SHALL use structured message formats with proper role and content fields
3. THE System SHALL implement router agents for delegating tasks to appropriate specialist agents
4. THE Subagent pattern SHALL allow wrapping agents as tools for parallel execution and delegation
5. THE System SHALL implement shared state management for coordinated agent workflows
6. THE Agent coordination SHALL support both synchronous and asynchronous communication patterns
7. THE System SHALL implement agent skill sharing through centralized skill registry
8. WHEN agents need to escalate decisions, THE System SHALL provide human-in-the-loop integration points
9. THE Communication layer SHALL implement proper error handling and retry mechanisms
10. THE Agent coordination SHALL maintain audit trails for all inter-agent communications

### Requirement 14: Development and Debugging Infrastructure

**User Story:** As a developer, I want comprehensive development tools and debugging capabilities, so that I can effectively build, test, and maintain the LangGraph-based agent system.

#### Acceptance Criteria

1. THE System SHALL implement langgraph.json configuration for graph definitions and dependencies
2. THE Development environment SHALL support langgraph dev command for local development server
3. THE System SHALL integrate with LangSmith Studio for visual debugging at https://smith.langchain.com/studio/
4. THE Agent graphs SHALL support streaming with stream_mode="messages" for real-time token streaming
5. THE System SHALL implement proper checkpointing for multi-turn conversations and state persistence
6. THE Development tools SHALL provide graph visualization and execution tracing capabilities
7. THE System SHALL implement comprehensive logging for agent state transitions and decision points
8. THE Debugging infrastructure SHALL support step-by-step execution and state inspection
9. THE System SHALL provide performance monitoring for agent execution times and resource usage
10. THE Development environment SHALL support hot reloading of agent configurations and code changes

### Requirement 15: Frontend Integration Points (Out of Scope)

**User Story:** As a frontend developer, I want to understand the backend API endpoints and data structures that will be available, so that I can plan frontend integration even though frontend implementation is not part of this specification.

#### Acceptance Criteria

1. THE Backend SHALL expose REST API endpoints for PropFirm account management and daily snapshots
2. THE Backend SHALL provide WebSocket endpoints for real-time EA heartbeat monitoring
3. THE Backend SHALL expose ChromaDB search endpoints for strategy and knowledge base queries
4. THE Backend SHALL provide JSON API for agent workspace file management and task queue operations
5. THE Backend SHALL expose database query endpoints for trade proposal management and approval workflows
6. THE Backend SHALL provide configuration endpoints for PropCommander and PropGovernor parameter updates
7. THE Backend SHALL expose logging and monitoring endpoints for system health dashboards
8. THE Backend SHALL provide LangGraph agent status and execution monitoring endpoints
9. THE Backend SHALL expose LangMem memory search and management endpoints for agent memory visualization
10. THE API documentation SHALL be provided for future frontend integration but frontend implementation is explicitly out of scope