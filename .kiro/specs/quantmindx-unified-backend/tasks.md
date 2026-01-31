# Implementation Tasks: QuantMindX Unified Backend

## Task Group 0: Project Setup and Dependencies

- [x] 0.1 Install core dependencies (sqlalchemy, chromadb, langchain, langgraph, langmem)
- [x] 0.2 Install development dependencies (langgraph-cli[inmem], hypothesis, pytest)
- [x] 0.3 Configure LangSmith integration for agent monitoring
- [x] 0.4 Verify mcp-metatrader5-server is active and accessible
- [x] 0.5 Create project configuration files (langgraph.json, .env.example)

## Task Group 1: Agent Workspaces and Queue System

- [x] 1.1 Create workspace directory structure (workspaces/analyst/, quant/, executor/)
- [x] 1.2 Create analyst subdirectories (specs/, logs/, inputs/)
- [x] 1.3 Create quant subdirectories (strategies/, backtests/)
- [x] 1.4 Create executor subdirectories (deployment/, heartbeat/)
- [x] 1.5 Implement TaskQueue class with FIFO operations (enqueue, dequeue, peek, size)
- [x] 1.6 Create queue files (data/queues/analyst_tasks.json, quant_tasks.json, executor_tasks.json)
- [x] 1.7 Implement file locking mechanism for concurrent queue access
- [x] 1.8 Write unit tests for TaskQueue operations
- [x] 1.9 Write property test for FIFO ordering guarantee

## Task Group 2: Database Layer Implementation

- [x] 2.1 Create SQLAlchemy models for PropFirmAccounts table
- [x] 2.2 Create SQLAlchemy models for DailySnapshots table
- [x] 2.3 Create SQLAlchemy models for AgentTasks table
- [x] 2.4 Create SQLAlchemy models for StrategyPerformance table
- [x] 2.5 Implement DatabaseManager class with session management
- [x] 2.6 Create database initialization script with schema creation
- [x] 2.7 Implement database connection retry logic with exponential backoff
- [x] 2.8 Initialize ChromaDB collections (strategy_dna, market_research, agent_memory)
- [x] 2.9 Implement ChromaDB wrapper methods (search, add, update)
- [x] 2.10 Write unit tests for database models and operations
- [x] 2.11 Write integration tests for SQLite + ChromaDB coordination

## Task Group 3: QuantMind Standard Library (QSL) - Core Modules

- [x] 3.1 Create QSL directory structure (Core/, Risk/, Signals/, Utils/)
- [x] 3.2 Implement Core/BaseAgent.mqh with base functionality
- [x] 3.3 Implement Core/Constants.mqh with system constants
- [x] 3.4 Implement Core/Types.mqh with custom data types
- [x] 3.5 Write MQL5 compilation tests for Core modules

## Task Group 4: QSL - Risk Management Modules

- [x] 4.1 Implement Risk/PropManager.mqh with CPropManager class
- [x] 4.2 Implement PropManager daily drawdown tracking logic
- [x] 4.3 Implement PropManager news guard functionality
- [x] 4.4 Implement PropManager quadratic throttle calculation
- [x] 4.5 Implement Risk/RiskClient.mqh with CRiskClient class
- [x] 4.6 Implement RiskClient heartbeat transmission (WebRequest POST)
- [x] 4.7 Implement RiskClient GlobalVariable fast path retrieval
- [x] 4.8 Implement RiskClient file fallback mechanism (risk_matrix.json)
- [x] 4.9 Implement Risk/KellySizer.mqh with position sizing logic
- [x] 4.10 Write MQL5 compilation tests for Risk modules
- [x] 4.11 Write property test for Kelly Criterion calculation accuracy

## Task Group 5: QSL - Utility Modules

- [x] 5.1 Implement Utils/JSON.mqh with CJAVal parser
- [x] 5.2 Implement JSON parsing functions (FindJsonObject, ExtractJsonDouble)
- [x] 5.3 Implement Utils/Sockets.mqh for WebSocket communication
- [x] 5.4 Implement Utils/RingBuffer.mqh with CRiBuff for O(1) operations
- [x] 5.5 Write unit tests for JSON parsing with various input formats
- [x] 5.6 Write property test for Ring Buffer performance characteristics
- [x] 5.7 Create MQL5 asset index documentation (docs/knowledge/mql5_asset_index.md)

## Task Group 6: PropFirm Python Implementation

- [x] 6.1 Implement PropCommander class extending BaseCommander
- [x] 6.2 Implement Kelly Filter logic with 0.8 threshold
- [x] 6.3 Implement preservation mode trade rejection logic
- [x] 6.4 Implement Coin Flip Bot activation for minimum trading days
- [x] 6.5 Implement PropGovernor class extending BaseGovernor
- [x] 6.6 Implement Quadratic Throttle calculation method
- [x] 6.7 Implement hard stop logic (4.5% threshold with 1% buffer)
- [x] 6.8 Implement news guard (KILL_ZONE) detection and response
- [x] 6.9 Implement PropState database-backed storage
- [x] 6.10 Write unit tests for PropCommander and PropGovernor
- [x] 6.11 Write property tests for Quadratic Throttle formula
- [x] 6.12 Write property tests for Kelly Filter threshold enforcement

## Task Group 7: MQL5-Python Integration Layer

- [x] 7.1 Implement DiskSyncer class for atomic file operations
- [x] 7.2 Implement atomic risk_matrix.json write (temp file + rename)
- [x] 7.3 Implement GlobalVariable update via MT5 connection
- [x] 7.4 Implement file watcher for risk_matrix.json changes
- [x] 7.5 Implement heartbeat endpoint (POST /heartbeat)
- [x] 7.6 Implement heartbeat payload validation (ea_name, symbol, magic_number, etc.)
- [x] 7.7 Implement DatabaseErrorHandler with retry logic
- [x] 7.8 Implement MQL5BridgeErrorHandler for integration errors
- [x] 7.9 Implement CircuitBreaker pattern for external service calls
- [x] 7.10 Write integration tests for MQL5-Python bridge
- [x] 7.11 Write property test for atomic file write operations

## Task Group 8: LangGraph Agent Architecture

- [ ] 8.1 Define AgentState TypedDict with add_messages annotation
- [ ] 8.2 Implement Analyst agent workflow graph (research, extraction, synthesis, validation)
- [ ] 8.3 Implement QuantCode agent workflow graph (planning, coding, backtesting, analysis, reflection)
- [ ] 8.4 Implement Executor agent workflow graph (deployment, compilation, validation, monitoring)
- [ ] 8.5 Implement Router agent for task delegation
- [ ] 8.6 Implement agent node functions with partial state updates
- [ ] 8.7 Implement conditional edges for dynamic routing
- [ ] 8.8 Configure MemorySaver for agent checkpointing
- [ ] 8.9 Implement agent graph compilation with proper START/END connections
- [ ] 8.10 Write unit tests for agent state transitions
- [ ] 8.11 Write integration tests for complete agent workflows

## Task Group 9: MCP Tool Integration

- [x] 9.1 Initialize FastMCP server instance
- [x] 9.2 Implement database query MCP tool with Pydantic validation
- [x] 9.3 Implement memory search MCP tool (semantic, episodic, procedural)
- [x] 9.4 Implement file operations MCP tools (read, write, list)
- [x] 9.5 Implement MT5 integration MCP tools (account info, positions, orders)
- [x] 9.6 Implement knowledge base search MCP tool (ChromaDB)
- [x] 9.7 Implement skill loading MCP tool with dynamic registration
- [x] 9.8 Implement proper error handling with actionable messages
- [x] 9.9 Implement tool result streaming for long-running operations
- [x] 9.10 Write unit tests for each MCP tool
- [x] 9.11 Write property tests for MCP tool schema validation

## Task Group 10: LangMem Memory Management

- [x] 10.1 Implement SemanticMemory class with Triple storage
- [x] 10.2 Implement EpisodicMemory class with Episode storage
- [x] 10.3 Implement ProceduralMemory class with Instruction storage
- [x] 10.4 Configure hierarchical memory namespaces (user/team/project)
- [x] 10.5 Implement ReflectionExecutor for deferred memory processing
- [x] 10.6 Configure 30-minute delay for memory consolidation
- [x] 10.7 Integrate ChromaDB for vector-based memory search
- [x] 10.8 Implement create_manage_memory_tool for agent access
- [x] 10.9 Implement create_search_memory_tool for agent retrieval
- [x] 10.10 Configure AsyncPostgresStore for production persistence
- [x] 10.11 Write unit tests for memory storage and retrieval
- [x] 10.12 Write property tests for memory namespace hierarchy

## Task Group 11: Agent Communication and Coordination

- [ ] 11.1 Implement agent handoff patterns using LangGraph multi-agent coordination
- [ ] 11.2 Implement structured message formats with role/content fields
- [ ] 11.3 Implement router agent for task delegation
- [ ] 11.4 Implement subagent wrapping pattern for parallel execution
- [ ] 11.5 Implement shared state management for coordinated workflows
- [ ] 11.6 Implement synchronous and asynchronous communication patterns
- [ ] 11.7 Implement centralized skill registry for agent skill sharing
- [ ] 11.8 Implement human-in-the-loop integration points
- [ ] 11.9 Implement error handling and retry mechanisms
- [ ] 11.10 Implement audit trail logging for inter-agent communications
- [ ] 11.11 Write integration tests for agent coordination patterns

## Task Group 12: Development and Debugging Infrastructure

- [ ] 12.1 Create langgraph.json configuration file
- [ ] 12.2 Configure graph definitions and dependencies
- [ ] 12.3 Set up langgraph dev development server
- [ ] 12.4 Configure LangSmith Studio integration
- [ ] 12.5 Implement streaming support with stream_mode="messages"
- [ ] 12.6 Implement checkpointing for multi-turn conversations
- [ ] 12.7 Set up graph visualization and execution tracing
- [ ] 12.8 Implement comprehensive logging for state transitions
- [ ] 12.9 Set up performance monitoring for execution times
- [ ] 12.10 Configure hot reloading for agent configurations
- [ ] 12.11 Create debugging documentation and troubleshooting guide

## Task Group 13: Migration Strategy (v6 to v7)

- [ ] 13.1 Create backup of existing QuantMind_Risk.mqh
- [ ] 13.2 Add deprecation warnings to legacy components
- [ ] 13.3 Extract CJAVal parser to Utils/JSON.mqh
- [ ] 13.4 Extract Heartbeat logic to Risk/RiskClient.mqh
- [ ] 13.5 Extract GetRiskMultiplier to Risk/RiskClient.mqh
- [ ] 13.6 Implement PropManager.mqh with new PropFirm logic
- [ ] 13.7 Create migration script for database initialization
- [ ] 13.8 Implement rollback capability for migration
- [ ] 13.9 Validate existing EA functionality with new modules
- [ ] 13.10 Create migration documentation and changelog
- [ ] 13.11 Write integration tests for backward compatibility

## Task Group 14: Error Handling and Resilience

- [ ] 14.1 Implement DatabaseErrorHandler with exponential backoff
- [ ] 14.2 Implement MQL5BridgeErrorHandler for integration failures
- [ ] 14.3 Implement AgentErrorHandler for LangGraph execution errors
- [ ] 14.4 Implement CircuitBreaker pattern for external services
- [ ] 14.5 Implement graceful degradation for heartbeat failures
- [ ] 14.6 Implement fallback mechanisms for GlobalVariable access
- [ ] 14.7 Implement state recovery for agent transition errors
- [ ] 14.8 Implement comprehensive error logging with context
- [ ] 14.9 Create error handling documentation
- [ ] 14.10 Write unit tests for error handling scenarios

## Task Group 15: Testing Infrastructure

- [ ] 15.1 Set up Hypothesis for property-based testing
- [ ] 15.2 Configure pytest with proper markers and fixtures
- [ ] 15.3 Create test configuration for minimum 100 iterations per property
- [ ] 15.4 Implement mock MQL5 environment for testing
- [ ] 15.5 Create test fixtures for database operations
- [ ] 15.6 Create test fixtures for agent workflows
- [ ] 15.7 Implement test data generators for property tests
- [ ] 15.8 Set up test coverage reporting
- [ ] 15.9 Create testing documentation and guidelines
- [ ] 15.10 Configure CI/CD pipeline for automated testing

## Task Group 16: Property-Based Tests (Critical Properties)

- [ ] 16.1 Write property test: Workspace Initialization Completeness (Property 1)
- [ ] 16.2 Write property test: Task Queue FIFO Ordering (Property 2)
- [ ] 16.3 Write property test: Concurrent Workspace Isolation (Property 3)
- [ ] 16.4 Write property test: Database Initialization Completeness (Property 4)
- [ ] 16.5 Write property test: Database Reconnection Resilience (Property 5)
- [ ] 16.6 Write property test: Quadratic Throttle Formula Accuracy (Property 6)
- [ ] 16.7 Write property test: Kelly Filter Threshold Enforcement (Property 7)
- [ ] 16.8 Write property test: Hard Stop Activation (Property 8)
- [ ] 16.9 Write property test: PropState Database Retrieval (Property 9)
- [ ] 16.10 Write property test: Heartbeat Payload Completeness (Property 10)
- [ ] 16.11 Write property test: Risk Retrieval Fallback Chain (Property 11)
- [ ] 16.12 Write property test: Atomic File Write Operations (Property 12)
- [ ] 16.13 Write property test: File Change Detection (Property 13)
- [ ] 16.14 Write property test: Agent Execution Mode Support (Property 14)
- [ ] 16.15 Write property test: MCP Tool Schema Validation (Property 15)
- [ ] 16.16 Write property test: Agent State Persistence (Property 16)
- [ ] 16.17 Write property test: Memory Namespace Hierarchy (Property 17)
- [ ] 16.18 Write property test: Memory Consolidation Timing (Property 18)
- [ ] 16.19 Write property test: QSL Module Self-Containment (Property 19)
- [ ] 16.20 Write property test: Kelly Criterion Calculation Accuracy (Property 20)
- [ ] 16.21 Write property test: JSON Parsing Robustness (Property 21)
- [ ] 16.22 Write property test: Ring Buffer Performance (Property 22)
- [ ] 16.23 Write property test: Legacy Compatibility Preservation (Property 23)
- [ ] 16.24 Write property test: Migration Reversibility (Property 24)
- [ ] 16.25 Write property test: Coin Flip Bot Activation (Property 25)
- [ ] 16.26 Write property test: ChromaDB Semantic Search (Property 26)
- [ ] 16.27 Write property test: Agent Coordination Handoffs (Property 27)
- [ ] 16.28 Write property test: Audit Trail Completeness (Property 28)
- [ ] 16.29 Write property test: Performance Monitoring Coverage (Property 29)
- [ ] 16.30 Write property test: Documentation Synchronization (Property 30)

## Task Group 17: Integration Tests

- [ ] 17.1 Write integration test: MQL5-Python bridge end-to-end workflow
- [ ] 17.2 Write integration test: Heartbeat failure and recovery
- [ ] 17.3 Write integration test: File watcher functionality
- [ ] 17.4 Write integration test: Atomic file operations under load
- [ ] 17.5 Write integration test: Complete Analyst agent workflow
- [ ] 17.6 Write integration test: Complete QuantCode agent workflow
- [ ] 17.7 Write integration test: Complete Executor agent workflow
- [ ] 17.8 Write integration test: Agent coordination and handoffs
- [ ] 17.9 Write integration test: Memory persistence across sessions
- [ ] 17.10 Write integration test: SQLite and ChromaDB coordination
- [ ] 17.11 Write integration test: Concurrent database access patterns
- [ ] 17.12 Write integration test: Migration and rollback procedures

## Task Group 18: Performance Testing

- [ ] 18.1 Implement load test: Multiple concurrent EA connections
- [ ] 18.2 Implement load test: High-frequency heartbeat processing
- [ ] 18.3 Implement load test: Large-scale memory operations
- [ ] 18.4 Implement load test: Bulk database operations
- [ ] 18.5 Benchmark: Heartbeat response time < 100ms
- [ ] 18.6 Benchmark: Risk multiplier retrieval < 50ms
- [ ] 18.7 Benchmark: Database query response < 200ms
- [ ] 18.8 Benchmark: Agent workflow completion < 30s
- [ ] 18.9 Create performance testing documentation
- [ ] 18.10 Set up continuous performance monitoring

## Task Group 19: Documentation

- [ ] 19.1 Create MQL5 asset index (docs/knowledge/mql5_asset_index.md)
- [ ] 19.2 Document QSL module interfaces with code examples
- [ ] 19.3 Document database schema with relationships
- [ ] 19.4 Create migration guide from v6 to v7
- [ ] 19.5 Document PropCommander and PropGovernor APIs
- [ ] 19.6 Create troubleshooting guide for common issues
- [ ] 19.7 Create changelog documenting v6 vs v7 differences
- [ ] 19.8 Document LangGraph agent workflows with diagrams
- [ ] 19.9 Document LangMem memory management patterns
- [ ] 19.10 Create API documentation for frontend integration
- [ ] 19.11 Set up automatic documentation generation

## Task Group 20: Verification and Validation

- [ ] 20.1 Verify agent execution with /run ls -la command
- [ ] 20.2 Verify skill generation (Pivot Points calculation)
- [ ] 20.3 Verify heartbeat functionality with TestRisk.mq5
- [ ] 20.4 Verify database connectivity with test records
- [ ] 20.5 Verify QSL module imports with test EA compilation
- [ ] 20.6 Verify PropCommander Kelly Filter with test proposals
- [ ] 20.7 Verify PropGovernor Quadratic Throttle with test scenarios
- [ ] 20.8 Verify LangGraph agent workflows with test executions
- [ ] 20.9 Verify LangMem memory operations (semantic, episodic, procedural)
- [ ] 20.10 Verify backward compatibility with existing QuantMind_Risk.mqh
- [ ] 20.11 Run complete test suite and verify all tests pass
- [ ] 20.12 Perform end-to-end system validation

## Task Group 21: Deployment and Production Readiness

- [ ] 21.1 Create production configuration templates
- [ ] 21.2 Set up environment variable management
- [ ] 21.3 Configure AsyncPostgresStore for production
- [ ] 21.4 Set up database backup and recovery procedures
- [ ] 21.5 Configure logging for production environment
- [ ] 21.6 Set up monitoring and alerting
- [ ] 21.7 Create deployment scripts and procedures
- [ ] 21.8 Create rollback procedures
- [ ] 21.9 Perform security audit
- [ ] 21.10 Create production deployment documentation
- [ ] 21.11 Conduct final production readiness review

## Notes

- All property tests should be tagged with: **Feature: quantmindx-unified-backend, Property {number}: {property_text}**
- Minimum 100 iterations per property test
- Use Hypothesis for Python property-based testing
- Follow the dual testing approach: unit tests for specific cases, property tests for universal guarantees
- Maintain backward compatibility during migration period
- Document all breaking changes and migration steps
