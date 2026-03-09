# New Chat Feature - Technical Specification

**Date:** 2026-03-09
**Status:** Draft
**Feature:** Unified Chat System with Per-Agent Endpoints

---

## 1. Overview

### 1.1 Purpose

Replace the current fragmented chat system with a unified, persistent chat architecture that supports multiple agents (Workshop Copilot, Floor Manager, Departments) with server-side session persistence and full integration with the system's capabilities (memory, tools, skills, workflows, MCP, file access).

### 1.2 Scope

- **Backend:** New chat session service with SQLite persistence
- **API:** Per-agent endpoints for each department
- **Frontend:** Enhanced chat UI with agent switching
- **Integration:** Memory, Tools, Skills, Workflows, MCP, File Access

---

## 2. Architecture

### 2.1 Current State

| Component | Current Implementation |
|-----------|----------------------|
| Workshop Copilot | `/api/workshop/copilot/chat` (exists) |
| Floor Manager | `/api/floor-manager/chat` (exists) |
| Departments | No dedicated chat endpoints |
| Persistence | Client-side localStorage only |
| Memory | Not integrated with chat |

### 2.2 Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Svelte)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ChatPanel   │  │ AgentSwitch │  │ SessionHistory      │ │
│  │             │  │             │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ /api/chat/session/{session_id}                       │  │
│  │ /api/chat/workshop/*                                │  │
│  │ /api/chat/floor-manager/*                           │  │
│  │ /api/chat/departments/{dept}/*                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Chat Session Service                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ SessionMgr  │  │ MessageMgr  │  │ ContextBuilder  │   │
│  └─────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Integration Layer                           │
│  ┌────────┐ ┌──────┐ ┌───────┐ ┌────────┐ ┌────────────┐  │
│  │Memory  │ │Tools  │ │Skills │ │Workflow│ │ MCP       │  │
│  │Facade  │ │Registry│ │Manager│ │Orchestr│ │ Servers   │  │
│  └────────┘ └───────┘ └───────┘ └────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              SQLite Database (chat_sessions)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema

### 3.1 Tables

```sql
-- Chat sessions (one per conversation thread)
CREATE TABLE chat_sessions (
    id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,           -- 'workshop', 'floor-manager', 'department'
    agent_id TEXT NOT NULL,             -- 'copilot', 'research', 'development', etc.
    title TEXT,
    user_id TEXT,
    context JSON,                        -- {project_id, file_context, etc.}
    metadata JSON,                       -- {tags, pinned, etc.}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP
);

-- Individual messages in a session
CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,                  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    artifacts JSON,                      -- Generated code, files, etc.
    tool_calls JSON,                     -- Tools invoked
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);

-- Indexes
CREATE INDEX idx_sessions_agent ON chat_sessions(agent_type, agent_id);
CREATE INDEX idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
```

---

## 4. API Endpoints

### 4.1 Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chat/sessions` | List user's chat sessions |
| POST | `/api/chat/sessions` | Create new chat session |
| GET | `/api/chat/sessions/{id}` | Get session with messages |
| PATCH | `/api/chat/sessions/{id}` | Update session (title, metadata) |
| DELETE | `/api/chat/sessions/{id}` | Delete session |

### 4.2 Per-Agent Chat Endpoints

| Method | Endpoint | Agent |
|--------|----------|-------|
| POST | `/api/chat/workshop/message` | Workshop Copilot |
| POST | `/api/chat/floor-manager/message` | Floor Manager |
| POST | `/api/chat/departments/{dept}/message` | Department |

**Department Values:** `research`, `development`, `risk`, `trading`, `portfolio`

### 4.3 Request/Response Models

```python
# POST /api/chat/sessions
class CreateSessionRequest(BaseModel):
    agent_type: str              # 'workshop', 'floor-manager', 'department'
    agent_id: str                # Specific agent identifier
    title: Optional[str]         # Optional session title
    context: Optional[dict]      # {project_id, file_path, etc.}

class CreateSessionResponse(BaseModel):
    session_id: str
    agent_type: str
    agent_id: str
    title: str
    created_at: datetime

# POST /api/chat/{workshop|floor-manager|departments/{dept}}/message
class ChatMessageRequest(BaseModel):
    message: str                 # User message
    session_id: Optional[str]   # Existing session or None for new
    stream: bool = False         # Stream response

class ChatMessageResponse(BaseModel):
    session_id: str
    message_id: str
    reply: str                   # Assistant response
    artifacts: List[dict]        # Generated content
    action_taken: Optional[str] # Tool/workflow executed
    delegation: Optional[str]   # If delegated to another agent
    context: Optional[dict]      # Updated context
```

---

## 5. Integration Points

### 5.1 Memory Integration

```python
class ChatMemoryService:
    """Integrates chat with unified memory facade"""

    async def store_interaction(
        session_id: str,
        user_message: str,
        assistant_response: str,
        context: dict
    ):
        """Store chat interaction in memory"""
        # Use unified_memory_facade.store()
        pass

    async def retrieve_context(
        session_id: str,
        query: str
    ) -> List[dict]:
        """Retrieve relevant context from memory"""
        # Use unified_memory_facade.search()
        pass

    async def build_session_context(
        session_id: str
    ) -> dict:
        """Build context for new message from session history"""
        # Retrieve recent messages + relevant memories
        pass
```

### 5.2 Tool Registry Integration

```python
class ChatToolService:
    """Handles tool invocation from chat"""

    async def execute_tool(
        tool_name: str,
        parameters: dict,
        agent_id: str
    ) -> ToolResult:
        """Execute tool with permission check"""
        # Check tool_registry permissions for agent
        pass

    async def list_available_tools(agent_id: str) -> List[dict]:
        """List tools available to agent"""
        # Query tool_registry for agent permissions
        pass
```

### 5.3 Skill Manager Integration

```python
class ChatSkillService:
    """Handles skill execution from chat"""

    async def execute_skill(
        skill_name: str,
        parameters: dict,
        context: dict
    ) -> SkillResult:
        """Execute skill with full context"""
        # Use skill_manager.execute()
        pass

    async def list_skills(agent_id: str) -> List[dict]:
        """List skills available to agent"""
        pass
```

### 5.4 Workflow Integration

```python
class ChatWorkflowService:
    """Handles workflow triggers from chat"""

    async def trigger_workflow(
        workflow_name: str,
        parameters: dict,
        context: dict
    ) -> WorkflowResult:
        """Trigger department workflow"""
        # Use workflow orchestrator
        pass

    async def get_workflow_status(task_id: str) -> dict:
        """Get workflow execution status"""
        pass
```

### 5.5 MCP Integration

```python
class ChatMCPService:
    """Handles MCP server communication"""

    async def call_mcp_tool(
        server: str,
        tool: str,
        parameters: dict
    ) -> Any:
        """Call MCP tool"""
        pass

    async def list_mcp_tools(agent_id: str) -> List[dict]:
        """List MCP tools available to agent"""
        pass
```

### 5.6 File Access Integration

```python
class ChatFileService:
    """Handles file operations from chat"""

    async def read_file(self, path: str, agent_id: str) -> str:
        """Read file with permission check"""
        pass

    async def write_file(self, path: str, content: str, agent_id: str) -> bool:
        """Write file with permission check"""
        pass

    async def search_files(self, query: str, agent_id: str) -> List[dict]:
        """Search files in workspace"""
        pass
```

---

## 6. Backend Implementation

### 6.1 Service Structure

```
src/api/services/
├── chat_session_service.py     # Main chat service
├── chat_memory_integration.py  # Memory integration
├── chat_tool_integration.py    # Tool execution
├── chat_skill_integration.py   # Skill execution
├── chat_workflow_integration.py # Workflow triggers
├── chat_mcp_integration.py     # MCP communication
└── chat_file_integration.py    # File operations
```

### 6.2 Core Service Class

```python
# src/api/services/chat_session_service.py
class ChatSessionService:
    """Main service for chat session management"""

    def __init__(self, db_path: str):
        self.db = Database(db_path)
        self.memory = get_memory_facade()
        self.tool_registry = get_tool_registry()
        self.skill_manager = get_skill_manager()
        self.workflow_orchestrator = get_workflow_orchestrator()
        self.mcp_manager = get_mcp_manager()
        self.file_service = get_file_service()

    # Session CRUD
    async def create_session(...) -> ChatSession: ...
    async def get_session(session_id) -> ChatSession: ...
    async def list_sessions(user_id, agent_type) -> List[ChatSession]: ...
    async def update_session(session_id, data) -> ChatSession: ...
    async def delete_session(session_id) -> None: ...

    # Message handling
    async def process_message(
        message: str,
        session_id: Optional[str],
        agent_type: str,
        agent_id: str,
        context: dict
    ) -> ChatResponse:
        # 1. Build context (memory + session history)
        # 2. Route to appropriate agent handler
        # 3. Execute tools/skills as needed
        # 4. Store in database
        # 5. Store in memory
        pass
```

---

## 7. Frontend Implementation

### 7.1 Store Updates

```typescript
// quantmind-ide/src/lib/stores/chatStore.ts

export interface ChatSession {
  id: string;
  agentType: 'workshop' | 'floor-manager' | 'department';
  agentId: string;
  title: string;
  messages: Message[];
  context: Record<string, any>;
  metadata: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  artifacts?: any[];
  toolCalls?: any[];
  createdAt: Date;
}

// Store actions
export const chatActions = {
  // Server sync
  async loadSessions(): Promise<ChatSession[]>;
  async createSession(agentType, agentId): Promise<ChatSession>;
  async sendMessage(sessionId, message): Promise<Message>;
  async deleteSession(sessionId): Promise<void>;

  // Local state
  setActiveSession(sessionId: string): void;
  addMessage(message: Message): void;
};
```

### 7.2 Component Structure

```
quantmind-ide/src/lib/components/
├── Chat/
│   ├── ChatPanel.svelte          # Main chat container
│   ├── ChatMessage.svelte        # Single message
│   ├── ChatInput.svelte          # Message input
│   ├── AgentSwitcher.svelte      # Agent dropdown
│   ├── SessionList.svelte        # Session sidebar
│   └── SessionItem.svelte        # Single session in list
```

### 7.3 API Client

```typescript
// quantmind-ide/src/lib/api/chatApi.ts

export const chatApi = {
  // Sessions
  async getSessions(agentType?: string): Promise<ChatSession[]>;
  async createSession(data: CreateSessionRequest): Promise<ChatSession>;
  async getSession(id: string): Promise<ChatSession>;
  async updateSession(id: string, data: UpdateSessionRequest): Promise<ChatSession>;
  async deleteSession(id: string): Promise<void>;

  // Messages
  async sendMessage(
    endpoint: string,
    message: string,
    sessionId?: string,
    stream?: boolean
  ): Promise<ChatMessageResponse | AsyncIterable<ChatMessageResponse>>;
};
```

---

## 8. Agent Routing

### 8.1 Intent Classification

```python
# Route message to appropriate agent
async def route_message(message: str, context: dict) -> AgentRoute:
    """
    Determine which agent should handle the message
    """
    # Workshop Copilot handles: simple queries, workflow requests
    # Floor Manager handles: trading decisions, execution
    # Department handles: specialized tasks

    intent = await classify_intent(message)

    if intent in ['simple_query', 'help', 'explanation']:
        return AgentRoute(type='workshop', agent_id='copilot')
    elif intent in ['trade_execution', 'order_management']:
        return AgentRoute(type='floor-manager', agent_id='floor-manager')
    elif intent in ['research', 'analysis']:
        return AgentRoute(type='department', agent_id='research')
    # ... etc
```

### 8.2 Delegation Flow

```
User Message
     │
     ▼
┌─────────────┐
│  Workshop   │
│  Copilot    │
└──────┬──────┘
       │
       ▼ (delegation)
┌─────────────┐     ┌─────────────┐
│   Floor     │────►│ Department  │
│   Manager   │     │             │
└─────────────┘     └─────────────┘
```

---

## 9. Security & Permissions

### 9.1 Agent Permissions

```python
# Each agent has scoped permissions
AGENT_PERMISSIONS = {
    'copilot': {
        'tools': ['read_file', 'search', 'list_skills'],
        'departments': [],
        'memory_read': True,
        'memory_write': True,
    },
    'floor-manager': {
        'tools': ['*'],  # All tools
        'departments': ['trading', 'risk'],
        'memory_read': True,
        'memory_write': True,
    },
    'research': {
        'tools': ['read_file', 'search', 'run_analysis'],
        'departments': [],
        'memory_read': True,
        'memory_write': True,
    },
}
```

### 9.2 Tool Permission Check

```python
async def check_tool_permission(agent_id: str, tool_name: str) -> bool:
    """Verify agent can use tool"""
    permissions = AGENT_PERMISSIONS.get(agent_id, {})
    allowed_tools = permissions.get('tools', [])

    if '*' in allowed_tools:
        return True
    return tool_name in allowed_tools
```

---

## 10. Implementation Phases

### Phase 1: Backend Foundation (Week 1)

- [ ] Create database schema
- [ ] Implement ChatSessionService
- [ ] Create session CRUD endpoints
- [ ] Add message persistence

### Phase 2: Agent Endpoints (Week 2)

- [ ] Create `/api/chat/workshop/*` endpoints
- [ ] Create `/api/chat/floor-manager/*` endpoints
- [ ] Create `/api/chat/departments/{dept}/*` endpoints
- [ ] Implement intent routing

### Phase 3: Integration (Week 2-3)

- [ ] Integrate memory facade
- [ ] Integrate tool registry
- [ ] Integrate skill manager
- [ ] Integrate workflow orchestrator
- [ ] Integrate MCP servers

### Phase 4: Frontend (Week 3-4)

- [ ] Update chatStore with server sync
- [ ] Create SessionList component
- [ ] Create AgentSwitcher component
- [ ] Add streaming support
- [ ] Update ChatPanel

### Phase 5: Testing & Polish (Week 4)

- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Error handling
- [ ] Documentation

---

## 11. Acceptance Criteria

### Backend

- [ ] Sessions persist across browser refreshes
- [ ] Each department has dedicated chat endpoint
- [ ] Messages stored with full metadata
- [ ] Memory integration provides context
- [ ] Tools execute with permission checks

### Frontend

- [ ] User can switch between agents
- [ ] Session list shows all chats
- [ ] Messages load from server
- [ ] Streaming responses work
- [ ] Offline graceful degradation

### Integration

- [ ] Memory stores and retrieves context
- [ ] Tools execute correctly
- [ ] Skills can be invoked
- [ ] Workflows can be triggered
- [ ] File operations work

---

## 12. Related Files

- `src/api/workshop_copilot_endpoints.py` - Existing workshop endpoints
- `src/api/floor_manager_endpoints.py` - Existing floor manager endpoints
- `src/agents/departments/tool_registry.py` - Tool permissions
- `src/agents/memory/unified_memory_facade.py` - Memory integration
- `src/agents/skills/skill_manager.py` - Skill execution
- `quantmind-ide/src/lib/stores/chatStore.ts` - Frontend state
