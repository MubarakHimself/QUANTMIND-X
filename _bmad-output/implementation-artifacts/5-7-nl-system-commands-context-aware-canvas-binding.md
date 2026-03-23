# Story 5.7: NL System Commands & Context-Aware Canvas Binding

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader controlling the system through Copilot,
I want to issue system commands in natural language with canvas context automatically applied,
So that I can operate the platform conversationally regardless of which canvas I am on.

## Acceptance Criteria

**Given** I type "pause GBPUSD strategy" in Copilot,
**When** FloorManager interprets the command,
**Then** it identifies intent as `STRATEGY_PAUSE`, confirms: "I'll pause the GBPUSD strategy. Confirm?",
**And** on confirmation, executes the pause via the risk API.

**Given** I am on the Live Trading canvas,
**When** I ask "what are my open positions?",
**Then** FloorManager resolves the question against live trading data (canvas context binding).

**Given** the command is ambiguous,
**When** FloorManager cannot determine intent with high confidence,
**Then** it asks a clarifying question rather than executing blindly.

## Tasks / Subtasks

- [x] Task 1 (AC: #1 - NL Command Classification)
  - [x] Subtask 1.1: Extend FloorManager with intent classification module
  - [x] Subtask 1.2: Implement command patterns: STRATEGY_PAUSE, STRATEGY_RESUME, POSITION_CLOSE, POSITION_INFO, etc.
  - [x] Subtask 1.3: Add confirmation flow for destructive commands
- [x] Task 2 (AC: #1 - Canvas Context Binding)
  - [x] Subtask 2.1: Integrate CanvasContextTemplate loading per canvas
  - [x] Subtask 2.2: Pass canvas context as metadata to FloorManager
  - [x] Subtask 2.3: Implement context-aware data resolution per canvas type
- [x] Task 3 (AC: #2 - Context-Aware Queries)
  - [x] Subtask 3.1: Bind live trading data when on Live Trading canvas
  - [x] Subtask 3.2: Bind risk data when on Risk canvas
  - [x] Subtask 3.3: Bind portfolio data when on Portfolio canvas
- [x] Task 4 (AC: #3 - Ambiguity Handling)
  - [x] Subtask 4.1: Implement confidence scoring for intent classification
  - [x] Subtask 4.2: Add clarifying question flow when confidence < threshold
  - [x] Subtask 4.3: Add fallback to general search when canvas-specific resolution fails

## Dev Notes

### Previous Story Intelligence (Story 5.6 - Copilot Kill Switch)

Story 5.6 implemented the Copilot Kill Switch:
- **CopilotKillSwitch** at `src/router/copilot_kill_switch.py`:
  - Has `activate()`, `resume()`, `get_status()` methods
  - Independently controls agent/AI activity (separate from Trading Kill Switch)
- **FloorManager** at `src/agents/departments/floor_manager.py`:
  - Has `chat()` and `chat_stream()` methods
  - Integrated with kill switch via `_check_kill_switch()` method
  - Returns "suspended" status when kill switch active

**Key Learnings from Story 5.6:**
- FloorManager accepts `canvas_context` in message metadata
- Streaming responses use SSE with line buffering
- Agent Panel shows status messages with color coding (amber for suspended)

### Previous Story Intelligence (Story 5.3 - Canvas Context System)

Story 5.3 implemented the CanvasContextTemplate system:
- **CanvasContextTemplate** schema defined in architecture
- Templates at `src/canvas_context/templates/` (YAML files per canvas)
- `context_loader.py` loads templates on canvas/session start
- Template includes: `memory_scope`, `workflow_namespaces`, `skill_index`, `required_tools`

**Key Learnings from Story 5.3:**
- Canvas context includes: `canvas` identifier, `session_id`, `entity` (if department-specific)
- Templates define what data to fetch, not the data itself
- Session type is always "interactive" for canvas sessions

### Architecture Prerequisites (CRITICAL)

1. **CanvasContextTemplate** must be loaded on session start
   - Each canvas has a template: `live_trading.yaml`, `risk.yaml`, `portfolio.yaml`, etc.
   - Template loaded when user starts a new chat on a canvas

2. **FloorManager intent classification**:
   - Input: natural language message + canvas_context metadata
   - Output: intent classification + entities + confidence score
   - Supported intents: STRATEGY_PAUSE, STRATEGY_RESUME, POSITION_CLOSE, POSITION_INFO, REGIME_QUERY, etc.

3. **Command confirmation flow**:
   - Destructive commands (pause, close, stop) require explicit confirmation
   - Confirmation message shows: "I'll [action] [entity]. Confirm?"
   - On confirm: execute via appropriate API
   - On cancel: return to conversation

4. **Canvas context binding**:
   - Live Trading canvas: positions, bots, equity, P&L
   - Risk canvas: risk params, regime, drawdown
   - Portfolio canvas: accounts, attribution, metrics
   - Workshop canvas: agent tasks, skills, memory

### Backend Implementation Details

**FloorManager intent classification extension:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class CommandIntent(Enum):
    STRATEGY_PAUSE = "strategy_pause"
    STRATEGY_RESUME = "strategy_resume"
    POSITION_CLOSE = "position_close"
    POSITION_INFO = "position_info"
    REGIME_QUERY = "regime_query"
    ACCOUNT_INFO = "account_info"
    GENERAL_QUERY = "general_query"
    CLARIFICATION_NEEDED = "clarification_needed"

@dataclass
class IntentClassification:
    intent: CommandIntent
    entities: List[str]  # ["GBPUSD", "EURUSD"] or ["all"]
    confidence: float  # 0.0 - 1.0
    raw_command: str
    requires_confirmation: bool

class FloorManager:
    # ... existing code ...

    async def classify_intent(self, message: str, canvas_context: Dict[str, Any]) -> IntentClassification:
        """Classify user message into actionable intent."""
        # 1. Extract canvas type from context
        canvas_type = canvas_context.get("canvas", "workshop")

        # 2. Build prompt with canvas-specific context
        system_prompt = self._build_intent_prompt(canvas_type)

        # 3. Use LLM to classify intent
        # OR use pattern matching for known commands

        # 4. Return classification with confidence score
        return IntentClassification(...)

    async def handle_command(self, message: str, canvas_context: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
        """Handle natural language command with confirmation flow."""
        # 1. Classify intent
        classification = await self.classify_intent(message, canvas_context)

        # 2. If confidence < threshold, ask clarification
        if classification.confidence < 0.7:
            return {
                "type": "clarification_needed",
                "message": "I'm not sure what you mean. Could you clarify?",
                "suggestions": ["pause all strategies", "show my positions", "what is the current regime?"]
            }

        # 3. If requires_confirmation and not confirmed, ask for confirmation
        if classification.requires_confirmation and not confirmed:
            return {
                "type": "confirmation_needed",
                "message": f"I'll {classification.intent.value.replace('_', ' ')} {', '.join(classification.entities)}. Confirm?",
                "intent": classification.intent.value,
                "entities": classification.entities
            }

        # 4. Execute command
        return await self._execute_command(classification, canvas_context)
```

**Canvas context binding:**

```python
class CanvasContextBinder:
    """Bind canvas-specific data to queries."""

    def __init__(self):
        self._binders = {
            "live_trading": LiveTradingBinder(),
            "risk": RiskBinder(),
            "portfolio": PortfolioBinder(),
            "workshop": WorkshopBinder(),
        }

    async def bind_context(self, canvas_type: str, query: str) -> Dict[str, Any]:
        """Bind canvas-specific data to enhance query understanding."""
        binder = self._binders.get(canvas_type)
        if not binder:
            return {}

        # Fetch canvas-specific data
        context_data = await binder.fetch_context()
        return {
            "canvas_type": canvas_type,
            "context_data": context_data,
            "query": query
        }

class LiveTradingBinder:
    async def fetch_context(self) -> Dict[str, Any]:
        """Fetch live trading context."""
        # Fetch positions, bots, equity, P&L
        # Return structured context for FloorManager
```

### Frontend Implementation Details

**CopilotPanel modifications for context-aware commands:**

```svelte
<script>
  import { canvasContextStore } from '$lib/stores/canvas';

  let pendingConfirmation = $state(null);

  async function handleMessage(message, confirmed = false) {
    const canvasContext = $canvasContextStore;

    const response = await fetch('/api/floor-manager/command', {
      method: 'POST',
      body: JSON.stringify({
        message,
        canvas_context: canvasContext,
        confirmed
      })
    });

    const data = await response.json();

    if (data.type === 'confirmation_needed') {
      // Show confirmation dialog
      pendingConfirmation = {
        message: data.message,
        intent: data.intent,
        entities: data.entities
      };
    } else if (data.type === 'clarification_needed') {
      // Show suggestions
      showSuggestions(data.suggestions);
    } else {
      // Stream response as normal
      streamResponse(data);
    }
  }

  async function confirmAction() {
    if (pendingConfirmation) {
      await handleMessage(pendingConfirmation.message, true);
      pendingConfirmation = null;
    }
  }
</script>

<!-- Confirmation dialog -->
{#if pendingConfirmation}
  <div class="confirmation-dialog">
    <p>{pendingConfirmation.message}</p>
    <button onclick={confirmAction}>Confirm</button>
    <button onclick={() => pendingConfirmation = null}>Cancel</button>
  </div>
{/if}
```

**Canvas context store:**

```typescript
// quantmind-ide/src/lib/stores/canvas.ts
import { writable } from 'svelte/store';

interface CanvasContext {
  canvas: 'live_trading' | 'risk' | 'portfolio' | 'workshop' | 'research' | 'development';
  session_id: string;
  entity?: string;
}

export const canvasContextStore = writable<CanvasContext>({
  canvas: 'workshop',
  session_id: ''
});
```

### File Structure Requirements

```
src/
├── agents/
│   └── departments/
│       └── floor_manager.py           [MODIFY] - Add intent classification
├── canvas_context/
│   ├── templates/                     [EXISTS from Story 5.3]
│   │   ├── live_trading.yaml
│   │   ├── risk.yaml
│   │   ├── portfolio.yaml
│   │   └── workshop.yaml
│   └── context_loader.py              [EXISTS from Story 5.3]
├── api/
│   └── floor_manager_endpoints.py     [MODIFY] - Add /command endpoint

src/intent/                            [NEW]
├── __init__.py
├── classifier.py                       [NEW] - Intent classification
├── patterns.py                         [NEW] - Command pattern matching
└── binders/                            [NEW]
    ├── __init__.py
    ├── base.py                        [NEW]
    ├── live_trading.py                [NEW]
    ├── risk.py                        [NEW]
    └── portfolio.py                  [NEW]

quantmind-ide/src/lib/
├── stores/
│   └── canvas.ts                      [NEW] - Canvas context store
├── components/
│   └── trading-floor/
│       └── CopilotPanel.svelte        [MODIFY] - Add confirmation UI
└── services/
    └── intentService.ts               [NEW] - Frontend intent handling

tests/
├── intent/
│   ├── test_classifier.py             [NEW]
│   └── test_patterns.py              [NEW]
└── api/
    └── test_floor_manager_commands.py [NEW]
```

### Testing Requirements

1. **Intent Classification Tests:**
   - Test STRATEGY_PAUSE classification for "pause GBPUSD strategy"
   - Test STRATEGY_RESUME classification for "resume trading"
   - Test POSITION_INFO for "show my positions"
   - Test confidence scoring accuracy

2. **Confirmation Flow Tests:**
   - Test destructive commands require confirmation
   - Test confirmation execution
   - Test cancellation returns to conversation

3. **Canvas Context Tests:**
   - Test live trading context binding
   - Test risk context binding
   - Test portfolio context binding
   - Test fallback to general query when canvas-specific fails

4. **Ambiguity Handling Tests:**
   - Test low confidence triggers clarification
   - Test clarification suggestions are relevant
   - Test multiple clarification rounds

5. **Integration Tests:**
   - Test command execution via appropriate APIs
   - Test audit logging of commands
   - Test kill switch integration (commands blocked when active)

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§6.2 Canvas Context System, §10 Department System)
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.7)
- Previous story (Canvas Context): `_bmad-output/implementation-artifacts/5-3-canvas-context-system-canvascontexttemplate-per-department.md`
- Previous story (Kill Switch): `_bmad-output/implementation-artifacts/5-6-copilot-kill-switch.md`
- FloorManager: `src/agents/departments/floor_manager.py`
- CanvasContextTemplate: `src/canvas_context/templates/`
- CopilotPanel: `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte`
- Project Context: `_bmad-output/project-context.md`

### NFR Requirements

- **FR20**: Copilot operates context-aware on any canvas
- **FR46**: All agent decisions log reasoning for transparency
- **NFR-P3**: Copilot responses begin streaming within 5 seconds
- Canvas context passed as metadata: `{ message, canvas_context: "live-trading", session_id }`
- All destructive commands require in-conversation confirmation

### Technical Notes

1. **Intent Classification Approach:**
   - Use pattern matching for known commands (fast path)
   - Use LLM for ambiguous/natural language (accuracy path)
   - Confidence threshold: 0.7 (below = clarification)

2. **Canvas Context Binding:**
   - CanvasContextTemplate loaded on session start
   - Context binder fetches current data at query time
   - Data cached briefly (5s) to avoid excessive API calls

3. **Command Execution:**
   - Strategy pause/resume: via risk API
   - Position close: via trading API
   - Position info: via trading/positions API
   - All commands logged to audit trail

4. **Kill Switch Integration:**
   - Commands blocked when Copilot Kill Switch active
   - Return "Agent activity suspended" status

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

- Backend: `src/intent/` - Created intent classifier module
- API: `src/api/floor_manager_endpoints.py` - Added /command endpoint
- Frontend: `quantmind-ide/src/lib/stores/canvas.ts` - Created canvas context store
- Frontend: `quantmind-ide/src/lib/services/intentService.ts` - Created intent service
- Frontend: `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` - Added confirmation UI

### Completion Notes List

- Implemented Task 1 (NL Command Classification):
  - Created intent classifier module at `src/intent/`
  - Implemented pattern matching for commands: STRATEGY_PAUSE, STRATEGY_RESUME, POSITION_CLOSE, POSITION_INFO, REGIME_QUERY, ACCOUNT_INFO
  - Added confidence scoring with 0.7 threshold
  - Added confirmation flow for destructive commands
  - Extended FloorManager with classify_intent() and handle_command() methods

- Implemented Task 2 (Canvas Context Binding):
  - Integrated canvas context binders (LiveTrading, Risk, Portfolio, Workshop)
  - Canvas context passed via context dictionary to FloorManager
  - Context-aware data resolution implemented per canvas type

- Implemented Task 3 (Context-Aware Queries):
  - LiveTradingBinder: positions, bots, equity data
  - RiskBinder: regime, risk params, drawdown data
  - PortfolioBinder: accounts, attribution, metrics
  - WorkshopBinder: tasks, skills, memory info

- Implemented Task 4 (Ambiguity Handling):
  - Confidence scoring: 0.7 threshold for actionable commands
  - Clarification flow: Returns clarification_needed type with suggestions
  - Fallback: Returns GENERAL_QUERY for non-matching commands

### File List

- src/intent/__init__.py [NEW]
- src/intent/classifier.py [NEW]
- src/intent/patterns.py [NEW]
- src/intent/binders/__init__.py [NEW]
- src/intent/binders/base.py [NEW]
- src/intent/binders/live_trading.py [NEW]
- src/intent/binders/risk.py [NEW]
- src/intent/binders/portfolio.py [NEW]
- src/intent/binders/workshop.py [NEW]
- src/agents/departments/floor_manager.py [MODIFIED]
- src/api/floor_manager_endpoints.py [MODIFIED] - Added /command endpoint
- tests/intent/test_classifier.py [NEW]
- quantmind-ide/src/lib/stores/canvas.ts [NEW]
- quantmind-ide/src/lib/services/intentService.ts [NEW]
- quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte [MODIFIED] - Added confirmation UI
