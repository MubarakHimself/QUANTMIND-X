# Story 7.2: Development Department — Real MQL5 EA Code Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

**As a** strategy developer,
**I want** the Development Department Head to generate real MQL5 EA code from TRD documents,
**So that** the strategy factory pipeline produces actual tradeable EA files.

## Acceptance Criteria

1. **Given** a validated TRD document is passed to Development,
   **When** the Development Department Head processes it,
   **Then** it generates a complete MQL5 `.mq5` file with: EA header, `OnInit()`, `OnTick()`, `OnDeinit()`, and all parameters from the TRD (session_mask, force_close_hour, overnight_hold, daily_loss_cap, spread_filter, etc.).

2. **Given** the TRD is ambiguous on a parameter,
   **When** the Development agent encounters ambiguity,
   **Then** it flags it and asks FloorManager for clarification rather than guessing.

3. **Given** the EA file is generated,
   **When** it is saved,
   **Then** it is stored in the EA output directory with a version number,
   **And** a compile trigger fires (Story 7.3).

## Tasks / Subtasks

- [x] Task 1: Implement TRD document parsing and validation (AC: 1)
  - [x] Subtask 1.1: Define TRD document schema with all required fields
  - [x] Subtask 1.2: Implement TRD parser/validator in DevelopmentHead
  - [x] Subtask 1.3: Handle missing/ambiguous parameter detection
- [x] Task 2: Implement MQL5 EA code generation (AC: 1)
  - [x] Subtask 2.1: Create MQL5 template with OnInit, OnTick, OnDeinit
  - [x] Subtask 2.2: Map TRD parameters to EA input variables
  - [x] Subtask 2.3: Generate complete .mq5 file with proper MQL5 syntax
- [x] Task 3: Implement ambiguity handling and clarification flow (AC: 2)
  - [x] Subtask 3.1: Add parameter ambiguity detection logic
  - [x] Subtask 3.2: Integrate with FloorManager for clarification requests
- [x] Task 4: Implement EA storage and versioning (AC: 3)
  - [x] Subtask 4.1: Create EA output directory structure
  - [x] Subtask 4.2: Implement version numbering scheme
  - [x] Subtask 4.3: Add compile trigger for Story 7.3
- [x] Task 5: Integration testing
  - [x] Subtask 5.1: End-to-end test with FloorManager delegation
  - [x] Subtask 5.2: Verify EA output is valid MQL5

## Dev Notes

### Project Structure Notes

- **Epic Context**: Epic 7 is "in-progress" — Department Agent Platform where departments do real work
- **Previous Story**: Story 7.1 (Research Department Real Hypothesis Generation) is in review — provides context on department handoff
- **Dependencies**: Story 7.3 (MQL5 Compilation) depends on this story
- **FR Coverage**: FR25 — MQL5 EA code generation
- **NFR**: NFR-M3: Python files under 500 lines

### Source Tree Components to Touch

```
src/agents/departments/
├── heads/
│   ├── development_head.py          # MAIN IMPLEMENTATION TARGET - replace stub with full impl
│   ├── base.py                     # DepartmentHead base class
│   └── research_head.py            # Reference for pattern (just completed)
├── floor_manager.py                 # Task delegation (delegate_to_department)
src/mql5/
├── templates/                       # MQL5 EA templates (new directory)
│   ├── ea_base_template.mq5       # Base EA template
│   └── ea_parameter_mapping.py    # TRD to MQL5 param mapping
src/trd/
├── parser.py                        # TRD document parser (new)
├── validator.py                     # TRD validator (new)
├── schema.py                       # TRD schema definition (new)
src/strategy/
├── output.py                        # EA output storage with versioning (new)
tools/
tests/agents/departments/heads/
├── test_development_head.py         # Unit tests (new)
└── test_mql5_generation.py          # MQL5 generation tests (new)
```

### Architecture Patterns

1. **DepartmentHead Base Pattern** (from `heads/base.py`):
   - Each department head inherits from `DepartmentHead`
   - Implements `process_task()`, `can_handle()`, etc.
   - Uses department mail for cross-dept communication

2. **Development Sub-Agent Types** (from `types.py`):
   - `PYTHON_DEV`: Python-based strategy implementation
   - `PINESCRIPT_DEV`: TradingView PineScript development
   - `MQL5_DEV`: MetaTrader 5 EA development (primary focus for this story)

3. **TRD Document Schema** (to be defined):
   ```python
   {
     "strategy_id": str,              # Unique identifier
     "strategy_name": str,
     "symbol": str,                   # e.g., "EURUSD"
     "timeframe": str,                # e.g., "H4", "D1"
     "entry_conditions": list[str],   # Entry rule descriptions
     "exit_conditions": list[str],    # Exit rule descriptions
     "position_sizing": {
       "method": str,                 # "fixed_lot", "dynamic", etc.
       "risk_percent": float,
       "max_lots": float
     },
     "parameters": {
       "session_mask": str,           # Trading session mask
       "force_close_hour": int,       # Hour to force close
       "overnight_hold": bool,        # Hold over night
       "daily_loss_cap": float,       # Daily loss limit
       "spread_filter": float,        # Max spread to trade
       # ... additional TRD-specific parameters
     },
     "version": int,
     "created_at": datetime
   }
   ```

4. **MQL5 EA Output Requirements**:
   - Must include EA header with copyright, version, description
   - `OnInit()`: Initialize indicators, variables, inputs
   - `OnTick()`: Main trading logic loop
   - `OnDeinit()`: Cleanup on removal
   - Input parameters mapped from TRD
   - Trading session logic (session_mask)
   - Risk management (daily_loss_cap)
   - Spread filtering

5. **First Version Rule** (from architecture):
   - First EA from a video source must mirror the source strategy
   - Vanilla variant = mirror image
   - Spiced variants = system-native adaptations

6. **Department Mail Pattern**:
   - Used for cross-department communication
   - Clarification requests go to FloorManager
   - Handoff to Compilation (Story 7.3) via department mail

### Key Technical Details

- **EA Output Directory**: `{data_dir}/strategies/{strategy_id}/versions/{version}/`
- **Version Scheme**: Incremental integer starting at 1
- **File Naming**: `{strategy_name}_{version}.mq5`
- **Compilation Trigger**: Automatic on successful generation (Story 7.3 dependency)

- **Ambiguity Handling**:
  - Flag parameters that are missing or unclear
  - Return clarification request to FloorManager
  - Do NOT guess parameter values

- **Parameter Validation**:
  - session_mask: Valid session string (e.g., "UK/US")
  - force_close_hour: 0-23 integer
  - overnight_hold: boolean
  - daily_loss_cap: Positive float
  - spread_filter: Non-negative float

### Testing Standards

- Unit tests for TRD parsing and validation
- Unit tests for MQL5 code generation (syntax validation)
- Integration test with FloorManager delegation
- Test ambiguity detection and clarification flow
- Test EA storage and versioning
- Verify generated MQL5 compiles (Story 7.3 integration)

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.2]
- [Source: _bmad-output/implementation-artifacts/7-1-research-department-real-hypothesis-generation.md]
- [Source: _bmad-output/planning-artifacts/architecture.md#Department-Agent-Platform]
- [Source: src/agents/departments/heads/development_head.py]
- [Source: src/agents/departments/heads/research_head.py]
- [Source: src/agents/departments/types.py]
- [Source: _bmad-output/project-context.md]

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

- Story 7.1 Research Department provides hypothesis to Development via TRD
- FloorManager delegates TRD to Development via department mail
- Development generates MQL5 EA code and triggers compilation

### Completion Notes List

- Created TRD module (src/trd/) with schema, parser, and validator
- Created MQL5 module (src/mql5/) with generator and base EA template
- Created Strategy output module (src/strategy/) for EA storage with versioning
- Implemented full TRD parsing and validation workflow
- Implemented MQL5 code generation with proper syntax
- Implemented ambiguity detection for parameter clarification
- Implemented EA storage with version numbering
- Created comprehensive unit tests (15 tests passing)
- Generated MQL5 code includes: OnInit, OnTick, OnDeinit, session management, risk controls

### File List

New files created:
- src/trd/__init__.py
- src/trd/schema.py
- src/trd/parser.py
- src/trd/validator.py
- src/mql5/__init__.py
- src/mql5/generator.py
- src/mql5/templates/__init__.py
- src/mql5/templates/ea_base_template.py
- src/strategy/__init__.py
- src/strategy/output.py
- tests/agents/departments/heads/test_development_head.py
- tests/agents/departments/heads/test_mql5_generation.py

Modified files:
- src/agents/departments/heads/development_head.py (full implementation)
