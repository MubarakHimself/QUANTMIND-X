# Story 7.3: MQL5 Compilation Integration

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

**As a** developer building the strategy factory,
**I want** the Development Department to trigger MQL5 compilation after EA code generation,
**So that** compile errors are caught immediately and the `.ex5` file is ready for backtesting.

## Acceptance Criteria

1. **Given** a new `.mq5` file is generated,
   **When** the compile step triggers,
   **Then** the MetaEditor compiler is invoked via the MT5 bridge or Docker-based MT5 compiler on Contabo,
   **And** compilation output (errors, warnings) is captured and returned.

2. **Given** compilation succeeds,
   **When** `.ex5` is produced,
   **Then** it is stored alongside `.mq5` in the EA output directory,
   **And** the strategy record updates: `compile_status: success`.

3. **Given** compilation fails,
   **When** error output is received,
   **Then** the Development Department analyses errors and attempts auto-correction (≤2 iterations),
   **And** if still failing, escalates to FloorManager with error detail.

## Tasks / Subtasks

- [x] Task 1: Implement Docker MT5 compiler integration (AC: 1)
  - [x] Subtask 1.1: Set up Docker-based MT5 compiler on Contabo
  - [x] Subtask 1.2: Create compilation API endpoint
  - [x] Subtask 1.3: Handle compilation request/response flow
- [x] Task 2: Implement successful compilation handling (AC: 2)
  - [x] Subtask 2.1: Store .ex5 alongside .mq5 in EA output directory
  - [x] Subtask 2.2: Update strategy record compile_status to success
  - [ ] Subtask 2.3: Trigger backtest pipeline (Story 7.8 integration) - DEPENDS ON STORY 7.8
- [x] Task 3: Implement error handling and auto-correction (AC: 3)
  - [x] Subtask 3.1: Parse compilation error output
  - [x] Subtask 3.2: Implement auto-correction logic (≤2 iterations)
  - [x] Subtask 3.3: Implement escalation to FloorManager
- [x] Task 4: Integration with Story 7.2
  - [x] Subtask 4.1: Wire compile trigger after EA generation (Subtask 4.3 was incomplete)
  - [x] Subtask 4.2: Test end-to-end from TRD → EA generation → compilation
- [x] Task 5: Testing
  - [x] Subtask 5.1: Unit tests for compilation service
  - [x] Subtask 5.2: Integration tests with Development Department
  - [x] Subtask 5.3: Auto-correction iteration tests

## Dev Notes

### Project Structure Notes

- **Epic Context**: Epic 7 is "in-progress" — Department Agent Platform where departments do real work
- **Previous Story**: Story 7.2 (Development Department MQL5 EA Code Generation) is in review
- **Dependencies**: Story 7.8 (Risk, Trading & Portfolio) depends on compiled .ex5 for backtesting
- **FR Coverage**: FR25 — MQL5 compilation
- **NFR**: NFR-M3: Python files under 500 lines

### Source Tree Components to Touch

```
src/agents/departments/
├── heads/
│   ├── development_head.py          # ADD: compilation trigger call
src/mql5/
├── compiler/                        # NEW: compilation module
│   ├── __init__.py
│   ├── docker_compiler.py          # Docker MT5 compiler wrapper
│   ├── error_parser.py            # Parse MQL5 compilation errors
│   └── autocorrect.py              # Auto-correction logic (≤2 iterations)
src/strategy/
├── output.py                        # ADD: .ex5 storage alongside .mq5
│                                    # UPDATE: compile_status field
api/
├── compile_endpoints.py             # NEW: compilation API routes
tests/
├── agents/departments/heads/
│   └── test_compilation.py         # NEW: compilation tests
```

### Architecture Patterns

1. **Docker MT5 Compiler on Contabo**:
   - Architecture note: Compilation + testing only — no live trading
   - Docker container with MetaTrader 5 build environment
   - Remote execution via SSH/API from main application

2. **Compilation Flow**:
   ```
   Story 7.2 (.mq5 generated) → Compile Trigger → Docker Compiler →
   [Success] → .ex5 stored, compile_status=success → Story 7.8 (backtest)
   [Failure] → Error analysis → Auto-correct (≤2 iter) → Escalate
   ```

3. **EA Output Directory Structure**:
   - `{data_dir}/strategies/{strategy_id}/versions/{version}/`
   - Files: `{strategy_name}_{version}.mq5`, `{strategy_name}_{version}.ex5`
   - Version scheme: Incremental integer (from Story 7.2)

4. **Strategy Record Schema** (update from Story 7.2):
   ```python
   {
     "strategy_id": str,
     "strategy_name": str,
     "version": int,
     "mq5_path": str,
     "ex5_path": str | None,         # NEW: compiled EA path
     "compile_status": str,          # NEW: "pending" | "success" | "failed"
     "compile_errors": list[str] | None,
     "compile_attempts": int,
     "compile_last_attempt": datetime | None
   }
   ```

5. **Auto-Correction Logic**:
   - Parse MQL5 error messages
   - Common fixes: missing semicolons, typos, parameter types
   - Maximum 2 auto-correction attempts
   - Escalate to FloorManager if all attempts fail

### Key Technical Details

- **Contabo Server**: Docker MT5 compiler resides on Contabo (not Cloudzy)
- **Compilation Trigger**: Fired automatically after Story 7.2 generates .mq5
- **Error Handling**: Must capture both errors AND warnings
- **Islamic Compliance**: Compiled EA must include session_mask, force_close_hour parameters
- **Backtesting Dependency**: Compilation is a hard dependency for backtesting (Story 7.8)

### Previous Story Intelligence (Story 7.2)

- Created `src/trd/` module with schema, parser, validator
- Created `src/mql5/` module with generator and base EA template
- Created `src/strategy/` module for EA storage with versioning
- Generated MQL5 code includes: OnInit, OnTick, OnDeinit, session management, risk controls
- Subtask 4.3 (compile trigger) was marked incomplete — THIS STORY MUST IMPLEMENT IT
- 15 unit tests passing
- Story 7.2 is in review status

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.3]
- [Source: _bmad-output/implementation-artifacts/7-2-development-department-real-mql5-ea-code-generation.md]
- [Source: _bmad-output/planning-artifacts/architecture.md#Alpha-Forge]
- [Source: src/agents/departments/heads/development_head.py]
- [Source: src/mql5/generator.py]
- [Source: src/strategy/output.py]
- [Source: _bmad-output/project-context.md]

## Dev Agent Record

### Agent Model Used

Claude Code (MiniMax-M2.5)

### Debug Log References

- Story 7.2 generated .mq5 files and stored in EA output directory
- Subtask 4.3 "Add compile trigger for Story 7.3" was NOT completed in Story 7.2
- This story MUST wire the compile trigger AND implement the full compilation pipeline
- Docker MT5 compiler on Contabo is the target architecture

### Completion Notes List

- Implemented compile trigger in DevelopmentHead._generate_ea_from_trd() - automatically triggers compilation after EA generation
- Added _escalate_compilation_failure() method to handle escalation to FloorManager
- All 190 mql5 tests pass
- 29 new compilation service tests added and passing

### File List

- src/agents/departments/heads/development_head.py - ADDED: compilation trigger call after EA generation
- src/mql5/compiler/__init__.py - Already exists
- src/mql5/compiler/docker_compiler.py - Already exists
- src/mql5/compiler/error_parser.py - Already exists
- src/mql5/compiler/autocorrect.py - Already exists
- src/mql5/compiler/service.py - Already exists
- src/strategy/output.py - Already has compile_status fields
- src/api/compile_endpoints.py - Already exists
- tests/mql5/test_compilation_service.py - NEW: 29 tests for compilation

### Change Log

- Date: 2026-03-19 - Wired compile trigger in DevelopmentHead after EA generation, added _escalate_compilation_failure method, added comprehensive test suite (29 tests) - Story complete

### Code Review Fixes (2026-03-20)

- Fixed: Added `contabo_user` as instance attribute in DockerMQL5Compiler for proper per-instance configuration
- Fixed: Un-skipped SSH compilation test (test_compile_via_ssh_success now passes)
- Fixed: Error parser now extracts column numbers from fallback plain text errors
- Fixed: CompilationStatusResponse endpoint now returns warnings field
- Tests: All 30 tests now passing

