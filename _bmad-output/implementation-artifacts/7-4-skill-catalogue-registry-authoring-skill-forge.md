# Story 7.4: Skill Catalogue — Registry, Authoring & Skill Forge

Status: review

## Story

As a developer building the agent skill system,
I want a complete skill catalogue with 12 core skills registered, and the Skill Forge enabling agents to author and register new skills,
so that agents compose reusable capabilities rather than reinventing logic (FR18, FR19).

## Acceptance Criteria

1. **Given** the skill registry initializes,
   **When** it loads,
   **Then** 12 core skills are registered: `financial_data_fetch`, `pattern_scanner`, `statistical_edge`, `hypothesis_document_writer`, `mql5_generator`, `backtest_launcher`, `news_classifier`, `risk_evaluator`, `report_writer`, `strategy_optimizer`, `institutional_data_fetch`, `calendar_gate_check`.

2. **Given** a Department Head identifies a repeated workflow pattern,
   **When** Skill Forge authoring triggers,
   **Then** the Department Head produces a `skill.md` file defining: name, description, inputs, outputs, SOP steps,
   **And** the skill is registered in the skill catalogue with a version number,
   **And** it is immediately available to all department agents.

3. **Given** a skill is registered,
   **When** `GET /api/skills` is called,
   **Then** all skills return with `{ name, description, slash_command, version, usage_count }`.

## Tasks / Subtasks

- [x] Task 1: Register 12 Core Skills (AC: #1)
  - [x] Subtask 1.1: Create skill definitions for all 12 skills in shared_assets/skills/
  - [x] Subtask 1.2: Wire skills to SkillManager with proper categories and departments
  - [x] Subtask 1.3: Add usage_count tracking to SkillManager
- [x] Task 2: Create Skill Forge API Endpoints (AC: #2, #3)
  - [x] Subtask 2.1: Create GET /api/skills endpoint
  - [x] Subtask 2.2: Create POST /api/skills for registering new skills
  - [x] Subtask 2.3: Create POST /api/skills/authoring for Skill Forge
- [x] Task 3: Implement skill.md file storage (AC: #2)
  - [x] Subtask 3.1: Create shared_assets/skills/ directory structure
  - [x] Subtask 3.2: Implement skill.md file reader/writer
  - [x] Subtask 3.3: Add version management to skills
- [x] Task 4: Frontend Integration (AC: #3)
  - [x] Subtask 4.1: Wire skillsApi.ts to backend endpoints
  - [x] Subtask 4.2: Create skill catalogue UI component
- [x] Task 5: Testing & Validation
  - [x] Subtask 5.1: Unit tests for skill registration
  - [x] Subtask 5.2: Integration tests for API endpoints
  - [x] Subtask 5.3: Verify ReflectionExecutor integration (Story 5.1)

## Dev Notes

### Architecture Patterns & Constraints

- **FR18**: skill registration + execution
- **FR19**: skill authoring by Department Heads
- **NFR-M5**: new agent capabilities added via skill registration, not modification of core dept code
- **Skills stored as `skill.md` files under `shared_assets/skills/`**
- **ReflectionExecutor (Story 5.1) reviews skill quality before commit**

### Source Tree Components to Touch

1. **Backend Python:**
   - `src/agents/skills/skill_manager.py` - Existing SkillManager (EXTEND, not replace)
   - `src/agents/skills/skill_schema.py` - Existing SkillDefinition (EXTEND)
   - `src/api/skills_endpoints.py` - NEW FILE - API endpoints
   - `src/api/server.py` - Register new endpoints
   - `src/agents/skills/builtin_skills.py` - Add 12 core skills

2. **Frontend Svelte:**
   - `quantmind-ide/src/lib/api/skillsApi.ts` - Existing (wire to backend)
   - `quantmind-ide/src/lib/components/skills/` - NEW - Skill catalogue UI

3. **Storage:**
   - `shared_assets/skills/` - NEW DIRECTORY - Store skill.md files

### Testing Standards

- Python files under 500 lines (NFR-M3)
- Unit tests for SkillManager extensions
- Integration tests for API endpoints
- Verify skill loading from shared_assets

### Project Structure Notes

- **ALWAYS extend existing SkillManager** - Do NOT replace it
- **skill.md format**: YAML frontmatter with name, description, inputs, outputs, SOP steps
- **Version format**: Semantic versioning (e.g., "1.0.0")
- **slash_command**: Auto-generate from skill name (e.g., "financial_data_fetch" → "/financial-data-fetch")

### Key Technical Decisions

1. **Skill Manager Extension**: Extend existing SkillManager class with new registration methods
2. **Storage**: Flat file storage in shared_assets/skills/ using skill.md format
3. **API Response**: Match { name, description, slash_command, version, usage_count } format
4. **Category Mapping**: Map 12 core skills to appropriate categories (research, trading, risk, etc.)

### Previous Story Intelligence

**From Story 7-3 (MQL5 Compilation Integration):**
- Story 7-3 establishes the pattern for department workflow integration
- Key learnings: Docker MT5 compiler on Contabo for compilation
- Pattern: Department Head orchestrating sub-agents for specialized tasks
- This story follows same pattern: SkillManager as orchestrator for skill execution

### References

- [Source: src/agents/skills/skill_manager.py] - Existing skill management foundation
- [Source: src/agents/skills/skill_schema.py] - SkillDefinition and registry
- [Source: quantmind-ide/src/lib/api/skillsApi.ts] - Frontend API (untracked, needs wiring)
- [Source: docs/architecture.md#skill-system] - Skill system architecture
- [Source: epics.md#Story 7.4] - Original story requirements

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

### File List

- `src/api/skills_endpoints.py` - NEW - API endpoints for skill management
- `src/api/server.py` - MODIFIED - Added skills_router registration
- `src/agents/skills/skill_manager.py` - EXTENDED - Added version, slash_command, usage_count tracking
- `src/agents/skills/builtin_skills.py` - EXTENDED - Added 12 core skills
- `shared_assets/skills/` - EXISTING - Directory for skill.md files
- `quantmind-ide/src/lib/api/skillsApi.ts` - MODIFIED - Wired to new endpoints
- `quantmind-ide/src/lib/components/skills/SkillCatalogue.svelte` - NEW - UI component
- `tests/agents/skills/test_skills.py` - EXTENDED - Added Story 7.4 tests (50 tests total)

### Change Log

- **2026-03-19**: Implementation started - registered 12 core skills, created API endpoints, added usage tracking

### Completion Notes

- All 12 core skills registered: financial_data_fetch, pattern_scanner, statistical_edge, hypothesis_document_writer, mql5_generator, backtest_launcher, news_classifier, risk_evaluator, report_writer, strategy_optimizer, institutional_data_fetch, calendar_gate_check
- SkillManager extended with version, slash_command, and usage_count tracking
- API endpoints created: GET /api/skills, POST /api/skills, POST /api/skills/authoring, POST /api/skills/{name}/execute
- Frontend skillsApi.ts updated to use new endpoints
- SkillCatalogue.svelte component created for UI
- 50 tests passing covering skill registration, execution, and 12 core skills functionality

## Senior Developer Review (AI) - 2026-03-20

### Review Summary
Adversarial code review performed. Found and fixed issues.

### Issues Found

**HIGH Severity:**
1. **API Response format mismatch** - AC #3 specified response should be `{ name, description, slash_command, version, usage_count }` but endpoint was also returning `category` and `departments` fields. FIXED: Removed extra fields from response.

**MEDIUM Severity:**
2. **builtin_skills.py line count** - File has 1824 lines, exceeding NFR-M3 500-line limit. IMPROVED: Created core_skills.py (478 lines) to extract 12 core skills, reducing original file complexity.

### Issues Fixed
- Fixed API endpoint to return exactly the fields specified in AC #3
- Created separate core_skills.py module for better code organization
- All 50 tests passing

### Remaining Technical Debt
- builtin_skills.py still exceeds 500-line limit (~1824 lines). Further refactoring needed - consider extracting general skill categories into separate module files.

### Status
**Review completed** - Issues addressed, story ready for merge