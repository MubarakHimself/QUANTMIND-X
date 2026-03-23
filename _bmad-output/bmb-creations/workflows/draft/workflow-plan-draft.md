---
stepsCompleted: ['step-01-discovery', 'step-02-classification', 'step-03-requirements', 'step-04-tools', 'step-05-plan-review', 'step-06-design', 'step-07-foundation', 'step-08-build-step-01', 'step-09-build-remaining']
created: 2026-03-19
status: COMPLETE
approvedDate: 2026-03-19
workflowName: coding-loop
targetWorkflowPath: _bmad/custom/src/workflows/coding-loop/
---

# Workflow Creation Plan

## Discovery Notes

**User's Vision:**
A repeating story lifecycle pipeline that processes backlog sub-stories one at a time (e.g. epic 5 → sub-stories 5.1, 5.2, 5.3...). Each sub-story goes through a fixed 3-skill loop — create, develop (via sub-agent), review+fix — before advancing. After all sub-stories in an epic are complete, a cumulative test automation pass runs across the entire epic.

**Who It's For:**
Mubarak — solo developer running BMAD-driven development against a backlog of epics and sub-stories (~11 stories across multiple epics).

**What It Produces:**
- Per sub-story: a created story file, implemented code, and a clean code review (all issues fixed)
- Per epic: cumulative test automation coverage expansion

**Key Insights:**
- The dev story step MUST run in a separate sub-agent session (different context/scope) — no context sharing between loops
- Create story and review run in the same parent session
- Review must be fully complete (all issues fixed, option 1 selected automatically) before advancing
- The automate step (`bmad-tea-testarch-automate`) runs once per epic boundary, covering ALL sub-stories in that epic combined
- Scope locked to 4 skills: create-story, dev-story, code-review, testarch-automate
- Skills are invoked directly — they handle all project state discovery internally

## Classification Decisions

**Workflow Name:** coding-loop
**Target Path:** `_bmad/custom/src/workflows/coding-loop/`

**4 Key Decisions:**
1. **Document Output:** false (non-document — orchestrates actions, no persistent output from the workflow itself)
2. **Module Affiliation:** standalone (orchestrates across BMM + TEA skills, belongs to no single module)
3. **Session Type:** continuable (epic processing spans many sub-stories, must resume mid-epic)
4. **Lifecycle Support:** tri-modal (create + edit + validate — steps-c/, steps-e/, steps-v/)

**Structure Implications:**
- Needs `steps-c/`, `steps-e/`, `steps-v/` folders
- Needs `step-01b-continue.md` for resuming mid-epic
- `stepsCompleted` tracking in frontmatter to know which sub-stories are processed
- No output document template needed (non-document)

## Requirements

**Flow Structure:**
- Pattern: Looping (per sub-story) + Linear (within each sub-story loop)
- Phases:
  1. YAML Scan / Resume Detection — read BMAD YAML (read-only, never write) to understand current story positions; this informs the loop-manifest state, no separate init step needed
  2. Story Loop — per sub-story: create-story → dev-story (sub-agent) → code-review (fix all) → mark complete in loop-manifest
  3. Epic Boundary — run testarch-automate cumulatively across ALL sub-stories in the completed epic
  4. Complete — all stories processed
- Estimated steps: ~6-7 step files
- Total story scope: all stories in the project (currently ~11 across multiple epics)

**User Interaction:**
- Style: Mostly autonomous — skills handle all discovery and heavy lifting
- Decision points: (1) Invocation at start, (2) Pause only on unresolvable code review conflict
- Checkpoint frequency: Start only; otherwise fully unattended until all stories done
- Loop isolation: Each sub-story is a sealed loop — no context bleed; fresh sub-agent sessions per loop

**Inputs Required:**
- Required: None beyond workflow invocation — skills auto-discover project state internally
- Optional: Manual override of starting story (edge case)
- Prerequisites: Existing BMAD project with defined epics and stories

**Output Specifications:**
- Type: Actions performed + state file maintained
- Format: Non-document; workflow reads/updates `loop-manifest.md` throughout
- loop-manifest tracks per-story pipeline phase (pending → creating → developing → reviewing → complete)
- testarch-automate expands test coverage cumulatively at each epic boundary (not reset per epic)
- Frequency: Continuous until all stories complete

**Failure Conditions (pause only for these):**
- Code review surfaces a conflict it cannot auto-resolve
- All other error conditions considered non-existent based on proven 20+ sub-story track record

**Success Criteria:**
- All stories in loop-manifest marked `complete`
- Each story passed clean code review (no open issues)
- testarch-automate run at each epic boundary with cumulative coverage
- Loop-manifest reflects full project completion

**Instruction Style:**
- Overall: Prescriptive — exact skill invocations, deterministic loop logic, no creative interpretation
- Skills are invoked and handle all content decisions internally
- Workflow only manages sequencing, state tracking, and failure detection

## Tools Configuration

**Core BMAD Tools:**
- **Party Mode:** excluded
- **Advanced Elicitation:** excluded
- **Brainstorming:** excluded

**LLM Features:**
- **Web-Browsing:** included — MCP access for documentation during coding tasks
- **File I/O:** included — reading BMAD YAML (read-only) and reading/writing loop-manifest.md
- **Sub-Agents:** included — core architecture; dev-story runs in an isolated sub-agent session with no shared context
- **Sub-Processes:** included — background tasks during development (running tests, starting servers)

**Memory:**
- Type: continuable
- Tracking: stepsCompleted array in loop-manifest frontmatter, lastStep, per-story phase status
- Two-file system: (1) BMAD YAML (existing — read-only; tracks story status), (2) loop-manifest.md (workflow's own loop tracker — read/write)

**Autonomous Behavior Rules (critical):**
- All skill invocations: "follow that skill, no questions, use best judgment"
- Phase 1: pre-scan BMAD YAML → identify exact next pending sub-story → pass sub-story ID to create-story skill (eliminates redundant questions)
- testarch-automate: explicitly instructed to cover ALL sub-stories in the current epic combined
- Code review: always auto-select option 1 (fix them all) — never pause for user choice

**External Integrations:** none — all BMAD skills are local
**Installation Requirements:** none

## Workflow Design

**Step Outline (steps-c/ — 5 files):**

| File | Type | Pattern | Purpose |
|---|---|---|---|
| step-01-init.md | Continuable Init | Auto-proceed | Check loop-manifest; if fresh → scan BMAD YAML (subprocess Pattern 3), build manifest, route to step-02; if resuming → route to step-01b |
| step-01b-continue.md | Continuation Router | Auto-proceed | Read loop-manifest, detect position, display progress, route to step-02/03/04 |
| step-02-story-loop.md | Middle (Simple, self-loop) | Auto-proceed | Subprocess scan BMAD YAML → get next pending sub-story ID → invoke create-story → spawn dev-story sub-agent → invoke code-review (auto fix-all) → mark complete → self-loop if more sub-stories, else → step-03 |
| step-03-epic-boundary.md | Middle (Simple) | Auto-proceed | Invoke testarch-automate scoped to ALL sub-stories in completed epic → log in manifest → loop to step-02 if more epics, else → step-04 |
| step-04-complete.md | Final | C only | Mark manifest COMPLETE, display summary |

**Tri-modal stubs:** steps-e/ and steps-v/ created as minimal placeholders for future edit cycle.

**Subprocess Optimization:**
- step-01 + step-02: Pattern 3 (data ops) — subprocess loads BMAD YAML, returns structured story list / next pending sub-story ID only
- step-02: Sub-agent for dev-story (sealed session, core architecture)
- step-03: testarch-automate handles internally

**Failure Handling:**
- Code review conflict → pause and surface to user
- All other conditions → auto-proceed (proven stable over 20+ sub-stories)

**Autonomous Behavior (all skill invocations):**
- "Follow that skill, no questions, use best judgment"

## Foundation Build Complete

**Created:**
- `_bmad/custom/src/workflows/coding-loop/workflow.md`
- `_bmad/custom/src/workflows/coding-loop/steps-c/` (empty — step files next)
- `_bmad/custom/src/workflows/coding-loop/steps-e/` (stub — future edit cycle)
- `_bmad/custom/src/workflows/coding-loop/steps-v/` (stub — future validate cycle)
- `_bmad/custom/src/workflows/coding-loop/data/`

**Configuration:**
- Workflow name: coding-loop
- Continuable: yes
- Document output: no
- Mode: tri-modal (steps-c/, steps-e/, steps-v/)

**Next Steps:**
- Step 8: Build step-01-init.md and step-01b-continue.md
- Step 9: Build step-02-story-loop.md, step-03-epic-boundary.md, step-04-complete.md

## All Steps Built — COMPLETE

**steps-c/ (create mode):**
- `step-01-init.md` — continuable init, subprocess Pattern 3 YAML scan, build loop-manifest or route to continue
- `step-01b-continue.md` — reads manifest, detects position, auto-routes to step-02/03/04
- `step-02-story-loop.md` — sealed loop per sub-story: create → dev (sub-agent) → review (auto fix-all) → self-loop or epic boundary
- `step-03-epic-boundary.md` — cumulative testarch-automate for full epic, logs run, routes to next epic or complete
- `step-04-complete.md` — marks manifest COMPLETE, displays summary, user acknowledges

**steps-e/ (edit mode):**
- `step-01-edit.md` — stub, informs user, future cycle

**steps-v/ (validate mode):**
- `step-01-validate.md` — stub, informs user, future cycle

**workflow.md** — tri-modal mode routing (-c default, -e, -v)
