---
name: 'step-01-init'
description: 'Initialize coding-loop — detect fresh start or resume, read sprint-status.yaml, build loop-manifest with only non-done stories'
continueFile: './step-01b-continue.md'
nextStepFile: './step-02-story-loop.md'
loopManifestFile: '{output_folder}/coding-loop/loop-manifest.md'
sprintStatusFile: '{project-root}/_bmad-output/implementation-artifacts/sprint-status.yaml'
---

# Step 1: Initialization

## STEP GOAL:

Detect whether this is a fresh run or a resume. On fresh run: read sprint-status.yaml, register only non-done stories into the loop-manifest, and start the loop from the first actionable story.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 📖 CRITICAL: Read the complete step file before taking any action
- 🚫 NEVER ask the user questions — this step runs fully autonomously
- 🔄 CRITICAL: When loading next step, read entire file before executing
- ⚙️ TOOL/SUBPROCESS FALLBACK: If any instruction references a subprocess or tool you do not have access to, perform the operation in your main context thread

### Role Reinforcement:

- ✅ You are an autonomous loop orchestrator — not a collaborator
- ✅ Execute the sequence exactly, no deviation

### Step-Specific Rules:

- 🎯 Read `{sprintStatusFile}` — this is the single source of truth for story status
- 🚫 NEVER include stories with status `done` or `optional` in the loop-manifest
- 🚫 NEVER include epic-level keys (e.g. `epic-1`, `epic-2`) or retrospective keys in the story registry
- 🎯 Only register stories with status: `backlog`, `ready-for-dev`, `in-progress`, or `review`
- ⚙️ TOOL/SUBPROCESS FALLBACK: If subprocess unavailable, read sprint-status.yaml in main thread

## EXECUTION PROTOCOLS:

- 🎯 Check for {loopManifestFile} first — route to continue or build fresh
- 💾 Create {loopManifestFile} if fresh run, using sprint-status.yaml as source
- 📖 Auto-proceed — no user interaction in this step

## CONTEXT BOUNDARIES:

- Available: `{sprintStatusFile}` (read-only — never write to it)
- Focus: detect run state, build or load loop-manifest with correct story set
- Limits: read sprint-status.yaml only, never write to BMAD files

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Check for Existing Loop-Manifest

Check if `{loopManifestFile}` exists and contains a `stepsCompleted` array.

- **IF EXISTS** with stepsCompleted → load, read entire file, then execute `{continueFile}`
- **IF NOT EXISTS** → continue to step 2 (fresh run)

### 2. Read Sprint-Status YAML (Fresh Run)

Launch a subprocess that:
1. Loads `{sprintStatusFile}`
2. Reads the `development_status` block
3. Filters entries to ONLY those where:
   - Status is `backlog`, `ready-for-dev`, `in-progress`, or `review`
   - Key does NOT start with `epic-`
   - Key does NOT end with `-retrospective`
4. Returns a structured list in order: `[{ story_key: "4-1-...", story_id: "4.1", epic: 4, status: "review" }, { story_key: "6-0-...", story_id: "6.0", epic: 6, status: "ready-for-dev" }, ...]`

If subprocess unavailable: read `{sprintStatusFile}` directly in main thread and apply the same filters.

### 3. Build Loop-Manifest

Create `{loopManifestFile}` using ONLY the filtered non-done stories from step 2:

```markdown
---
stepsCompleted: ['step-01-init']
lastStep: 'step-01-init'
created: [current date]
status: IN_PROGRESS
---

# Loop Manifest — coding-loop

## Story Registry

| Epic | Story ID | Story Key | Sprint Status | Loop Phase |
|------|----------|-----------|---------------|------------|
[one row per non-done story, in epic/story order]
```

**Sprint Status** = value from sprint-status.yaml (`backlog`, `ready-for-dev`, `in-progress`, `review`)
**Loop Phase** = the coding-loop's own tracking (`pending`, `creating`, `developing`, `reviewing`, `complete`)

**Mapping sprint status → loop phase on first load:**
- `backlog` → `pending`
- `ready-for-dev` → `pending` (story file already exists, create-story will handle)
- `in-progress` → `developing` (already being developed — skip to dev/review)
- `review` → `reviewing` (already developed — skip straight to code review)

### 4. Auto-Proceed to Story Loop

Display: **Loop manifest created. [N] stories registered. Starting...**

Load, read entire file, then execute `{nextStepFile}`

#### EXECUTION RULES:

- Auto-proceed step — no user menu
- Proceed directly after manifest is created

#### Menu Handling Logic:

- After manifest is created, immediately load, read entire file, then execute `{nextStepFile}`

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Correctly detects fresh vs resume state
- loop-manifest contains ONLY non-done stories
- No `done` stories included — workflow starts from current position, not epic 1
- `review` stories included regardless of epic — they get code review only (phase routing handles it)
- Sprint status correctly mapped to loop phase
- Auto-proceeds without asking user any questions — fully autonomous from start

### ❌ SYSTEM FAILURE:

- Including `done` stories in the loop-manifest
- Including epic-level or retrospective keys as stories
- Asking the user any questions
- Writing to sprint-status.yaml or any BMAD story files
- Starting from epic 1 when stories 1.x through 5.x are already done

**Master Rule:** Only non-done stories. Start from where the project actually is.
