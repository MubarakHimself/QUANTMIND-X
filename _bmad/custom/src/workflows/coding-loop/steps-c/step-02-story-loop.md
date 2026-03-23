---
name: 'step-02-story-loop'
description: 'Process one sub-story — detect its phase, run only the needed skills, update loop-manifest — then loop or advance to epic boundary'
loopManifestFile: '{output_folder}/coding-loop/loop-manifest.md'
selfFile: './step-02-story-loop.md'
nextStepFile: './step-03-epic-boundary.md'
---

# Step 2: Story Loop

## STEP GOAL:

Process exactly one pending sub-story. Detect its current phase from the loop-manifest (pending / developing / reviewing) and run only the skills needed from that phase forward. Update the loop-manifest, then self-loop for the next sub-story or advance to the epic boundary.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 📖 CRITICAL: Read the complete step file before taking any action
- 🚫 NEVER ask the user questions — execute the sequence autonomously
- 🔄 CRITICAL: When loading next step, read entire file before executing
- ⚙️ TOOL/SUBPROCESS FALLBACK: If any instruction references a subprocess or tool you do not have access to, perform the operation in your main context thread

### Role Reinforcement:

- ✅ You are the **orchestrator** — you manage state and spawn sub-agents, nothing more
- ✅ You NEVER run skills directly in this session — all skills run in isolated sub-agent sessions
- ✅ Your only actions: read/write loop-manifest, spawn sub-agents via Task tool, wait for returns, route

### Step-Specific Rules:

- 🎯 Read loop-manifest to find the first non-complete story — use its Loop Phase to determine entry point
- 🚫 NEVER run create-story, dev-story, or code-review directly in the main session
- 🚫 NEVER re-run create-story if loop phase is `developing` or `reviewing`
- 🚫 NEVER re-run dev-story if loop phase is `reviewing`
- 🚫 NEVER mix context between sub-stories — each sub-agent session is completely isolated
- 🚫 NEVER wait for user input mid-loop unless an unresolvable review conflict is reported back
- ⚙️ TOOL/SUBPROCESS FALLBACK: If Task tool unavailable, note the limitation and pause for user

## EXECUTION PROTOCOLS:

- 🎯 Main session = orchestrator only: read manifest → spawn sub-agent → wait → update manifest → route
- 💾 Update {loopManifestFile} at every phase transition (the only writing this session does)
- 📖 Self-loop via {selfFile} if more sub-stories remain in current epic
- 🚫 FORBIDDEN to run any skill directly in the main session
- 🚫 FORBIDDEN to process more than one sub-story per execution of this step

## CONTEXT BOUNDARIES:

- Available: loop-manifest.md (current state — single source of truth for loop phases)
- Focus: exactly one sub-story per loop execution
- Limits: read loop-manifest, update loop-manifest; never write to sprint-status.yaml
- Dependencies: loop-manifest must exist (created in step-01)

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Identify Next Actionable Story

Read `{loopManifestFile}` — find the first row where Loop Phase is NOT `complete`.

Store:
- `currentStory` = Story ID (e.g. `6.0`)
- `currentStoryKey` = Story Key (e.g. `6-0-knowledge-infrastructure-audit`)
- `currentEpic` = Epic number (e.g. `6`)
- `currentPhase` = Loop Phase value (`pending`, `creating`, `developing`, `reviewing`)

### 2. Route by Current Phase

**IF** `currentPhase` is `pending` or `creating` → proceed to **Step 3** (create-story)
**IF** `currentPhase` is `developing` → skip to **Step 5** (dev-story sub-agent)
**IF** `currentPhase` is `reviewing` → skip to **Step 7** (code review)

### 3. Update Loop-Manifest: Creating

Update `{loopManifestFile}` — set `currentStory` Loop Phase to `creating`.

### 4. Spawn Sub-Agent — Create Story

Use the Task tool to spawn an isolated sub-agent with ONLY this instruction:

> "You are in the QUANTMINDX project at `{project-root}`. Invoke the `/bmad-bmm-create-story` skill. Follow that skill, no questions, use best judgment. Do not ask for confirmation — proceed directly with story creation. Complete and return."

Wait for the sub-agent to return before continuing. Do not run the skill in the main session.

Once sub-agent returns → continue to step 5.

### 5. Update Loop-Manifest: Developing

Update `{loopManifestFile}` — set `currentStory` Loop Phase to `developing`.

### 6. Spawn Isolated Sub-Agent — Dev Story

Use the Task tool to spawn an isolated sub-agent with ONLY this instruction:

> "You are in the QUANTMINDX project at `{project-root}`. Invoke the `/bmad-bmm-dev-story` skill. Follow that skill, no questions, use best judgment. Complete the development and return."

The sub-agent runs in complete isolation — no context from the current session is shared.
Wait for the sub-agent to return before continuing.

Once sub-agent returns → continue to step 7.

### 7. Update Loop-Manifest: Reviewing

Update `{loopManifestFile}` — set `currentStory` Loop Phase to `reviewing`.

### 8. Spawn Sub-Agent — Code Review

Use the Task tool to spawn an isolated sub-agent with ONLY this instruction:

> "You are in the QUANTMINDX project at `{project-root}`. Invoke the `/bmad-bmm-code-review` skill. Follow that skill, no questions, use best judgment. When the skill presents review options, automatically select option 1 (fix them all) without asking the user. Complete all fixes and return."

Wait for the sub-agent to return before continuing. Do not run the skill in the main session.

**Conflict Exception:** If the sub-agent returns reporting an unresolvable conflict, pause and surface it to the user. Wait for their input, then re-spawn the sub-agent to continue.

### 9. Update Loop-Manifest: Complete

Once code review finishes with no open issues, update `{loopManifestFile}` — set `currentStory` Loop Phase to `complete`.

Display: `**✓ Story [currentStory] complete.** Checking for next...`

### 10. Determine Next Action

Read `{loopManifestFile}`:

**Check A — More non-complete stories in `currentEpic`?**
- **YES** → load, read entire file, then execute `{selfFile}` (self-loop)

**Check B — All stories in `currentEpic` are complete?**
- **YES** → load, read entire file, then execute `{nextStepFile}` (epic boundary)

#### EXECUTION RULES:

- Auto-proceed step — no user menu
- Self-loop or advance immediately after completing sub-story

#### Menu Handling Logic:

- IF more stories in epic: load, read entire file, execute `{selfFile}`
- IF epic complete: load, read entire file, execute `{nextStepFile}`

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Exactly one sub-story processed per execution
- Phase routing correct — no redundant skill invocations
- Dev-story runs in completely isolated sub-agent session
- Code review option 1 selected automatically
- Loop-manifest updated at every phase transition

### ❌ SYSTEM FAILURE:

- Re-running create-story on a story already in `developing` or `reviewing` phase
- Asking user any questions mid-loop (except unresolvable review conflict)
- Processing more than one sub-story per execution
- Waiting for user to choose code review option

**Master Rule:** Phase-aware routing. One sub-story per execution. Auto-proceed always.
