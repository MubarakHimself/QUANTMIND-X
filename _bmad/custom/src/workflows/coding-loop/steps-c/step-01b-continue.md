---
name: 'step-01b-continue'
description: 'Resume coding-loop from previous session — read loop-manifest, detect position, route to correct step'
loopManifestFile: '{output_folder}/coding-loop/loop-manifest.md'
storyLoopFile: './step-02-story-loop.md'
epicBoundaryFile: './step-03-epic-boundary.md'
completeFile: './step-04-complete.md'
---

# Step 1b: Continue

## STEP GOAL:

Resume a previous coding-loop session by reading the loop-manifest, determining current position, displaying brief progress, and routing directly to the correct step.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 📖 CRITICAL: Read the complete step file before taking any action
- 🚫 NEVER ask the user questions — auto-detect position and route
- 🔄 CRITICAL: When loading next step, read entire file before executing
- ⚙️ TOOL/SUBPROCESS FALLBACK: If any instruction references a subprocess or tool you do not have access to, perform the operation in your main context thread

### Role Reinforcement:

- ✅ You are an autonomous loop orchestrator resuming from a known position
- ✅ Route without asking — the manifest has all the information needed

### Step-Specific Rules:

- 🎯 Read loop-manifest completely before routing
- 🚫 NEVER restart from scratch — always resume from detected position
- 💬 Display one brief progress line before routing

## EXECUTION PROTOCOLS:

- 🎯 Read {loopManifestFile} and extract current state
- 💾 Do not modify loop-manifest in this step
- 📖 Auto-proceed to correct step based on manifest state

## CONTEXT BOUNDARIES:

- Available: loop-manifest.md with per-story statuses
- Focus: detect position, route correctly
- Limits: do not modify any files in this step

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Read Loop-Manifest

Load `{loopManifestFile}` and read:
- Total story count and how many are marked `complete`
- The first story NOT marked `complete` — this is the current position
- Whether any story is mid-phase (creating/developing/reviewing — not yet complete)
- Whether an epic just completed all sub-stories (all sub-stories complete, automate not yet logged)

### 2. Display Progress Summary

Display one line only:
`**Resuming coding-loop:** [N] of [Total] stories complete. Continuing from [Epic].[Story]...`

### 3. Determine Route

**Routing logic:**

- **IF** any story has status `creating`, `developing`, or `reviewing` → treat it as the current active story → load `{storyLoopFile}`
- **IF** all sub-stories in an epic are `complete` AND no automate run logged for that epic yet → load `{epicBoundaryFile}`
- **IF** all stories across all epics are `complete` AND all automate runs are logged → load `{completeFile}`
- **DEFAULT** → find first `pending` story → load `{storyLoopFile}`

### 4. Auto-Route

Load, read entire file, then execute the determined step file.

#### EXECUTION RULES:

- This is an auto-proceed step — no user menu
- Route immediately after determining position

#### Menu Handling Logic:

- Based on routing logic above, immediately load, read entire file, then execute the appropriate step

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Correctly reads loop-manifest position
- Routes to the right step without user input
- Brief progress summary displayed

### ❌ SYSTEM FAILURE:

- Asking user where to resume
- Restarting from scratch when manifest exists
- Routing to wrong step
- Modifying loop-manifest in this step

**Master Rule:** Auto-detect and route. Never ask the user.
