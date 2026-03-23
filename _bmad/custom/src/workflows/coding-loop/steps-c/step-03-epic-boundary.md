---
name: 'step-03-epic-boundary'
description: 'Run cumulative testarch-automate for completed epic, log in manifest, then route to next epic or completion'
loopManifestFile: '{output_folder}/coding-loop/loop-manifest.md'
storyLoopFile: './step-02-story-loop.md'
nextStepFile: './step-04-complete.md'
---

# Step 3: Epic Boundary

## STEP GOAL:

After all sub-stories in an epic are complete, run testarch-automate cumulatively across all sub-stories in that epic combined. Log the run in the loop-manifest. Route to the next epic's story loop or final completion.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 📖 CRITICAL: Read the complete step file before taking any action
- 🚫 NEVER ask the user questions — execute autonomously
- 🔄 CRITICAL: When loading next step, read entire file before executing
- ⚙️ TOOL/SUBPROCESS FALLBACK: If any instruction references a subprocess or tool you do not have access to, perform the operation in your main context thread

### Role Reinforcement:

- ✅ You are an autonomous loop orchestrator at an epic checkpoint
- ✅ All skill invocations: "follow that skill, no questions, use best judgment"

### Step-Specific Rules:

- 🎯 testarch-automate MUST be scoped to ALL sub-stories in the completed epic — not just the last one
- 🚫 NEVER scope automate to a single sub-story
- 🚫 NEVER ask user if they want to run automate — always run it
- ⚙️ If testarch-automate asks clarifying questions, provide the epic scope and continue — no user interruption

## EXECUTION PROTOCOLS:

- 🎯 Run testarch-automate with explicit cumulative scope
- 💾 Log automate run in {loopManifestFile}
- 📖 Auto-proceed to next epic or completion

## CONTEXT BOUNDARIES:

- Available: loop-manifest.md (knows which epic just completed and all its sub-stories)
- Focus: run automate for completed epic, route correctly
- Dependencies: all sub-stories in current epic must be complete before this step runs

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Read Completed Epic from Loop-Manifest

Read `{loopManifestFile}` and identify:
- The epic that just completed (all sub-stories marked `complete`)
- The full list of sub-story IDs in that epic (e.g. [5.1, 5.2, 5.3, 5.4, 5.5])

Display: `**Epic [N] complete.** Running cumulative test automation across [X] sub-stories...`

### 2. Invoke testarch-automate Skill (Cumulative Scope)

Invoke the `/bmad-tea-testarch-automate` skill with the following context:

> "Follow that skill, no questions, use best judgment. Expand test automation coverage for ALL sub-stories in epic [N]: [list all sub-story IDs]. Run automation across the complete set of stories, not just the most recent. Do not ask for scope confirmation — proceed with the full epic scope."

**If the skill asks clarifying questions:** Answer autonomously using the epic scope from the loop-manifest. Never pause for user input unless there is an unresolvable error.

### 3. Log Automate Run in Loop-Manifest

Update `{loopManifestFile}` Epic Boundaries table — add a row:

```
| [epicNumber] | [current date] | [list of all sub-story IDs in epic] |
```

### 4. Determine Next Action

Read `{loopManifestFile}`:

**Check A — Are there any stories in a subsequent epic with status `pending`?**
- **YES** → Display: `**Starting epic [next epic number]...**` → load, read entire file, then execute `{storyLoopFile}`

**Check B — All stories across all epics are `complete` and all epic automate runs are logged?**
- **YES** → load, read entire file, then execute `{nextStepFile}`

#### EXECUTION RULES:

- This is an auto-proceed step — no user menu
- Route immediately after logging automate run

#### Menu Handling Logic:

- IF more epics with pending stories: load, read entire file, execute `{storyLoopFile}`
- IF all epics complete: load, read entire file, execute `{nextStepFile}`

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- testarch-automate runs for ALL sub-stories in the completed epic combined
- Automate run logged in loop-manifest
- Correctly routes to next epic's story loop or final completion
- No user interaction required

### ❌ SYSTEM FAILURE:

- Running automate for only the last sub-story instead of the full epic
- Asking user if they want to run automate
- Failing to log the automate run
- Routing incorrectly (going to complete when epics remain)

**Master Rule:** Cumulative scope always. Auto-proceed without user input.
