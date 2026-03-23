---
name: coding-loop
description: Autonomous loop workflow that processes all backlog sub-stories through a sealed create → dev → review cycle per story, with cumulative test automation at each epic boundary
web_bundle: true
---

# Coding Loop

**Goal:** Autonomously process all backlog sub-stories one at a time — create, develop (isolated sub-agent), review+fix — looping until all stories are complete, with cumulative test automation at each epic boundary.

**Your Role:** You are an autonomous loop orchestrator. This is not a collaborative workflow — you execute a precise, deterministic sequence. Follow each step file exactly. Do not ask questions, do not deviate, do not wait for user input unless a failure condition is explicitly defined.

## WORKFLOW ARCHITECTURE

### Core Principles

- **Micro-file Design**: Each step is a self-contained instruction file — load one at a time, execute completely
- **Just-In-Time Loading**: Only the current step file is in memory — never load future step files until directed
- **Sequential Enforcement**: Sequence within step files must be completed in order, no skipping or optimization
- **State Tracking**: Progress tracked in `loop-manifest.md` — the workflow's own loop state file
- **Sealed Loops**: Each sub-story gets its own completely isolated loop — no context sharing between sub-stories
- **Autonomous Execution**: All skill invocations run with "follow that skill, no questions, use best judgment"

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before taking any action
2. **FOLLOW SEQUENCE**: Execute all numbered sections in order, never deviate
3. **AUTO-PROCEED**: Most steps auto-proceed — only halt when explicitly told to wait for user input
4. **SAVE STATE**: Update `loop-manifest.md` before loading next step
5. **LOAD NEXT**: When directed, load, read entire file, then execute the next step file

### Critical Rules (NO EXCEPTIONS)

- 🛑 **NEVER** load multiple step files simultaneously
- 📖 **ALWAYS** read entire step file before execution
- 🚫 **NEVER** skip steps or optimize the sequence
- 💾 **ALWAYS** update `loop-manifest.md` at state transition points
- 🎯 **ALWAYS** follow the exact instructions in the step file
- 🤖 **NEVER** ask the user questions unless a failure condition is detected
- 📋 **NEVER** create mental todo lists from future steps
- ⚙️ **ALWAYS** invoke skills with: "follow that skill, no questions, use best judgment"

---

## INITIALIZATION SEQUENCE

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/bmb/config.yaml` and resolve:

- `user_name`, `communication_language`, `output_folder`

### 2. Mode Routing

**IF** invoked with `-c` or no flag (default — create/run mode):
Load, read the full file, then execute `./steps-c/step-01-init.md`

**IF** invoked with `-e` (edit mode):
Load, read the full file, then execute `./steps-e/step-01-edit.md`

**IF** invoked with `-v` (validate mode):
Load, read the full file, then execute `./steps-v/step-01-validate.md`
