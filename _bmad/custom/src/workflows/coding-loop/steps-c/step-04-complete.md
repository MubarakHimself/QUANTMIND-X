---
name: 'step-04-complete'
description: 'Mark all stories complete in loop-manifest, display final summary, done'
loopManifestFile: '{output_folder}/coding-loop/loop-manifest.md'
---

# Step 4: Complete

## STEP GOAL:

Mark the loop-manifest as fully complete, display a final summary of everything processed, and close the workflow.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 📖 CRITICAL: Read the complete step file before taking any action
- ⚙️ TOOL/SUBPROCESS FALLBACK: If any instruction references a subprocess or tool you do not have access to, perform the operation in your main context thread

### Role Reinforcement:

- ✅ You are an autonomous loop orchestrator — final step, display and close
- ✅ No further skills to invoke — this is the completion gate

### Step-Specific Rules:

- 🎯 Update loop-manifest status to COMPLETE before displaying summary
- 💬 Display a clear, concise summary — stories processed, epics completed, automate runs

## EXECUTION PROTOCOLS:

- 💾 Update {loopManifestFile} status to COMPLETE
- 🎯 Display final summary
- 📖 Wait for user acknowledgment ([C] Continue) before closing

## CONTEXT BOUNDARIES:

- Available: loop-manifest.md (full history of all stories and automate runs)
- Focus: summarize and close
- Dependencies: all stories complete, all epic automate runs logged

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

### 1. Update Loop-Manifest to COMPLETE

Update `{loopManifestFile}` frontmatter:

```yaml
status: COMPLETE
completedDate: [current date]
```

### 2. Display Final Summary

Read `{loopManifestFile}` and display:

```
**coding-loop Complete**

Stories Processed: [total complete] / [total registered]
Epics Completed:   [list of epic numbers]
Automate Runs:     [count] (cumulative coverage across all stories)
Completed:         [current date]

All sub-stories have been created, developed, reviewed, and tested.
```

### 3. Present Completion Menu

Display: **All done. [C] Acknowledge**

#### EXECUTION RULES:

- Halt and wait for user acknowledgment
- This is the final step — no next step to load

#### Menu Handling Logic:

- IF C: Workflow is complete. No further action.
- IF Any other: Remind user this is the final step, redisplay menu

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Loop-manifest status updated to COMPLETE
- Accurate summary displayed
- User sees full count of what was accomplished

### ❌ SYSTEM FAILURE:

- Not updating loop-manifest to COMPLETE
- Inaccurate summary counts
- Proceeding to a non-existent next step

**Master Rule:** Final step. Update manifest, summarize, acknowledge.
