# Risk Allocation Implementation Progress

Date: 2026-04-03
Status: in progress
Current scope: Batch 1 authority cleanup

## Active worktree

- Branch: `feat/risk-allocation-batch-1-authority`
- Purpose: isolate Batch 1 runtime allocation authority cleanup from unrelated dirty work on `main`

## Execution status

### Batch 1: Authority Map And Runtime Contracts
- [ ] Create isolated worktree
- [ ] Run pre-edit caller scan
- [ ] Classify governor ownership and live callers
- [ ] Write or update failing tests for authority ownership changes
- [ ] Implement `engine.py` single-authority selection
- [ ] Implement `commander.py` injected-authority requirement
- [ ] Reclassify non-authority governor files
- [ ] Run targeted verification
- [ ] Record completion notes

### Batch 2: Session And Queue Authority
- [ ] Not started

### Batch 3: Drawdown, Pressure, And Lock Model
- [ ] Not started

## Rules

- Delete obsolete code after caller migration and verification
- Do not comment out dead code
- Do not let frontend become runtime truth
- Keep backend/trading-node ownership explicit
