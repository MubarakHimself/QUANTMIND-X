# Story 11.4: ITT Rebuild Portability — Full Backup & Restore

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

**As a** system owner,
**I want** the ITT to be fully restorable on a new machine from backup,
**So that** hardware failure does not cause permanent data or configuration loss (FR69).

## Acceptance Criteria

1. **Given** a full system backup has run,
   **When** the restore procedure runs on a clean machine,
   **Then** all configs (provider credentials, server connections, broker accounts, risk parameters) are restored,
   **And** the knowledge base (PageIndex data) is restored,
   **And** all strategy artifacts (TRDs, EAs, backtest results, graph memory) are restored.

2. **Given** a restore completes,
   **When** I start the application,
   **Then** the system is fully operational without manual re-configuration,
   **And** a restore completion report shows in Copilot.

## Tasks / Subtasks

- [x] Task 1: Backup Script Infrastructure (AC: #1)
  - [x] Task 1.1: Create backup script in scripts/backup_full_system.sh
  - [x] Task 1.2: Configure backup destinations (local + remote) - via env vars
  - [x] Task 1.3: Implement config snapshot module
  - [x] Task 1.4: Implement knowledge base snapshot module
  - [x] Task 1.5: Implement strategy artifacts snapshot module
- [x] Task 2: Restore Script Infrastructure (AC: #1)
  - [x] Task 2.1: Create restore script in scripts/restore_full_system.sh
  - [x] Task 2.2: Implement config restoration with validation
  - [x] Task 2.3: Implement knowledge base restoration
  - [x] Task 2.4: Implement strategy artifacts restoration
- [x] Task 3: Copilot Integration (AC: #2)
  - [x] Task 3.1: Add backup/restore query intents to FloorManager
  - [x] Task 3.2: Implement restore completion report in Copilot
  - [x] Task 3.3: Add restore status tracking

## Dev Notes

### Key Architecture Context

**FR69: machine portability**
- Backup includes: configs, knowledge base, strategy artifacts, graph memory
- NOT live trade state (MT5 owns that)
- FR70: server migration without data loss uses the same backup/restore mechanism

**From Epic 11 Story 11.0 Audit:**
- Existing backup/restore scripts need to be verified
- Graph memory in `src/memory/graph/`
- Knowledge base in `src/knowledge/`
- Configs in various locations: `src/database/models/`, config files

**Technical Stack:**
- Shell scripts for backup/restore (bash)
- SQLite database backup (sqlite3)
- DuckDB data backup
- Graph memory serialization

### Files to Create/Modify

**NEW FILES:**
- `scripts/backup_full_system.sh` — Main backup script
- `scripts/restore_full_system.sh` — Main restore script
- `scripts/backup_config.sh` — Config backup module
- `scripts/backup_knowledge.sh` — Knowledge base backup module
- `scripts/backup_strategies.sh` — Strategy artifacts backup module

**MODIFY:**
- `src/agents/departments/floor_manager.py` — Add backup/restore query handlers

### Technical Specifications

**Backup Script Structure:**
```bash
#!/bin/bash
# Full System Backup
# FR69: machine portability
set -euo pipefail

# Directories to backup:
# 1. Configs: ~/.quantmind/config/, src/database/models/
# 2. Knowledge base: src/knowledge/pageindex/
# 3. Strategy artifacts: src/strategy/, src/trd/, src/backtesting/results/
# 4. Graph memory: src/memory/graph/store/
# 5. Provider configs: src/database/models/provider_config.py

# Output: backup_YYYYMMDD.tar.gz
```

**Restore Script Structure:**
```bash
#!/bin/bash
# Full System Restore
# FR69: machine portability
set -euo pipefail

# Input: backup_YYYYMMDD.tar.gz
# Restore order: configs → knowledge → strategies → graph memory
# Validation after each step
```

**FloorManager Integration:**
- Intent: "Backup system", "Restore from backup", "What's in the backup"
- Query backup manifest
- Trigger restore procedure

### Testing Standards

- Unit test: Individual backup modules
- Integration test: Full backup/restore cycle (use test directory)
- Failure simulation: Missing files, corrupted archive, permission errors

### Project Structure Notes

- Epic 11: System Management & Resilience
- Three deployment nodes: Contabo (primary), Cloudzy (trading), Desktop (local dev)
- NODE_ROLE environment variable controls router group registration
- Backup/restore runs on Contabo primarily

### Previous Story Intelligence

**From Story 11.0 (Infrastructure & System State Audit):**
- Existing backup/restore scripts need verification
- Graph memory exists in `src/memory/graph/`
- Knowledge base in `src/knowledge/`

**From Story 11.1 (Nightly Rsync Cron):**
- Similar script patterns (set -euo pipefail, logging functions)
- Notification on failure patterns
- Audit trail logging

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Epics: `_bmad-output/planning-artifacts/epics.md` (Story 11.4)
- Source: `scripts/`, `src/memory/graph/`, `src/knowledge/`, `src/strategy/`
- FR69: machine portability
- FR70: server migration without data loss

---

## Developer Implementation Guide

### What NOT to Do

1. **DO NOT** backup live trade state — MT5 owns that data
2. **DO NOT** use proprietary backup formats — use standard tar.gz
3. **DO NOT** skip validation after restore — verify each component
4. **DO NOT** restore over existing data without confirmation

### What TO Do

1. **DO** follow existing script patterns from Story 11.1
2. **DO** log all operations to audit trail
3. **DO** validate checksums before and after transfer
4. **DO** implement incremental backup option for large datasets
5. **DO** use FloorManager for backup/restore queries

### Code Patterns

**Script header pattern:**
```bash
#!/bin/bash
# Full System Backup
# FR69: machine portability
set -euo pipefail
log() { echo "[$(date -Iseconds)] $*"; }
error() { log "ERROR: $*" >&2; exit 1; }
```

**Backup module pattern:**
```bash
backup_configs() {
    local backup_dir="$1"
    log "Backing up configurations..."
    # Backup provider configs, server connections, broker accounts, risk params
    tar czf "$backup_dir/configs.tar.gz" -C "$HOME" .quantmind/config/
    # Database models
    sqlite3 "$HOME/data/quantmind.db" ".backup $backup_dir/quantmind.db"
}
```

**Restore validation pattern:**
```bash
validate_restore() {
    local restore_dir="$1"
    log "Validating restore..."
    # Check all expected files exist
    # Verify checksums
    # Test database integrity
}
```

---

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

### Completion Notes List

- Implemented backup script with modules: configs, knowledge base, strategy artifacts, graph memory, canvas context
- Implemented restore script with validation and restore completion notification
- Added Copilot integration with backup/restore query intents
- Followed existing script patterns from Story 11.1 (sync_cloudzy_to_contabo.sh)
- Added 3 new command intents: BACKUP_SYSTEM, RESTORE_BACKUP, BACKUP_QUERY
- Restore completion report shows in Copilot via notification file

### File List

- scripts/backup_full_system.sh (NEW) - Main backup script with all backup modules
- scripts/restore_full_system.sh (NEW) - Main restore script with validation
- src/intent/patterns.py (MODIFY - added backup/restore intents)
- src/agents/departments/floor_manager.py (MODIFY - added backup/restore query handlers)
- tests/scripts/test_backup_restore.py (NEW) - Unit tests for backup/restore scripts

## Review Follow-ups (AI)

- [ ] [AI-Review][MEDIUM] Commit backup scripts to git - scripts are currently untracked
- [ ] [AI-Review][LOW] Add integration test for full backup/restore cycle with temp directory
- [ ] [AI-Review][LOW] Add cron job configuration documentation for automated backups

## Change Log

- 2026-03-21: Initial implementation complete
  - Created backup_full_system.sh with modules for configs, knowledge base, strategy artifacts, graph memory, canvas context
  - Created restore_full_system.sh with validation and completion notification
  - Added BACKUP_SYSTEM, RESTORE_BACKUP, BACKUP_QUERY intents to intent patterns
  - Added backup/restore query handlers to FloorManager with Copilot integration
- 2026-03-21: Code review fixes
  - Fixed database backup path (data/db/quantmind.db instead of src/database/)
  - Added .quantmind directory backup
  - Added unit tests for backup/restore scripts (tests/scripts/test_backup_restore.py)
  - Made scripts executable
