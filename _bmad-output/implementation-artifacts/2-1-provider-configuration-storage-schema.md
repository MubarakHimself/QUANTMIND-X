# Story 2.1: Provider Configuration Storage Schema

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader setting up QUANTMINDX for the first time,
I want a secure, typed storage schema for AI provider credentials,
So that all downstream agent components retrieve provider settings from a single source of truth without touching `.env` directly.

## Acceptance Criteria

1. [x] [AC-1] Given no provider has been configured, when the backend initialises, then a providers config store is created with schema: `{ id, provider_type, display_name, api_key_encrypted, base_url, model_list, tier_assignment, is_active, created_at_utc }`.

2. [x] [AC-2] Given an API key is saved, when written to the store, then it is encrypted at rest using Fernet keyed to a machine-local secret (not hardcoded), and the raw key never appears in logs or API responses.

## Tasks / Subtasks

- [x] Task 1: Update ProviderConfig model schema (AC: 1)
  - [x] Subtask 1.1: Add new columns to ProviderConfig model
  - [x] Subtask 1.2: Add tier_assignment JSON field
  - [x] Subtask 1.3: Add model_list JSON field
- [x] Task 2: Implement Fernet encryption for API keys (AC: 2)
  - [x] Subtask 2.1: Create encryption utility module
  - [x] Subtask 2.2: Add machine-local key generation using machine UUID
  - [x] Subtask 2.3: Encrypt on write, decrypt on read
- [x] Task 3: Update API endpoints for new schema (AC: 1, 2)
  - [x] Subtask 3.1: Update response models to exclude raw API keys
  - [x] Subtask 3.2: Add tier_assignment in requests/responses
  - [x] Subtask 3.3: Add test endpoint for provider validation
- [x] Task 4: Add database migration
  - [x] Subtask 4.1: Create migration for new columns
  - [x] Subtask 4.2: Handle existing data migration

## Dev Notes

### Project Structure Notes

- **Model location:** `src/database/models/provider_config.py`
- **API location:** `src/api/provider_config_endpoints.py`
- **Encryption:** Use `cryptography.fernet` package

### Key Architectural Context

- **Current state (from audit):**
  - ProviderConfig table exists with basic schema
  - API keys stored as plain text (need encryption)
  - No tier_assignment or model_list fields
- **Required changes:**
  - Add encryption using Fernet
  - Add tier_assignment JSON field
  - Add model_list JSON field
  - Machine-local secret from machine UUID (not hardcoded)

### Critical Rules from Project Context

1. **Encryption**: Use `cryptography` package with Fernet
2. **Machine key**: Derive from machine UUID, store in secure local config
3. **Never log API keys**: Ensure encrypted keys don't appear in logs
4. **Python imports**: Use `src.` prefix from project root

### Testing Standards Summary

- Python: pytest with `asyncio_mode = auto`
- Test encryption/decryption roundtrip
- Test API key masking in responses

### References

- Epic 2 overview: `docs/epics.md#Epic-2`
- Audit findings: `2-0-provider-infrastructure-audit-findings.md`
- Project Context §2.6 (Environment Variables)

---

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (via Claude Code)

### Debug Log References

N/A — Implementation story

### Completion Notes List

- Added Fernet encryption for API keys at rest using machine-local key
- Updated ProviderConfig model with new fields: provider_type, display_name, tier_assignment, model_list
- Updated API endpoints to exclude raw API keys from responses
- Added provider test endpoint (/api/providers/test) for validation
- Created migration file for existing databases
- Added cryptography package to requirements.txt
- Legacy field aliases maintain backward compatibility

### File List

**Files modified:**
- `requirements.txt` — Added cryptography>=3.4.0
- `src/database/models/provider_config.py` — Added encrypted API key, new fields
- `src/api/provider_config_endpoints.py` — Updated response models, added test endpoint

**Files created:**
- `src/database/encryption.py` — Fernet encryption utilities
- `src/database/migrations/add_provider_config_extended_fields.py` — Schema migration
