# Story 2.2: Server Connection Configuration CRUD

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system administrator configuring QUANTMINDX,
I want a CRUD API for server connections (Cloudzy/Contabo/MT5),
So that I can manage server configurations through the UI without editing environment files.

## Acceptance Criteria

1. [AC-1] Given no server configs exist, when GET /api/servers is called, then it returns an empty list `[]`.

2. [AC-2] Given POST /api/servers with valid payload, when processed, then server config is created with UUID, stored in database, and returns `{ id, name, server_type, is_active }`.

3. [AC-3] Given PUT /api/servers/{id} is called, when processed, then only provided fields are updated; sensitive fields (passwords) are encrypted at rest.

4. [AC-4] Given DELETE /api/servers/{id} is called for a server marked as primary, then 409 Conflict is returned.

5. [AC-5] Given POST /api/servers/{id}/test is called, when processed, then it tests connectivity and returns `{ success: true, latency_ms }` or `{ success: false, error }`.

## Tasks / Subtasks

- [x] Task 1: Create ServerConfig database model
  - [x] Subtask 1.1: Define model with server_type enum
  - [x] Subtask 1.2: Add encryption for sensitive fields
- [x] Task 2: Create /api/servers CRUD endpoints
  - [x] Subtask 2.1: GET /api/servers - list all
  - [x] Subtask 2.2: POST /api/servers - create new
  - [x] Subtask 2.3: PUT /api/servers/{id} - update
  - [x] Subtask 2.4: DELETE /api/servers/{id} - delete
- [x] Task 3: Add test endpoint for server connectivity
  - [x] Subtask 3.1: POST /api/servers/{id}/test
  - [x] Subtask 3.2: Test SSH/API connectivity

## Dev Notes

### Server Types

- **cloudzy**: Live trading server (MT5)
- **contabo**: Agent/compute server
- **mt5**: MetaTrader 5 connection

### Model Fields

- id: UUID
- name: Display name
- server_type: enum (cloudzy, contabo, mt5)
- host: hostname/IP
- port: port number
- username: auth username (encrypted)
- password: auth password (encrypted)
- ssh_key_path: path to SSH key (optional)
- is_active: bool
- is_primary: bool (only one primary per type)
- metadata: JSON for extra config

### Encryption

Sensitive fields (password, ssh_key_path content) encrypted using same Fernet system as provider API keys.

### References

- Epic 2 overview: `docs/epics.md#Epic-2`
- Server config from audit: Contabo/Cloudzy env vars
