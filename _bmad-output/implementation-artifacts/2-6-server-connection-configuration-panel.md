# Story 2.6: Server Connection Configuration Panel

Status: done

## Story

As a user configuring QUANTMINDX,
I want to manage server connections (Cloudzy, Contabo, MT5) through the UI,
So that I can configure infrastructure without API calls.

## Acceptance Criteria

1. [AC-1] Given the Server panel is open, when it loads, then it fetches servers from /api/servers and displays them.

2. [AC-2] Given user clicks "Add Server", when processed, then a modal form appears with server details.

3. [AC-3] Given user enters server details and clicks Save, when processed, then POST /api/servers is called.

4. [AC-4] Given user clicks Test on a server, when processed, then POST /api/servers/{id}/test is called and shows result.

5. [AC-5] Given user clicks Delete on a server, when processed, then DELETE /api/servers/{id} is called after confirmation (unless primary).

6. [AC-6] Given a server is marked as primary, when displayed, then it shows a primary badge.

## Tasks / Subtasks

- [x] Task 1: Create ServersPanel component
  - [x] Subtask 1.1: Build list of servers from /api/servers
  - [x] Subtask 1.2: Add create/edit/delete functionality
- [x] Task 2: Add test connectivity
  - [x] Subtask 2.1: Call /api/servers/{id}/test
  - [x] Subtask 2.2: Show latency and status
- [x] Task 3: Integrate into SettingsView
  - [x] Subtask 3.1: Import ServersPanel in SettingsView
  - [x] Add servers tab to settings tabs

## Dev Notes

### Server Types

- cloudzy: Live trading server (MT5)
- contabo: Agent/compute server
- mt5: MetaTrader 5 connection

### API Endpoints

- GET /api/servers - List all servers
- POST /api/servers - Create server
- PUT /api/servers/{id} - Update server
- DELETE /api/servers/{id} - Delete server (409 if primary)
- POST /api/servers/{id}/test - Test connectivity

### Component Location

- Create: `quantmind-ide/src/lib/components/settings/ServersPanel.svelte`
- Update: `quantmind-ide/src/lib/components/SettingsView.svelte`