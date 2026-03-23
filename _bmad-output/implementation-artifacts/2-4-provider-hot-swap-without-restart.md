# Story 2.4: Provider Hot Swap Without Restart

Status: done

## Story

As a system administrator managing QUANTMINDX,
I want the system to switch between providers without restarting the application,
So that I can update API keys or change providers with zero downtime.

## Acceptance Criteria

1. [AC-1] Given a provider API key is updated in database, when next request is made, then it uses the new key without restart.

2. [AC-2] Given a provider is disabled, when a request is made, then it automatically routes to fallback without error.

3. [AC-3] Given a new provider is added with is_active=true, when a request is made, then it becomes available immediately.

4. [AC-4] Given provider configuration cache TTL is 5 minutes, when config is updated, then it reflects within 5 minutes.

## Tasks / Subtasks

- [x] Task 1: Add cache refresh mechanism
  - [x] Subtask 1.1: Implement time-based cache invalidation
  - [x] Subtask 1.2: Add manual refresh endpoint
- [x] Task 2: Ensure thread-safe provider access
  - [x] Subtask 2.1: Use thread-local storage for provider config
  - [x] Subtask 2.2: Handle concurrent requests
- [x] Task 3: Add hot-swap notification
  - [x] Subtask 3.1: Log provider changes
  - [x] Subtask 3.2: Emit event on provider change

## Dev Notes

### Implementation

The ProviderRouter already implements refresh() method for manual refresh.
Default cache TTL is 5 minutes - configurable via environment.

### Changes to router.py

- Add last_refresh timestamp
- Check TTL before using cached config
- Add refresh() method that's called automatically

### API Endpoint

- POST /api/providers/refresh - Force refresh provider config