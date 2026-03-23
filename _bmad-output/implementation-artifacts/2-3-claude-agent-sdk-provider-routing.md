# Story 2.3: Claude Agent SDK Provider Routing

Status: done

## Story

As a system administrator configuring QUANTMINDX,
I want the system to route AI requests to different providers based on configuration,
So that I can switch between providers (Anthropic, OpenAI, Google, etc.) without code changes.

## Acceptance Criteria

1. [AC-1] Given provider configurations exist in database, when an AI request is made, then it uses the configured provider's API endpoint.

2. [AC-2] Given multiple providers are configured, when a request is made, then it selects the primary provider by default.

3. [AC-3] Given a specific provider is specified in request, when processed, then it routes to that provider's endpoint.

4. [AC-4] Given provider has invalid/missing API key, when request is made, then it returns clear error.

5. [AC-5] Given fallback provider is configured, when primary provider fails, then it automatically retries with fallback.

## Tasks / Subtasks

- [x] Task 1: Create ProviderRouter class
  - [x] Subtask 1.1: Define routing logic based on config
  - [x] Subtask 1.2: Add primary/fallback selection
- [x] Task 2: Integrate with Claude Agent SDK
  - [x] Subtask 2.1: Create client wrapper with base_url support
  - [x] Subtask 2.2: Handle provider-specific request format
- [x] Task 3: Add error handling and fallbacks
  - [x] Subtask 3.1: Handle connection failures
  - [x] Subtask 3.2: Implement fallback retry logic

## Dev Notes

### Provider Routing

- Primary provider selected from DB where `is_primary=true`
- Fallback provider: second active provider by `created_at`
- Each provider has `base_url` in metadata for SDK configuration
- Timeout configurable per provider (default: 60s)

### Base URL Mapping

| Provider | Default Base URL |
|----------|-----------------|
| anthropic | https://api.anthropic.com |
| openai | https://api.openai.com/v1 |
| google | https://generativelanguage.googleapis.com/v1 |
| cohere | https://api.cohere.ai/v1 |
| mistral | https://api.mistral.ai/v1 |

### Implementation Location

- `src/agents/providers/router.py` - Main routing logic
- `src/agents/providers/client.py` - SDK client wrapper