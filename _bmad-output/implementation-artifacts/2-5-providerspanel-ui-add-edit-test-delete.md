# Story 2.5: ProvidersPanel UI - Add, Edit, Test, Delete

Status: done

## Story

As a user configuring QUANTMINDX,
I want to add, edit, test, and delete provider configurations through the UI,
So that I can manage AI providers without using the API directly.

## Acceptance Criteria

1. [AC-1] Given the ProvidersPanel is open, when it loads, then it fetches providers from /api/providers and displays them.

2. [AC-2] Given user clicks "Add Provider", when processed, then a modal form appears with provider_type, base_url, api_key fields.

3. [AC-3] Given user enters provider details and clicks Save, when processed, then POST /api/providers is called and provider appears in list.

4. [AC-4] Given user clicks Test on a configured provider, when processed, then POST /api/providers/test is called and shows success/failure.

5. [AC-5] Given user clicks Delete on a provider, when processed, then DELETE /api/providers/{id} is called after confirmation.

6. [AC-6] Given user clicks Edit on a provider, when processed, then fields become editable and Save/Cancel appear.

## Tasks / Subtasks

- [x] Task 1: Update ProvidersPanel to use new API format
  - [x] Subtask 1.1: Parse /api/providers response correctly
  - [x] Subtask 1.2: Handle provider_type instead of id
- [x] Task 2: Add Test functionality
  - [x] Subtask 2.1: Call /api/providers/test endpoint
  - [x] Subtask 2.2: Show latency and success/failure
- [x] Task 3: Add confirmation for delete
  - [x] Subtask 3.1: Show confirmation dialog
  - [x] Subtask 3.2: Handle 409 response gracefully

## Dev Notes

### API Changes Needed

The old API expected `name` field, but new API expects `provider_type`.
The legacy alias in the API handles this, but we should use the correct field.

### Response Format

/api/providers returns:
```json
{
  "providers": [
    {
      "id": "uuid",
      "provider_type": "anthropic",
      "display_name": "Anthropic Claude",
      "base_url": "https://api.anthropic.com/v1",
      "is_active": true,
      "model_count": 6
    }
  ]
}
```

### Component Location

- `quantmind-ide/src/lib/components/settings/ProvidersPanel.svelte`