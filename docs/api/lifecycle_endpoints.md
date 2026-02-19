# Lifecycle API Endpoints

## Overview

The Lifecycle API provides endpoints for managing bot lifecycle progression, checking status, and manually controlling bot tags.

## Base URL

```
http://localhost:8000/api/lifecycle
```

## Authentication

All endpoints require authentication. Include your API key in the header:

```http
Authorization: Bearer <your-api-key>
```

Admin-only endpoints require elevated permissions.

---

## Endpoints

### Get All Bots Lifecycle Status

Returns lifecycle status for all registered bots.

**Endpoint:** `GET /api/lifecycle/status`

**Response:**
```json
{
  "bots": [
    {
      "bot_id": "london_breakout_01",
      "name": "London Breakout EURUSD",
      "current_tag": "@pending",
      "next_tag": "@perfect",
      "meets_criteria": false,
      "days_active": 25,
      "progress": {
        "trades": 45,
        "trades_required": 50,
        "win_rate": 0.58,
        "win_rate_required": 0.55,
        "sharpe_ratio": 1.6,
        "sharpe_required": 1.5,
        "days_remaining": 5
      },
      "last_check": "2026-02-16T03:00:00Z"
    }
  ],
  "total": 1,
  "last_check": "2026-02-16T03:00:00Z"
}
```

---

### Get Specific Bot Status

Returns detailed lifecycle status for a single bot.

**Endpoint:** `GET /api/lifecycle/status/{bot_id}`

**Parameters:**
| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| bot_id | string | path | Yes | Unique bot identifier |

**Response:**
```json
{
  "bot_id": "london_breakout_01",
  "name": "London Breakout EURUSD",
  "current_tag": "@pending",
  "tag_history": [
    {"tag": "@primal", "assigned_at": "2026-01-01T00:00:00Z"},
    {"tag": "@pending", "assigned_at": "2026-01-15T03:00:00Z"}
  ],
  "performance": {
    "total_trades": 45,
    "win_rate": 0.58,
    "sharpe_ratio": 1.6,
    "max_drawdown": 0.08,
    "profit_factor": 1.8,
    "consecutive_losing_days": 0
  },
  "next_progression": {
    "target_tag": "@perfect",
    "meets_criteria": false,
    "missing_criteria": ["min_trades", "min_days_active"],
    "estimated_date": "2026-02-21T00:00:00Z"
  },
  "quarantine_risk": {
    "at_risk": false,
    "triggers": []
  }
}
```

**Error Response (404):**
```json
{
  "error": "Bot not found",
  "bot_id": "invalid_bot_id"
}
```

---

### Trigger Lifecycle Check

Manually trigger a lifecycle check for all bots.

**Endpoint:** `POST /api/lifecycle/check`

**Response:**
```json
{
  "status": "completed",
  "checked": 15,
  "promoted": 2,
  "quarantined": 0,
  "killed": 0,
  "results": [
    {
      "bot_id": "bot_001",
      "action": "promoted",
      "from_tag": "@pending",
      "to_tag": "@perfect"
    },
    {
      "bot_id": "bot_002",
      "action": "promoted",
      "from_tag": "@primal",
      "to_tag": "@pending"
    }
  ],
  "timestamp": "2026-02-16T10:30:00Z"
}
```

---

### Manually Promote Bot

Promote a bot to the next lifecycle tag (Admin only).

**Endpoint:** `POST /api/lifecycle/promote/{bot_id}`

**Parameters:**
| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| bot_id | string | path | Yes | Unique bot identifier |

**Request Body (optional):**
```json
{
  "reason": "Manual override - exceptional performance",
  "skip_validation": false
}
```

**Response:**
```json
{
  "success": true,
  "bot_id": "london_breakout_01",
  "previous_tag": "@pending",
  "new_tag": "@perfect",
  "promoted_at": "2026-02-16T10:35:00Z",
  "promoted_by": "admin_user",
  "reason": "Manual override - exceptional performance"
}
```

**Error Response (403):**
```json
{
  "error": "Admin access required",
  "message": "This endpoint requires elevated permissions"
}
```

**Error Response (400):**
```json
{
  "error": "Cannot promote bot",
  "message": "Bot is already at @live tag"
}
```

---

### Manually Quarantine Bot

Move a bot to quarantine (Admin only).

**Endpoint:** `POST /api/lifecycle/quarantine/{bot_id}`

**Parameters:**
| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| bot_id | string | path | Yes | Unique bot identifier |

**Request Body:**
```json
{
  "reason": "Suspicious trading pattern detected",
  "duration_hours": 48
}
```

**Response:**
```json
{
  "success": true,
  "bot_id": "suspicious_bot_01",
  "previous_tag": "@live",
  "new_tag": "@quarantine",
  "quarantined_at": "2026-02-16T10:40:00Z",
  "quarantined_by": "admin_user",
  "reason": "Suspicious trading pattern detected",
  "auto_release_at": "2026-02-18T10:40:00Z"
}
```

---

## Bot Limits Endpoints

### Get Bot Limit Status

Returns current bot limit tier and status.

**Endpoint:** `GET /api/bot-limits/status`

**Response:**
```json
{
  "tier": 3,
  "tier_name": "Medium Account",
  "balance_range": {
    "min": 200,
    "max": 500
  },
  "max_bots": 3,
  "active_bots": 2,
  "can_add_bot": true,
  "remaining_slots": 1,
  "safety_buffer": {
    "required_capital": 300,
    "current_capital": 350,
    "buffer_available": 50
  },
  "warning": null
}
```

**Response (at limit):**
```json
{
  "tier": 2,
  "tier_name": "Small Account",
  "balance_range": {
    "min": 100,
    "max": 200
  },
  "max_bots": 2,
  "active_bots": 2,
  "can_add_bot": false,
  "remaining_slots": 0,
  "warning": "Bot limit reached. Add capital to unlock more slots."
}
```

---

## Scanner Endpoints

### Get Scanner Alerts

Returns recent market scanner alerts.

**Endpoint:** `GET /api/scanner/alerts`

**Query Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| limit | int | No | 50 | Maximum alerts to return |
| type | string | No | all | Filter by alert type |
| session | string | No | all | Filter by trading session |

**Response:**
```json
{
  "alerts": [
    {
      "type": "session_breakout",
      "symbol": "EURUSD",
      "session": "LONDON",
      "setup": "Bullish breakout above 1.08500",
      "confidence": 0.85,
      "recommended_bots": ["london_breakout_01"],
      "priority": "high",
      "timestamp": "2026-02-16T08:35:00Z"
    }
  ],
  "total": 1,
  "filtered_by": {
    "type": "all",
    "session": "all"
  }
}
```

---

### Get Scanner Status

Returns current scanner operational status.

**Endpoint:** `GET /api/scanner/status`

**Response:**
```json
{
  "running": true,
  "current_session": "LONDON",
  "scan_interval_seconds": 300,
  "last_scan": "2026-02-16T08:30:00Z",
  "total_alerts_today": 5,
  "alerts_by_type": {
    "session_breakout": 2,
    "volatility_spike": 1,
    "ict_setup": 2
  }
}
```

---

### Trigger Manual Scan

Manually trigger a market scan.

**Endpoint:** `POST /api/scanner/scan`

**Request Body (optional):**
```json
{
  "symbols": ["EURUSD", "GBPUSD"],
  "scan_types": ["session_breakout", "ict_setup"]
}
```

**Response:**
```json
{
  "status": "completed",
  "alerts_found": 2,
  "alerts": [
    {
      "type": "ict_setup",
      "symbol": "EURUSD",
      "setup": "FVG: bullish at 1.08500",
      "confidence": 0.75
    }
  ],
  "scan_duration_ms": 150
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Admin access required |
| 404 | Not Found - Bot or resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| GET endpoints | 60/minute |
| POST endpoints | 10/minute |

## WebSocket Events

Lifecycle events are broadcast via WebSocket:

```javascript
ws://localhost:8000/ws/lifecycle
```

**Event Types:**
- `bot_promoted` - Bot advanced to next tag
- `bot_quarantined` - Bot moved to quarantine
- `bot_killed` - Bot marked as dead
- `lifecycle_check_complete` - Daily check finished

**Example Event:**
```json
{
  "type": "bot_promoted",
  "bot_id": "london_breakout_01",
  "from_tag": "@pending",
  "to_tag": "@perfect",
  "timestamp": "2026-02-16T03:00:00Z"
}
```

## Related Documentation

- [User Guide: Bot Lifecycle](../user-guide/bot_lifecycle.md)
- [Architecture: Lifecycle Management](../architecture/lifecycle_management.md)