# Bot Lifecycle Management

## Overview

QuantMind uses an automated tag-based progression system to manage bot lifecycle from initial deployment to live trading. This system ensures only proven, well-performing bots make it to production.

## Tag Progression System

Bots progress through the following lifecycle tags:

```
@primal → @pending → @perfect → @live
                 ↘         ↘
                  @quarantine → @dead
```

### Tag Meanings

| Tag | Description |
|-----|-------------|
| `@primal` | New bots in initial validation phase |
| `@pending` | Bots showing promise, under observation |
| `@perfect` | Well-performing bots ready for live promotion |
| `@live` | Production bots trading real capital |
| `@quarantine` | Underperforming bots under review |
| `@dead` | Retired bots no longer trading |

## Progression Criteria

### @primal → @pending

To advance from primal to pending, a bot must meet ALL criteria:

| Metric | Requirement |
|--------|-------------|
| Minimum Trades | 20 |
| Win Rate | ≥ 50% |
| Days Active | ≥ 7 |
| Critical Errors | 0 |

### @pending → @perfect

| Metric | Requirement |
|--------|-------------|
| Minimum Trades | 50 |
| Win Rate | ≥ 55% |
| Sharpe Ratio | ≥ 1.5 |
| Days Active | ≥ 30 |
| Max Drawdown | < 15% |

### @perfect → @live

| Metric | Requirement |
|--------|-------------|
| Minimum Trades | 100 |
| Win Rate | ≥ 58% |
| Sharpe Ratio | ≥ 2.0 |
| Days Active | ≥ 60 |
| Max Drawdown | < 10% |
| Profit Factor | > 1.5 |

## Quarantine Triggers

Bots are automatically quarantined when ANY of these conditions occur:

| Trigger | Threshold |
|---------|-----------|
| Win Rate Drop | < 45% |
| Sharpe Ratio | < 0.5 |
| Drawdown | > 20% |
| Consecutive Losing Days | ≥ 5 |

## Quarantine to Dead

Bots in quarantine are marked as dead when:

- 30 days in quarantine without improvement
- Win rate drops below 40%
- Total loss exceeds 50% of allocated capital

## Monitoring Lifecycle Status

### API Endpoints

**Get all bots lifecycle status:**
```bash
GET /api/lifecycle/status
```

**Get specific bot status:**
```bash
GET /api/lifecycle/status/{bot_id}
```

**Manually trigger lifecycle check:**
```bash
POST /api/lifecycle/check
```

### Manual Override (Admin Only)

**Promote a bot manually:**
```bash
POST /api/lifecycle/promote/{bot_id}
```

**Quarantine a bot manually:**
```bash
POST /api/lifecycle/quarantine/{bot_id}
```

## Daily Lifecycle Check

The system runs an automated daily check at **3:00 AM UTC** (after HMM training completes). This check:

1. Evaluates all registered bots
2. Applies progression criteria
3. Updates tags for qualifying bots
4. Emits Prometheus metrics for monitoring
5. Sends WebSocket notifications to UI

## Bot Limit Tiers

Account balance determines how many bots can run simultaneously:

| Tier | Balance Range | Max Bots |
|------|---------------|----------|
| 0 | $0 - $50 | 0 (trading disabled) |
| 1 | $50 - $100 | 1 |
| 2 | $100 - $200 | 2 |
| 3 | $200 - $500 | 3 |
| 4 | $500 - $1,000 | 5 |
| 5 | $1,000 - $5,000 | 10 |
| 6 | $5,000+ | 20 |

### Checking Bot Limits

```bash
GET /api/bot-limits/status
```

Returns:
```json
{
  "tier": 3,
  "max_bots": 3,
  "active_bots": 2,
  "can_add_bot": true,
  "warning": null
}
```

## Best Practices

1. **Monitor Daily**: Check lifecycle status dashboard daily
2. **Review Quarantined Bots**: Investigate why bots were quarantined
3. **Don't Rush Promotion**: Let bots prove themselves naturally
4. **Capital Management**: Ensure adequate capital for desired bot count
5. **Performance Review**: Regularly review performance metrics before promotions

## Troubleshooting

### Bot Not Progressing

If a bot isn't advancing:

1. Check if it meets minimum trade count
2. Verify win rate is above threshold
3. Ensure no critical errors occurred
4. Confirm sufficient days active

### Bot Quarantined Unexpectedly

If a bot was quarantined:

1. Review recent trading activity
2. Check for market condition changes
3. Analyze consecutive losing trades
4. Consider adjusting strategy parameters

### Bot Limit Reached

If you can't add more bots:

1. Check your account balance
2. Review active bot count
3. Consider retiring underperforming bots
4. Add capital to unlock higher tiers

## Related Documentation

- [API Documentation](../api/lifecycle_endpoints.md)
- [Architecture Guide](../architecture/lifecycle_management.md)
- [Bot Manifest](../architecture/bot_manifest.md)