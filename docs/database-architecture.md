# QuantMindX Database Architecture

## Database Technologies

### SQLite
- **Purpose:** Primary transactional data store
- **Location:** `data/quantmind.db`
- **Usage:** User data, settings, agent state, trading positions

### DuckDB
- **Purpose:** Analytics and OLAP queries
- **Location:** `data/analytics.duckdb`
- **Usage:** Historical analysis, backtesting results, performance metrics

### Redis
- **Purpose:** Caching and real-time messaging
- **Location:** localhost:6379 (configurable)
- **Usage:** Session cache, pub/sub for live updates, rate limiting
- **Note:** Currently not actively used - review if needed
