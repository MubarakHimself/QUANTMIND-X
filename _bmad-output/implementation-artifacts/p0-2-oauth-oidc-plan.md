# P0-2: OAuth/OIDC Authentication — Implementation Plan

## Current State Assessment

**Existing Auth**: None. The codebase has no user authentication:
- No `User` model in `src/database/models/`
- No auth middleware on any `/api/*` endpoints
- Frontend `chatApi.ts` uses plain `fetch()` with no auth headers
- Redis `GlobalCache` exists at `src/cache/redis_client.py` (ready for session storage)
- `Settings` class in `src/config.py` already has `secret_key` field

**Key Files**:
- `src/api/server.py` — Main FastAPI app, 2700+ lines, no auth middleware
- `src/cache/redis_client.py` — `GlobalCache` class with async Redis methods
- `src/config.py` — `Settings` with `secret_key`, `database_url`, `redis_url`
- `quantmind-ide/src/lib/api/` — TypeScript API clients (no auth headers)

---

## Phase 1: Architecture Design

### 1.1 OAuth 2.1 Provider Choice

**Primary: Auth0** (managed SaaS)
- OAuth 2.1 compliant (authorization code flow + PKCE)
- No self-hosted infrastructure
- Libraries: `python-jose` (JWT), `auth0` (SDK)
- Environment variables: `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`
- Free tier: 7,000 active users/month

**Alternative: Supabase Auth** (self-hosted optional)
- Similar DX to Auth0
- Can self-host for data sovereignty

**Decision: Auth0** — most production-ready with full OAuth 2.1/OIDC support.

### 1.2 Token Storage Strategy

**Tokens stored in httpOnly cookies** (not localStorage/JWT storage):
- Access token: 15-minute TTL, httpOnly cookie `access_token`
- Refresh token: stored server-side in Redis, linked via session ID cookie `session_id`
- SameSite=Strict for CSRF protection
- Secure flag in production

**Why httpOnly cookies over JWT storage**:
- XSS cannot steal tokens (httpOnly)
- CSRF mitigated with SameSite + CSRF tokens for state-changing ops
- OAuth provider handles token rotation

### 1.3 Session Management Approach

**Redis-backed sessions** using existing `GlobalCache`:
- Session key: `session:{session_id}` → Redis hash with:
  - `user_id`: Auth0 sub
  - `access_token`: encrypted access token
  - `refresh_token`: Auth0 refresh token
  - `expires_at`: Unix timestamp
  - `email`: user email
  - `roles`: list of application roles
- TTL: 7 days (refresh token lifetime)
- Cookie: `qm_session_id` (UUID), httpOnly, Secure, SameSite=Strict

---

## Phase 2: Backend Implementation

### 2.1 New Files

#### `src/auth/__init__.py`
Auth module exports.

#### `src/auth/config.py`
OAuth/OIDC configuration from environment:
```python
AUTH0_DOMAIN: str
AUTH0_CLIENT_ID: str
AUTH0_CLIENT_SECRET: str  # from env, never committed
AUTH0_AUDIENCE: str  # API identifier
AUTH0_CALLBACK_URL: str  # /api/auth/callback
AUTH0_LOGOUT_URL: str  # /api/auth/logout
```

#### `src/auth/models.py`
Pydantic models:
- `User` — extracted from ID token claims
- `OAuthTokens` — access/refresh token pair
- `AuthSession` — Redis session data

#### `src/auth/oauth.py`
Auth0 client implementation:
- `get_authorization_url(state, nonce)` — builds OAuth authorization URL with PKCE
- `exchange_code(code, verifier)` — exchanges authorization code for tokens
- `refresh_access_token(refresh_token)` — uses refresh token to get new access token
- `get_user_info(access_token)` — calls Auth0 /userinfo endpoint
- `logout(logout_token)` — calls Auth0 logout endpoint

#### `src/auth/session.py`
Redis session management:
- `create_session(user_info, tokens)` — creates session in Redis
- `get_session(session_id)` — retrieves session
- `refresh_session(session_id, new_tokens)` — updates tokens
- `delete_session(session_id)` — removes session (logout)
- `cleanup_expired_sessions()` — background task

#### `src/auth/middleware.py`
FastAPI middleware:
- `OAuthMiddleware` — extracts session from cookie, validates, attaches to request state
- Applied to all `/api/*` routes except `/api/auth/*`, `/health`, `/metrics`

#### `src/auth/dependencies.py`
FastAPI dependencies:
- `get_current_user()` — extracts validated user from request state
- `require_role(role)` — checks user has required role

### 2.2 New Router

#### `src/api/auth_endpoints.py`
Auth router with endpoints:
```
GET  /api/auth/login          → Redirect to Auth0 authorization URL
GET  /api/auth/callback       → Handle Auth0 callback, exchange code, set cookies
POST /api/auth/refresh         → Refresh access token using refresh token
GET  /api/auth/logout         → Clear session, redirect to Auth0 logout
GET  /api/auth/me             → Return current user info
POST /api/auth/migrate        → Link legacy user to OAuth account (migration)
```

### 2.3 Database Model Addition

#### `src/database/models/user.py` (new)
For existing user migration support:
```python
class OAuthAccount(Base):
    """Links OAuth identity to application user."""
    user_id: int
    provider: str  # "auth0"
    provider_user_id: str  # Auth0 sub
    email: str
    linked_at: datetime
```

### 2.4 Server Modifications

`src/api/server.py`:
1. Import and add `OAuthMiddleware` to FastAPI app
2. Include `auth_router` from `src/api/auth_endpoints`
3. Exclude auth routes from middleware (login, callback, logout)

---

## Phase 3: Frontend Integration

### 3.1 New Files

#### `quantmind-ide/src/lib/auth.ts`
Auth client module:
- `login()` — redirect to `/api/auth/login`
- `logout()` — call `/api/auth/logout`, clear local state
- `getCurrentUser()` — call `/api/auth/me`
- `refreshToken()` — call `/api/auth/refresh` (called automatically)
- `isAuthenticated()` — check if session cookie exists

#### `quantmind-ide/src/hooks.server.ts` (or create)
SvelteKit server hooks for auth:
- Check session cookie on every request
- Attach user to `event.locals`
- Protect `/api/*` routes requiring auth

### 3.2 Modifications

#### `quantmind-ide/src/lib/api/chatApi.ts` (and all other API clients)
- Add `credentials: 'include'` to fetch options (for cookies)
- Add request interceptor to call `refreshToken()` on 401

---

## Phase 4: Migration Plan

### 4.1 Existing User Migration Strategy

**Option A — Silent Link**: Existing users link via "Connect Auth0" in settings (one-time):
1. User logs in with existing credentials
2. User navigates to Settings → Connect Account
3. Initiates OAuth flow, Auth0 account linked to existing user record
4. After migration period, legacy login disabled

**Option B — Invite-based**: Admin creates Auth0 accounts, existing users receive invite email:
1. Admin exports user list
2. Admin bulk-creates Auth0 users via Management API
3. Users receive invite, set password in Auth0
4. Legacy system deprecated after 30 days

**Decision: Option A** — least disruptive, self-service.

### 4.2 Rollout Plan (Backwards Compatible)

1. **Week 1**: Deploy OAuth endpoints, no enforcement. All `/api/*` work as before.
2. **Week 2**: Enable middleware in ` permissive` mode (log unauthorized, allow through).
3. **Week 3**: Enforce auth on non-critical endpoints (`/api/knowledge/*`, `/api/settings/*`).
4. **Week 4**: Enforce auth on all `/api/*` endpoints.
5. **Week 5+**: Deprecate legacy login after >95% users migrated.

### 4.3 Migration Endpoints

```
POST /api/auth/migrate
Body: { legacy_token: string }
Response: { success: true, user_id: int }

GET  /api/auth/migrate/status
Response: { migrated: bool, email: str | null }
```

---

## File Manifest

### Backend (new)
- `src/auth/__init__.py`
- `src/auth/config.py`
- `src/auth/models.py`
- `src/auth/oauth.py`
- `src/auth/session.py`
- `src/auth/middleware.py`
- `src/auth/dependencies.py`
- `src/api/auth_endpoints.py`

### Backend (modified)
- `src/api/server.py` — add OAuthMiddleware, auth_router
- `src/config.py` — add AUTH0_* settings
- `src/database/models/user.py` (new OAuthAccount model)

### Frontend (new)
- `quantmind-ide/src/lib/auth.ts`

### Frontend (modified)
- `quantmind-ide/src/hooks.server.ts` (create if not exists)
- `quantmind-ide/src/lib/api/chatApi.ts` and all API clients — add `credentials: 'include'`

---

## Environment Variables Required

```env
# Auth0 OAuth 2.1
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret  # NEVER commit
AUTH0_AUDIENCE=https://api.quantmindx.com
AUTH0_CALLBACK_URL=http://localhost:8000/api/auth/callback

# Session
SESSION_COOKIE_SECRET=random_32_char_string
OAUTH_PKCE_CODE_VERIFIER_LENGTH=64
```

---

## Dependencies

```txt
# Backend
python-jose[cryptography]==3.3.0  # JWT handling
httpx==0.27.0  # Async HTTP client for Auth0 API calls
auth0-python==4.6.0  # Auth0 SDK (optional, can use httpx directly)
cryptography==42.0.0  # For token encryption at rest

# Frontend (no new deps needed — uses built-in fetch)
```

---

## Security Considerations

1. **PKCE** — Required for OAuth 2.1, protects against authorization code interception
2. **httpOnly cookies** — Access tokens never exposed to JavaScript
3. **SameSite=Strict** — CSRF protection for cookies
4. **Secure flag** — Cookies only sent over HTTPS in production
5. **Token encryption** — Refresh tokens encrypted at rest in Redis
6. **Short-lived access tokens** — 15-minute TTL limits exposure
7. **Session rotation** — New session ID on each token refresh
