/**
 * QuantMind IDE OAuth 2.1 Authentication Client
 *
 * Handles authentication flow using httpOnly cookies for session management.
 * Tokens are stored server-side in Redis, never in localStorage.
 */

import { API_CONFIG } from './config/api';

const API_BASE = `${API_CONFIG.API_URL}/api`;

export interface AuthUser {
  sub: string;
  email: string;
  email_verified: boolean;
  name?: string;
  nickname?: string;
  picture?: string;
  roles: string[];
}

export interface AuthMeResponse {
  user: AuthUser;
  session_id: string;
  expires_at: string;
}

export interface AuthRefreshResponse {
  success: boolean;
  expires_in: number;
  expires_at: string;
}

export interface AuthLogoutResponse {
  success: boolean;
  message: string;
}

export interface AuthMigrateResponse {
  success: boolean;
  message: string;
  user_id?: number;
}

export interface AuthMigrateStatus {
  migrated: boolean;
  oauth_connected: boolean;
  email: string | null;
}

const AUTH_COOKIE = 'qm_session_id';

/**
 * Get the current session ID from cookies.
 */
export function getSessionId(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp(`(?:^|; )${AUTH_COOKIE}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

/**
 * Check if the user is authenticated.
 */
export function isAuthenticated(): boolean {
  return getSessionId() !== null;
}

/**
 * Redirect to the OAuth login page.
 */
export function login(): void {
  window.location.href = `${API_BASE}/auth/login`;
}

/**
 * Log out the current user.
 */
export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    });
  } finally {
    // Clear local state regardless of API result
    clearAuthState();
    window.location.href = '/';
  }
}

/**
 * Get the current authenticated user.
 * Returns null if not authenticated.
 */
export async function getCurrentUser(): Promise<AuthUser | null> {
  try {
    const response = await fetch(`${API_BASE}/auth/me`, {
      method: 'GET',
      credentials: 'include',
    });

    if (response.status === 401) {
      return null;
    }

    if (!response.ok) {
      throw new Error(`Auth check failed: ${response.status}`);
    }

    const data: AuthMeResponse = await response.json();
    return data.user;
  } catch (error) {
    console.error('Failed to get current user:', error);
    return null;
  }
}

/**
 * Check if the access token needs refresh (expires within 2 minutes).
 */
export async function refreshTokenIfNeeded(): Promise<boolean> {
  const sessionId = getSessionId();
  if (!sessionId) return false;

  try {
    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    if (response.ok) {
      const data: AuthRefreshResponse = await response.json();
      return data.success;
    }

    return false;
  } catch (error) {
    console.error('Token refresh failed:', error);
    return false;
  }
}

/**
 * Migrate a legacy user account to OAuth.
 */
export async function migrateLegacyUser(legacyToken: string): Promise<AuthMigrateResponse> {
  const response = await fetch(`${API_BASE}/auth/migrate`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ legacy_token: legacyToken }),
  });

  if (!response.ok) {
    throw new Error(`Migration failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Get migration status for the current user.
 */
export async function getMigrationStatus(): Promise<AuthMigrateStatus> {
  const response = await fetch(`${API_BASE}/auth/migrate/status`, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error(`Migration status check failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Clear local authentication state.
 * Called after logout.
 */
function clearAuthState(): void {
  // Clear cookie via expired maxAge
  document.cookie = `${AUTH_COOKIE}=; path=/; max-age=0`;
}

/**
 * Auth error handler for API responses.
 * Returns true if the error indicates the user is not authenticated.
 */
export function isAuthError(error: unknown): boolean {
  if (error instanceof Error) {
    // Check for common auth error patterns
    const msg = error.message.toLowerCase();
    return (
      msg.includes('401') ||
      msg.includes('unauthorized') ||
      msg.includes('authentication required')
    );
  }
  return false;
}

/**
 * Create an authenticated fetch wrapper that:
 * 1. Automatically includes credentials (cookies)
 * 2. Attempts token refresh on 401
 * 3. Redirects to login on auth failure
 */
export async function authFetch<T>(
  input: RequestInfo,
  init?: RequestInit
): Promise<T> {
  const url = typeof input === 'string' ? input : input.url;
  const options: RequestInit = {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  };

  let response = await fetch(input, options);

  // If we got a 401, try to refresh the token
  if (response.status === 401) {
    const refreshed = await refreshTokenIfNeeded();
    if (refreshed) {
      // Retry the original request
      response = await fetch(input, options);
    } else {
      // Refresh failed, redirect to login
      login();
      throw new Error('Authentication required');
    }
  }

  if (!response.ok) {
    let errorMessage = `API Error: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      // If JSON parsing fails, use the default error message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}
