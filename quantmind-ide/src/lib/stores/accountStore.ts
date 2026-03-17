/**
 * Account Store - Manages active account state
 */

import { writable, derived, get } from 'svelte/store';
import { API_BASE } from '$lib/constants';

const apiBase = API_BASE || '';

export interface BrokerAccount {
  broker_id: string;
  broker_name: string;
  account_id: string;
  server: string;
  account_type: string;
  balance: number;
  equity: number;
  margin?: number;
  leverage?: number;
  currency: string;
  connected: boolean;
  is_active?: boolean;
  status?: string;
}

interface AccountStoreState {
  accounts: BrokerAccount[];
  activeAccount: BrokerAccount | null;
  loading: boolean;
  error: string | null;
}

function createAccountStore() {
  const { subscribe, set, update } = writable<AccountStoreState>({
    accounts: [],
    activeAccount: null,
    loading: false,
    error: null
  });

  return {
    subscribe,

    /**
     * Fetch all available accounts
     */
    async fetchAccounts() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/trading/broker-accounts`);
        if (!response.ok) {
          throw new Error(`Failed to fetch accounts: ${response.statusText}`);
        }
        const accounts = await response.json();
        update(state => ({
          ...state,
          accounts,
          loading: false,
          activeAccount: accounts.find((a: BrokerAccount) => a.is_active) || null
        }));
      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        }));
      }
    },

    /**
     * Switch to a different account
     */
    async switchAccount(accountId: string) {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/brokers/accounts/active`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ account_id: accountId })
        });

        if (!response.ok) {
          throw new Error(`Failed to switch account: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.success) {
          update(state => ({
            ...state,
            loading: false,
            activeAccount: result.account ? {
              ...result.account,
              account_id: result.account.account_id || accountId,
              broker_name: result.account.broker_name || result.account.broker_name
            } : null
          }));

          // Also update the is_active flag in accounts list
          update(state => ({
            ...state,
            accounts: state.accounts.map(a => ({
              ...a,
              is_active: a.account_id === accountId
            }))
          }));
        } else {
          throw new Error(result.message || 'Failed to switch account');
        }
      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        }));
      }
    },

    /**
     * Clear the active account
     */
    async clearActiveAccount() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/brokers/accounts/active`, {
          method: 'DELETE'
        });

        if (!response.ok) {
          throw new Error(`Failed to clear account: ${response.statusText}`);
        }

        update(state => ({
          ...state,
          loading: false,
          activeAccount: null,
          accounts: state.accounts.map(a => ({ ...a, is_active: false }))
        }));
      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        }));
      }
    },

    /**
     * Initialize the store
     */
    async initialize() {
      await this.fetchAccounts();
    }
  };
}

export const accountStore = createAccountStore();

// Derived stores for convenience
export const accounts = derived(accountStore, $store => $store.accounts);
export const activeAccount = derived(accountStore, $store => $store.activeAccount);
export const accountLoading = derived(accountStore, $store => $store.loading);
export const accountError = derived(accountStore, $store => $store.error);

// Filtered accounts
export const connectedAccounts = derived(
  accountStore,
  $store => $store.accounts.filter(a => a.connected)
);
