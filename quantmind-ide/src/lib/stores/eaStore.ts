import { writable, type Writable } from 'svelte/store';
import type { EADetails } from '$lib/types';

// Define the structure for EA (Expert Advisor) details
export interface EADetails {
    id: string;
    name: string;
    status: 'created' | 'validating' | 'validated' | 'backtesting' | 'backtested' | 'stress_testing' | 'stress_tested' | 'monte_carlo' | 'monte_carlo_complete' | 'paper_deployed' | 'monitoring' | 'stopped' | 'error';
    strategyCode: string;
    parameters: Record<string, any>;
    createdAt: number;
    updatedAt: number;
    backtestResults?: {
        metrics: Record<string, any>;
        equityCurve: Array<{ time: number; equity: number }>;
        drawdown: number;
        sharpeRatio: number;
        sortinoRatio: number;
    };
    paperTradingId?: string;
    currentEquity?: number;
    unrealizedPnl?: number;
}

// Define the structure for the EA store
interface EAStore {
    eas: Writable<EADetails[]>;
    listEAs: () => EADetails[];
    createEA: (name: string, strategyCode: string, parameters: Record<string, any>) => EADetails;
    updateEA: (id: string, updates: Partial<EADetails>) => void;
    getEA: (id: string) => EADetails | undefined;
    deleteEA: (id: string) => void;
}

// Create the EA store
function createEAStore(): EAStore {
    const { subscribe, set, update } = writable<EADetails[]>([]);

    return {
        eas: { subscribe, set, update },

        listEAs: () => {
            let eas: EADetails[] = [];
            subscribe(currentEAs => {
                eas = [...currentEAs];
            })();
            return eas;
        },

        createEA: (name: string, strategyCode: string, parameters: Record<string, any>): EADetails => {
            const newEA: EADetails = {
                id: `ea_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                name,
                status: 'created',
                strategyCode,
                parameters,
                createdAt: Date.now(),
                updatedAt: Date.now()
            };

            update(currentEAs => [...currentEAs, newEA]);
            return newEA;
        },

        updateEA: (id: string, updates: Partial<EADetails>): void => {
            update(currentEAs =>
                currentEAs.map(ea =>
                    ea.id === id ? { ...ea, ...updates, updatedAt: Date.now() } : ea
                )
            );
        },

        getEA: (id: string): EADetails | undefined => {
            let foundEA: EADetails | undefined;
            subscribe(currentEAs => {
                foundEA = currentEAs.find(ea => ea.id === id);
            })();
            return foundEA;
        },

        deleteEA: (id: string): void => {
            update(currentEAs => currentEAs.filter(ea => ea.id !== id));
        }
    };
}

export const eaStore = createEAStore();