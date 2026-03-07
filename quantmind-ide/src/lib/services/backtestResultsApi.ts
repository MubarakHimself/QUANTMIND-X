import { getJson, postJson } from "$lib/services/componentApi";

export function listBacktestResults() {
  return getJson<any[]>("/v1/backtest/results");
}

export function runMonteCarloSimulation(payload: unknown) {
  return postJson<any>("/v1/backtest/monte-carlo", payload);
}

export function calculatePbo(payload: unknown) {
  return postJson<any>("/v1/backtest/pbo/calculate", payload);
}
