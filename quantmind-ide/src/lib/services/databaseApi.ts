import { API_CONFIG } from "$lib/config/api";
import { fetchBlob, getJson, postForm, postJson, putJson } from "$lib/services/componentApi";

const API_BASE = API_CONFIG.API_BASE;

export function listDatabaseTables() {
  return getJson<{ tables?: any[] }>("/database/tables");
}

export function getDatabaseStats() {
  return getJson<any>("/database/stats");
}

export function getDatabaseTable(tableName: string, limit: number, offset: number) {
  return getJson<any>(`/database/table/${encodeURIComponent(tableName)}?limit=${limit}&offset=${offset}`);
}

export function getDatabaseSchema(tableName: string) {
  return getJson<{ columns?: any[] }>(`/database/schema/${encodeURIComponent(tableName)}`);
}

export function runDatabaseQuery(query: string) {
  return postJson<any>("/database/query", { query });
}

export function insertDatabaseRow(tableName: string, payload: unknown) {
  return postJson<any>(`/database/table/${encodeURIComponent(tableName)}`, payload);
}

export function updateDatabaseRow(tableName: string, payload: unknown) {
  return putJson<any>(`/database/table/${encodeURIComponent(tableName)}`, payload);
}

export async function deleteDatabaseRows(tableName: string, ids: string[]) {
  const response = await fetch(`${API_BASE}/database/table/${encodeURIComponent(tableName)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
}

export function exportDatabaseTable(tableName: string, format: "csv" | "json") {
  return fetchBlob(`/database/export/${encodeURIComponent(tableName)}?format=${format}`);
}

export function importDatabaseTable(tableName: string, formData: FormData) {
  return postForm<any>(`/database/import/${encodeURIComponent(tableName)}`, formData);
}
