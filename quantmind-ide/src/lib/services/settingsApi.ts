import { deleteRequest, getJson, patchJson, postJson } from "$lib/services/componentApi";

export function createSettingKey(payload: unknown) {
  return postJson<any>("/settings/keys", payload);
}

export function deleteSettingKey(id: string) {
  return deleteRequest(`/settings/keys/${encodeURIComponent(id)}`);
}

export function listSettingKeys() {
  return getJson<any[]>("/settings/keys");
}

export function createMcpServer(payload: unknown) {
  return postJson<any>("/settings/mcp", payload);
}

export function updateMcpServer(id: string, payload: unknown) {
  return patchJson<any>(`/settings/mcp/${encodeURIComponent(id)}`, payload);
}

export function deleteMcpServer(id: string) {
  return deleteRequest(`/settings/mcp/${encodeURIComponent(id)}`);
}

export function listMcpServers() {
  return getJson<any[]>("/settings/mcp");
}

export function getAgentsMarkdown() {
  return getJson<{ content?: string }>("/settings/agents-md");
}

export function saveAgentsMarkdown(content: string) {
  return postJson<any>("/settings/agents-md", { content });
}

export function getGeneralSettings() {
  return getJson<any>("/settings/general");
}

export function saveGeneralSettings(payload: unknown) {
  return postJson<any>("/settings/general", payload);
}

export function getRiskSettings() {
  return getJson<any>("/settings/risk");
}

export function saveRiskSettings(payload: unknown) {
  return postJson<any>("/settings/risk", payload);
}

export function getDatabaseSettings() {
  return getJson<any>("/settings/database");
}
