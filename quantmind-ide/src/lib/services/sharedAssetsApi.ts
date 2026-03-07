import { deleteRequest, getJson, postJson } from "$lib/services/componentApi";

export interface SharedAsset {
  id: string;
  name: string;
  category: "Indicator" | "Risk" | "Utils";
  version: string;
  filesystem_path: string;
  dependencies: string[];
  checksum: string;
  created_by: "QuantCode" | "user";
  used_by_count: number;
  created_at: string;
  updated_at: string;
  description?: string;
}

export interface AssetHistory {
  version: string;
  checksum: string;
  created_at: string;
  created_by: "QuantCode" | "user";
  change_description: string;
}

export function listSharedAssets() {
  return getJson<SharedAsset[]>("/assets/shared");
}

export function createSharedAsset(payload: {
  name: string;
  category: "Indicator" | "Risk" | "Utils";
  code: string;
  description: string;
  dependencies: string[];
}) {
  return postJson<SharedAsset>("/assets", payload);
}

export function deleteSharedAsset(assetId: string) {
  return deleteRequest(`/assets/${encodeURIComponent(assetId)}`);
}

export function getSharedAssetHistory(assetId: string) {
  return getJson<AssetHistory[]>(`/assets/${encodeURIComponent(assetId)}/history`);
}

export function rollbackSharedAsset(assetId: string, version: string) {
  return postJson<SharedAsset>(`/assets/${encodeURIComponent(assetId)}/rollback`, { version });
}
