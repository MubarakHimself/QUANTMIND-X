/**
 * Skills API Client
 * Provides frontend API functions for skill management
 *
 * Connects to the backend endpoints defined in src/api/skills_endpoints.py (Story 7.4)
 */

const API_BASE = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
}

// =============================================================================
// Types
// =============================================================================

export interface Skill {
  id?: string;
  name: string;
  description: string;
  slash_command: string;
  version: string;
  usage_count: number;
  category?: string;
  departments?: string[];
}

export interface SkillInfo extends Skill {
  parameters?: Record<string, any>;
  returns?: Record<string, any>;
  requires?: string[];
  tags?: string[];
}

export interface SkillCreateRequest {
  name: string;
  description: string;
  category?: string;
  departments?: string[];
  parameters?: Record<string, any>;
  returns?: Record<string, any>;
  tags?: string[];
  version?: string;
}

export interface SkillAuthoringRequest {
  name: string;
  description: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  sop_steps: string[];
  category?: string;
  departments?: string[];
  version?: string;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Get all registered skills
 * GET /api/skills
 */
export async function listSkills(department?: string): Promise<Skill[]> {
  const params = department ? `?department=${encodeURIComponent(department)}` : '';
  return apiFetch<Skill[]>(`/skills${params}`);
}

/**
 * Get detailed information about a specific skill
 * GET /api/skills/{skill_name}
 */
export async function getSkillInfo(skillName: string): Promise<SkillInfo> {
  return apiFetch<SkillInfo>(`/skills/${skillName}`);
}

/**
 * Create a new skill
 * POST /api/skills
 */
export async function createSkill(skill: SkillCreateRequest): Promise<Skill> {
  return apiFetch<Skill>('/skills', {
    method: 'POST',
    body: JSON.stringify(skill)
  });
}

/**
 * Skill Forge: Author a new skill
 * POST /api/skills/authoring
 */
export async function skillForgeAuthoring(request: SkillAuthoringRequest): Promise<{
  name: string;
  skill_md_path: string;
  status: string;
}> {
  return apiFetch<{
    name: string;
    skill_md_path: string;
    status: string;
  }>('/skills/authoring', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Execute a skill with given parameters
 * POST /api/skills/{skill_name}/execute
 */
export async function executeSkill(
  skillName: string,
  parameters: Record<string, any> = {}
): Promise<{ success: boolean; data: any; error?: string; execution_time_ms: number }> {
  return apiFetch<{ success: boolean; data: any; error?: string; execution_time_ms: number }>(
    `/skills/${skillName}/execute`,
    {
      method: 'POST',
      body: JSON.stringify({ parameters })
    }
  );
}
