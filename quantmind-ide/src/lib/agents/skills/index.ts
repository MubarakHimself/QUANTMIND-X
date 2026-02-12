/**
 * QuantMindX Skills System
 *
 * Following the Anthropic Skills format:
 * https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md
 *
 * A skill is a reusable capability that agents can use to perform specific tasks.
 * Each skill has:
 * - name: Unique identifier
 * - description: What the skill does
 * - parameters: Input schema (Zod)
 * - execute: Async function implementing the skill
 * - examples: Usage examples for the AI
 */

import { z } from 'zod';
import type { AgentType } from '../langchainAgent';
import { copilotSkills } from './copilotSkills';
import { analystSkills } from './analystSkills';
import { quantcodeSkills } from './quantcodeSkills';

// ============================================================================
// SKILL TYPES
// ============================================================================

/**
 * Skill parameter schema using Zod
 */
export type SkillSchema = z.ZodType<any, z.ZodTypeDef, any>;

/**
 * Skill execution result
 */
export interface SkillResult {
  success: boolean;
  data?: any;
  error?: string;
  metadata?: Record<string, any>;
}

/**
 * Skill definition following Anthropic Skills format
 */
export interface Skill {
  // Unique identifier for the skill
  id: string;

  // Human-readable name
  name: string;

  // Detailed description of what the skill does
  description: string;

  // Which agent(s) can use this skill
  agents: AgentType[];

  // Zod schema for input validation
  schema: SkillSchema;

  // Execution function
  execute: (params: any, context?: SkillContext) => Promise<SkillResult>;

  // Usage examples for AI understanding
  examples: SkillExample[];

  // Whether this skill is enabled by default
  defaultEnabled?: boolean;

  // Optional: Required API keys or services
  requirements?: string[];

  // Optional: Category for organization
  category?: string;
}

/**
 * Usage example for AI to understand how to use the skill
 */
export interface SkillExample {
  input: Record<string, any>;
  output: string;
  description?: string;
}

/**
 * Context provided to skill execution
 */
export interface SkillContext {
  agentType: AgentType;
  sessionId?: string;
  metadata?: Record<string, any>;
}

// ============================================================================
// SKILL REGISTRY
// ============================================================================

/**
 * Central registry for all skills
 */
class SkillRegistry {
  private skills: Map<string, Skill> = new Map();
  private enabledSkills: Set<string> = new Set();

  /**
   * Register a new skill
   */
  register(skill: Skill): void {
    if (this.skills.has(skill.id)) {
      throw new Error(`Skill ${skill.id} is already registered`);
    }

    this.skills.set(skill.id, skill);

    // Enable by default if specified
    if (skill.defaultEnabled !== false) {
      this.enabledSkills.add(skill.id);
    }
  }

  /**
   * Get a skill by ID
   */
  get(id: string): Skill | undefined {
    return this.skills.get(id);
  }

  /**
   * Get all skills for a specific agent
   */
  getByAgent(agent: AgentType): Skill[] {
    return Array.from(this.skills.values()).filter(
      skill => skill.agents.includes(agent) && this.enabledSkills.has(skill.id)
    );
  }

  /**
   * Get all skills
   */
  getAll(): Skill[] {
    return Array.from(this.skills.values());
  }

  /**
   * Get all enabled skills
   */
  getAllEnabled(): Skill[] {
    return Array.from(this.skills.values()).filter(skill =>
      this.enabledSkills.has(skill.id)
    );
  }

  /**
   * Enable or disable a skill
   */
  setEnabled(id: string, enabled: boolean): boolean {
    if (!this.skills.has(id)) {
      return false;
    }

    if (enabled) {
      this.enabledSkills.add(id);
    } else {
      this.enabledSkills.delete(id);
    }

    return true;
  }

  /**
   * Check if a skill is enabled
   */
  isEnabled(id: string): boolean {
    return this.enabledSkills.has(id);
  }

  /**
   * Get skills by category
   */
  getByCategory(category: string): Skill[] {
    return Array.from(this.skills.values()).filter(
      skill => skill.category === category && this.enabledSkills.has(skill.id)
    );
  }

  /**
   * Search skills by query
   */
  search(query: string): Skill[] {
    const lowerQuery = query.toLowerCase();
    return Array.from(this.skills.values()).filter(skill =>
      this.enabledSkills.has(skill.id) && (
        skill.name.toLowerCase().includes(lowerQuery) ||
        skill.description.toLowerCase().includes(lowerQuery) ||
        skill.id.toLowerCase().includes(lowerQuery)
      )
    );
  }

  /**
   * Export skills configuration
   */
  exportConfig(): Array<{ id: string; enabled: boolean }> {
    return Array.from(this.skills.values()).map(skill => ({
      id: skill.id,
      enabled: this.enabledSkills.has(skill.id)
    }));
  }

  /**
   * Import skills configuration
   */
  importConfig(config: Array<{ id: string; enabled: boolean }>): void {
    for (const { id, enabled } of config) {
      if (this.skills.has(id)) {
        if (enabled) {
          this.enabledSkills.add(id);
        } else {
          this.enabledSkills.delete(id);
        }
      }
    }
  }

  /**
   * Clear all skills (useful for testing)
   */
  clear(): void {
    this.skills.clear();
    this.enabledSkills.clear();
  }
}

// ============================================================================
// GLOBAL REGISTRY INSTANCE
// ============================================================================

export const skillRegistry = new SkillRegistry();

// ============================================================================
// SKILL EXECUTION HELPERS
// ============================================================================

/**
 * Execute a skill by ID with validation
 */
export async function executeSkill(
  skillId: string,
  params: any,
  context?: SkillContext
): Promise<SkillResult> {
  const skill = skillRegistry.get(skillId);

  if (!skill) {
    return {
      success: false,
      error: `Skill ${skillId} not found`
    };
  }

  if (!skillRegistry.isEnabled(skillId)) {
    return {
      success: false,
      error: `Skill ${skillId} is disabled`
    };
  }

  try {
    // Validate input
    const validatedParams = skill.schema.parse(params);

    // Execute skill
    const result = await skill.execute(validatedParams, context);

    return result;
  } catch (error: any) {
    return {
      success: false,
      error: error.message || 'Unknown error',
      metadata: { originalError: error }
    };
  }
}

/**
 * Get skills as LangChain tools
 */
export function getSkillsAsTools(agent: AgentType) {
  const skills = skillRegistry.getByAgent(agent);

  return skills.map(skill => ({
    name: skill.id,
    description: skill.description,
    schema: skill.schema,
    func: async (params: any) => {
      const result = await executeSkill(skill.id, params, { agent });
      if (result.success) {
        return JSON.stringify(result.data);
      } else {
        return `Error: ${result.error}`;
      }
    }
  }));
}

// ============================================================================
// INITIALIZE ALL SKILLS
// ============================================================================

/**
 * Initialize all skills on import
 */
function initializeSkills() {
  // Register Copilot skills
  for (const skill of copilotSkills) {
    skillRegistry.register(skill);
  }

  // Register Analyst skills
  for (const skill of analystSkills) {
    skillRegistry.register(skill);
  }

  // Register QuantCode skills
  for (const skill of quantcodeSkills) {
    skillRegistry.register(skill);
  }
}

// Auto-initialize on module load
initializeSkills();

// ============================================================================
// RE-EXPORTS
// ============================================================================

export { copilotSkills, analystSkills, quantcodeSkills };
export type { AgentType } from '../langchainAgent';
