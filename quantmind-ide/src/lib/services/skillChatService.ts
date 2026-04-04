/**
 * Skill Chat Service - Handles skill invocation from chat messages
 *
 * This service provides:
 * - Parsing skill mentions (@skill_name) from chat messages
 * - Skill suggestions based on current agent and context
 * - Skill execution with chat message formatting
 */

import { skillRegistry, executeSkill, type Skill, type SkillResult, type SkillContext } from '../agents/skills';
import type { AgentType } from '../stores/chatStore';

// ============================================================================
// TYPES
// ============================================================================

/**
 * Parsed skill invocation from a message
 */
export interface ParsedSkillInvocation {
  skillId: string;
  skillName: string;
  params: Record<string, any>;
  rawInvocation: string;
}

/**
 * Skill suggestion for UI display
 */
export interface SkillSuggestion {
  id: string;
  name: string;
  description: string;
  category: string;
  icon?: string;
}

/**
 * Skill message metadata for chat
 */
export interface SkillMessageMetadata {
  type: 'skill_invocation' | 'skill_result';
  skillId: string;
  skillName: string;
  params?: Record<string, any>;
  result?: SkillResult;
  executionTime?: number;
  [key: string]: unknown;
}

// ============================================================================
// REGEX PATTERNS
// ============================================================================

// Match @skill_name or @skill_name(param1=value1, param2=value2)
const SKILL_INVOCATION_REGEX = /@(\w+)(?:\(([^)]+)\))?/g;

// Match parameter values: param=value or param="string value"
const PARAM_REGEX = /(\w+)=(?:(?:"([^"]*)")|(\d+\.?\d*)|(\w+))/g;

// ============================================================================
// SERVICE IMPLEMENTATION
// ============================================================================

class SkillChatService {
  /**
   * Parse all skill invocations from a message
   * Supports formats:
   * - @skill_name
   * - @skill_name(param1=value1)
   * - @skill_name(param1=value1, param2=value2)
   * - @skill_name(param1="string value")
   */
  parseSkillInvocations(message: string): ParsedSkillInvocation[] {
    const invocations: ParsedSkillInvocation[] = [];
    const regex = new RegExp(SKILL_INVOCATION_REGEX.source, 'g');

    let match;
    while ((match = regex.exec(message)) !== null) {
      const skillId = match[1];
      const paramString = match[2];

      const params = this.parseParams(paramString || '');

      invocations.push({
        skillId,
        skillName: skillId.replace(/_/g, ' '),
        params,
        rawInvocation: match[0]
      });
    }

    return invocations;
  }

  /**
   * Parse parameters from parameter string
   */
  private parseParams(paramString: string): Record<string, any> {
    if (!paramString.trim()) {
      return {};
    }

    const params: Record<string, any> = {};
    const regex = new RegExp(PARAM_REGEX.source, 'g');

    let match;
    while ((match = regex.exec(paramString)) !== null) {
      const paramName = match[1];
      const stringValue = match[2];
      const numberValue = match[3];
      const booleanValue = match[4];

      if (stringValue !== undefined) {
        params[paramName] = stringValue;
      } else if (numberValue !== undefined) {
        params[paramName] = parseFloat(numberValue);
      } else if (booleanValue !== undefined) {
        params[paramName] = booleanValue === 'true';
      }
    }

    return params;
  }

  /**
   * Check if a message contains skill invocations
   */
  hasSkillInvocations(message: string): boolean {
    const invocations = this.parseSkillInvocations(message);
    return invocations.length > 0;
  }

  /**
   * Extract the skill ID being invoked from the start of a filter string
   * Used for skill suggestion filtering
   */
  extractSkillFilter(text: string): string {
    if (text.startsWith('@')) {
      return text.slice(1).toLowerCase();
    }
    return text.toLowerCase();
  }

  /**
   * Get skill suggestions based on agent type and optional filter
   */
  getSuggestions(agent: AgentType, filter: string = ''): SkillSuggestion[] {
    let skills: Skill[];

    if (filter) {
      // Search skills by name/description
      skills = skillRegistry.search(filter);
    } else {
      // Get all enabled skills for the agent
      skills = skillRegistry.getByAgent(agent);
    }

    // Map to suggestion format
    return skills.map(skill => ({
      id: skill.id,
      name: skill.id.replace(/_/g, ' '),
      description: skill.description,
      category: skill.category || 'general'
    }));
  }

  /**
   * Execute a skill and return formatted result for chat
   */
  async executeSkill(
    skillId: string,
    params: Record<string, any>,
    context: SkillContext
  ): Promise<{ success: boolean; content: string; metadata: SkillMessageMetadata }> {
    const startTime = performance.now();

    try {
      const result = await executeSkill(skillId, params, context);
      const executionTime = performance.now() - startTime;

      if (result.success) {
        const content = this.formatSkillResult(result);
        return {
          success: true,
          content,
          metadata: {
            type: 'skill_result',
            skillId,
            skillName: skillId.replace(/_/g, ' '),
            params,
            result,
            executionTime
          }
        };
      } else {
        return {
          success: false,
          content: `Skill execution failed: ${result.error}`,
          metadata: {
            type: 'skill_result',
            skillId,
            skillName: skillId.replace(/_/g, ' '),
            params,
            result,
            executionTime
          }
        };
      }
    } catch (error: any) {
      const executionTime = performance.now() - startTime;
      return {
        success: false,
        content: `Error executing skill: ${error.message}`,
        metadata: {
          type: 'skill_result',
          skillId,
          skillName: skillId.replace(/_/g, ' '),
          params,
          executionTime
        }
      };
    }
  }

  /**
   * Execute all skill invocations in a message and return results
   */
  async processMessageSkills(
    message: string,
    context: SkillContext
  ): Promise<{
    hasSkills: boolean;
    messageWithoutSkills: string;
    results: Array<{ invocation: ParsedSkillInvocation; result: { success: boolean; content: string; metadata: SkillMessageMetadata } }>;
  }> {
    const invocations = this.parseSkillInvocations(message);

    if (invocations.length === 0) {
      return {
        hasSkills: false,
        messageWithoutSkills: message,
        results: []
      };
    }

    // Remove skill invocations from message text
    let messageWithoutSkills = message;
    const results: Array<{ invocation: ParsedSkillInvocation; result: any }> = [];

    for (const invocation of invocations) {
      // Execute each skill
      const result = await this.executeSkill(
        invocation.skillId,
        invocation.params,
        context
      );

      results.push({ invocation, result });

      // Remove the invocation from the message text
      messageWithoutSkills = messageWithoutSkills.replace(invocation.rawInvocation, '');
    }

    // Clean up extra whitespace
    messageWithoutSkills = messageWithoutSkills.replace(/\s+/g, ' ').trim();

    return {
      hasSkills: true,
      messageWithoutSkills,
      results
    };
  }

  /**
   * Format skill result for chat display
   */
  private formatSkillResult(result: SkillResult): string {
    if (!result.success) {
      return `Error: ${result.error}`;
    }

    const data = result.data;

    // Format based on data type
    if (data === null || data === undefined) {
      return 'Skill executed successfully (no output)';
    }

    if (typeof data === 'string') {
      return data;
    }

    if (typeof data === 'object') {
      // Format as JSON with some美化
      try {
        return JSON.stringify(data, null, 2);
      } catch {
        return String(data);
      }
    }

    return String(data);
  }

  /**
   * Get skill info for display
   */
  getSkillInfo(skillId: string): Skill | undefined {
    return skillRegistry.get(skillId);
  }

  /**
   * Get all available skills for an agent
   */
  getAvailableSkills(agent: AgentType): SkillSuggestion[] {
    return this.getSuggestions(agent);
  }

  /**
   * Check if a skill exists
   */
  skillExists(skillId: string): boolean {
    return skillRegistry.get(skillId) !== undefined;
  }

  /**
   * Get the skill ID from a mention (with fuzzy matching)
   */
  resolveSkillId(mention: string): string | null {
    // Remove @ prefix if present
    const searchName = mention.startsWith('@') ? mention.slice(1) : mention;

    // Try exact match first
    if (skillRegistry.get(searchName)) {
      return searchName;
    }

    // Try fuzzy matching (replace spaces with underscores)
    const withUnderscore = searchName.replace(/\s+/g, '_');
    if (skillRegistry.get(withUnderscore)) {
      return withUnderscore;
    }

    // Search by name/description
    const results = skillRegistry.search(searchName);
    if (results.length > 0) {
      return results[0].id;
    }

    return null;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export const skillChatService = new SkillChatService();

export default skillChatService;
