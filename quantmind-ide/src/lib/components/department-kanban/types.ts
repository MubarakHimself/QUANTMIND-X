/**
 * Department Kanban Types
 *
 * Type definitions for the Department Kanban sub-page UI.
 */

export type TaskStatus = 'TODO' | 'IN_PROGRESS' | 'BLOCKED' | 'DONE';

export type TaskPriority = 'HIGH' | 'MEDIUM' | 'LOW';

export type DepartmentName = 'research' | 'development' | 'risk' | 'trading' | 'portfolio' | 'flowforge' | 'shared-assets';

export interface DepartmentTask {
  task_id: string;
  task_name: string;
  department: DepartmentName;
  priority: TaskPriority;
  status: TaskStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface TaskUpdate {
  task_id: string;
  status: TaskStatus;
  timestamp: string;
}

export interface DepartmentTasksResponse {
  department: DepartmentName;
  tasks: DepartmentTask[];
}